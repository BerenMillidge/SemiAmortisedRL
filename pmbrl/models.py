# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

SWISH = "swish"
LEAKY_RELU = "leaky_relu"
RELU = "relu"
TANH = "tanh"
LINEAR = "linear"


def swish(x):
    return x * torch.sigmoid(x)


class EnsembleDenseLayer(nn.Module):
    def __init__(self, in_size, out_size, ensemble_size, non_linearity=SWISH):
        super().__init__()

        weights = torch.zeros(ensemble_size, in_size, out_size).float()
        biases = torch.zeros(ensemble_size, 1, out_size).float()

        for weight in weights:
            if non_linearity == SWISH:
                nn.init.xavier_uniform_(weight)
            elif non_linearity == LEAKY_RELU:
                nn.init.kaiming_normal_(weight)
            elif non_linearity == TANH:
                nn.init.kaiming_normal_(weight)
            elif non_linearity == LINEAR:
                nn.init.xavier_normal_(weight)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

        if non_linearity == SWISH:
            self.non_linearity = swish
        elif non_linearity == LEAKY_RELU:
            self.non_linearity = F.leaky_relu
        elif non_linearity == TANH:
            self.non_linearity = torch.tanh
        elif non_linearity == LINEAR:
            self.non_linearity = lambda x: x

    def forward(self, x):
        op = torch.baddbmm(self.biases, x, self.weights)
        op = self.non_linearity(op)
        return op

    def reset_parameters(self):
        weights = torch.zeros(self.ensemble_size, self.in_size, self.out_size).float()
        biases = torch.zeros(self.ensemble_size, 1, self.out_size).float()

        for weight in weights:
            self._init_weight(weight, self.act_fn_name)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)


class EnsembleModel(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        ensemble_size,
        normalizer,
        non_linearity=SWISH,
        device="cpu",
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.device = device

        self.fc_1 = EnsembleDenseLayer(
            in_size, hidden_size, ensemble_size, non_linearity=non_linearity
        )
        self.fc_2 = EnsembleDenseLayer(
            hidden_size, hidden_size, ensemble_size, non_linearity=non_linearity
        )
        self.fc_3 = EnsembleDenseLayer(
            hidden_size, hidden_size, ensemble_size, non_linearity=non_linearity
        )
        self.fc_4 = EnsembleDenseLayer(
            hidden_size, out_size * 2, ensemble_size, non_linearity=LINEAR
        )

        self.normalizer = normalizer

        self.max_logvar = -1
        self.min_logvar = -5

        self.to(device)

    def _pre_process_model_inputs(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)

        if self.normalizer is None:
            return states, actions

        states = self.normalizer.normalize_states(states)
        actions = self.normalizer.normalize_actions(actions)
        return states, actions

    def _pre_process_model_targets(self, state_deltas):
        state_deltas = state_deltas.to(self.device)

        if self.normalizer is None:
            return state_deltas

        state_deltas = self.normalizer.normalize_state_deltas(state_deltas)
        return state_deltas

    def _post_process_model_outputs(self, delta_mean, delta_var):
        if self.normalizer is not None:
            delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
            delta_var = self.normalizer.denormalize_state_delta_vars(delta_var)
        return delta_mean, delta_var

    def _propagate_network(self, states, actions):
        inp = torch.cat((states, actions), dim=2)

        op = self.fc_1(inp)
        op = self.fc_2(op)
        op = self.fc_3(op)
        op = self.fc_4(op)

        delta_mean, delta_logvar = torch.split(op, op.size(2) // 2, dim=2)
        delta_logvar = torch.sigmoid(delta_logvar)
        delta_logvar = (
            self.min_logvar + (self.max_logvar - self.min_logvar) * delta_logvar
        )
        delta_var = torch.exp(delta_logvar)

        return delta_mean, delta_var

    def forward(self, states, actions):
        normalized_states, normalized_actions = self._pre_process_model_inputs(
            states, actions
        )

        normalized_delta_mean, normalized_delta_var = self._propagate_network(
            normalized_states, normalized_actions
        )

        delta_mean, delta_var = self._post_process_model_outputs(
            normalized_delta_mean, normalized_delta_var
        )

        return delta_mean, delta_var

    def sample(self, mean, var):
        return Normal(mean, torch.sqrt(var)).sample()

    def loss(self, states, actions, state_deltas):
        states, actions = self._pre_process_model_inputs(states, actions)
        delta_targets = self._pre_process_model_targets(state_deltas)
        delta_mu, delta_var = self._propagate_network(states, actions)
        loss = (delta_mu - delta_targets) ** 2 / delta_var + torch.log(delta_var)
        loss = loss.mean(-1).mean(-1).sum()
        return loss

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def reset_parameters(self):
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()
        self.fc_4.reset_parameters()
        self.to(self.device)


class RewardModel(nn.Module):
    def __init__(self, state_size, hidden_size, act_fn=RELU,device="cpu"):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.ensemble_reward_model = False
        self.device = device
        self.to(self.device)

    def forward(self, state):
        reward = self.act_fn(self.fc1(state))
        reward = self.act_fn(self.fc2(reward))
        reward = self.fc3(reward).squeeze(dim=1)
        return reward

    def loss(self, states, rewards):
        r_hat = self(states)
        return F.mse_loss(r_hat, rewards)

    def reset_parameters(self):
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.to(self.device)


class EnsembleRewardModel(nn.Module):
    def __init__(self, state_size, hidden_size, ensemble_size, non_linearity="swish", device="cpu",reward_ensemble_per_transition_ensemble=False):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.ensemble_size = ensemble_size
        self.non_linearity=  non_linearity
        self.ensemble_reward_model = True
        self.device=  device

        self.fc1 = EnsembleDenseLayer(self.state_size, self.hidden_size, self.ensemble_size, non_linearity=self.non_linearity)
        self.fc2 = EnsembleDenseLayer(self.hidden_size, self.hidden_size, self.ensemble_size, non_linearity=self.non_linearity)
        self.fc3 = EnsembleDenseLayer(self.hidden_size, 1, self.ensemble_size, non_linearity="linear")

        self.reward_ensemble_per_transition_ensemble = reward_ensemble_per_transition_ensemble

    def forward(self,state):
        batch_size = state.size(1)
        if not self.reward_ensemble_per_transition_ensemble:
          reward = self.fc1(state)
          reward = self.fc2(reward)
          reward = self.fc3(reward).squeeze(dim=1)
          return reward  # [N_ensembles, batch_size,1]
        if self.reward_ensemble_per_transition_ensemble:
          state = state.unsqueeze(0).repeat(self.ensemble_size,1,1,1) # repeat for ensemble_size
          result = torch.empty([self.ensemble_size, self.ensemble_size, batch_size,1]).to(self.device)
          for i in range(self.ensemble_size):
            s = state[:,i,:,:]
            reward = self.fc1(s)
            reward = self.fc2(reward)
            reward = self.fc3(reward).squeeze(dim=1)
            result[i,:,:,:] = reward
          return result #[N_ensmbles (reward), N_ensembles (t_model), batch_size,1]
    def loss(self, states, rewards):
        r_hat = self(states)
        if self.reward_ensemble_per_transition_ensemble:
          rewards = rewards.unsqueeze(0).repeat(self.ensemble_size,1,1,1) #tile for number of ensembles: [Ensemble_size (R_model),Ensemble_size (t_model),batch_size, 1]
        return F.mse_loss(r_hat, rewards) #check if this means correctly over reward ensemble dim

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.to(self.device)

# let's output an action variance too. This will be aleatoric and aid inproducing more varied trajectories for info gain, which would be great
class ActionModel(nn.Module):
  def __init__(self, in_size, hidden_size,action_dim,act_fn = F.elu, device="cpu"):
    super().__init__()
    self.in_size = in_size
    self.hidden_size = hidden_size
    self.action_dim = action_dim
    self.act_fn = act_fn
    self.device = device
    self.reset_parameters()
    self.mean_scale = 1
    self.mean_add = 0
    self.max_logvar = -1
    self.min_logvar = -5
    self.to(self.device)

  def forward(self, states):
    # states = states.detach()
    act = self.act_fn(self.fc1(states))
    act = self.act_fn(self.fc2(act))
    act = self.act_fn(self.fc3(act))
    # potentially scale mean and variance
    act = self.fc4(act)
    mean, logvar = torch.split(act, self.action_dim,dim=2)
    mean = torch.tanh(mean) * 2.0 # to keep it arbitrarily within the boundaries. Instead need to add a max/min action to auto scale it otherwise in likely trouble!
    logvar = self.min_logvar + (self.max_logvar - self.min_logvar) * torch.sigmoid(logvar)
    return mean, torch.exp(logvar)

  def detach_forward(self, states):
    states = states.detach()
    return self(states)

  def from_numpy_forward(self, state):
    state  = torch.from_numpy(state).float().to(self.device)
    state = state.unsqueeze(0).unsqueeze(0)
    return self(state)

  def reset_parameters(self):
    self.fc1 = nn.Linear(self.in_size, self.hidden_size)
    self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
    self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
    self.fc4 = nn.Linear(self.hidden_size, self.action_dim * 2)
    self.to(self.device)

class ValueModel(nn.Module):
  def __init__(self, in_size, hidden_size, act_fn = F.elu, device="cpu"):
    super().__init__()
    self.in_size = in_size
    self.hidden_size = hidden_size
    self.act_fn = act_fn
    self.device = device
    self.mean_scale = 1
    self.mean_add = 0
    self.std_epsilon = 0.01
    self.min_std = 1e-5
    self.reset_parameters()
    self.to(self.device)

  def forward(self, states):
    value = self.act_fn(self.fc1(states))
    value = self.act_fn(self.fc2(value))
    value = self.fc3(value)
    mean, var = torch.split(value,1,dim=2)
    #mean = (torch.tanh(mean) * self.mean_scale) + self.mean_add
    var =F.softplus(var + self.std_epsilon) + self.min_std
    return mean, var

  def reset_parameters(self):
    self.fc1 = nn.Linear(self.in_size, self.hidden_size)
    self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
    self.fc3 = nn.Linear(self.hidden_size, 2)
    self.to(self.device)
