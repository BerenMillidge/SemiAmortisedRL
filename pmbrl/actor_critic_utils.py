import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def imagine_forward(state, dynamics_model, reward_model, actor, critic,plan_horizon,use_mean=True):
  ensemble_size, batch_size, state_size = state.size()
  device = dynamics_model.device
  action_size = actor.action_dim
  T = plan_horizon
  pred_states = [torch.empty(0)] * (T+1)
  actions = [torch.empty(0)] * T
  pred_rewards = [torch.empty(0)] * T
  pred_values = [torch.empty(0)] * T
  pred_states[0] = state

  for t in range(plan_horizon):
    a_mu, a_var = actor(pred_states[t])
    #print("a_mu", a_mu.mean().item(),a_mu.var().item(),a_mu.min().item(), a_mu.max().item())
    #print("a_var", a_var.mean().item(),a_var.var().item(),a_var.min().item(), a_var.max().item())
    #a = a_mu + (a_var * torch.zeros(ensemble_size, batch_size, action_size, dtype=torch.float32,device=device).normal_() )# not sure this is necessarily differentiable, and it needs to be, unlike in planner!
    a = a_mu # we only do action mean now because I think the variance was potentially messing it up and I want to get the simplest possible case working first
    actions[t] = a
    delta_mean, delta_var = dynamics_model(pred_states[t], actions[t])
    if use_mean:
        pred_states[t + 1] = pred_states[t] + delta_mean
    else:
        pred_states[t + 1] = pred_states[t] + self.ensemble.sample(delta_mean, delta_var)
    pred_rewards[t] = reward_model(pred_states[t])
    mean_values, var_values = critic(pred_states[t])
    pred_values[t] = mean_values
  pred_states = torch.stack(pred_states)[1:]
  pred_rewards = torch.stack(pred_rewards)
  pred_values = torch.stack(pred_values)
  actions = torch.stack(actions)
  return pred_states, pred_rewards, pred_values, actions

def lambda_return(rewards, states,critic):
  raise NotImplementedError("Cannot figure out precisely what this is")

def MonteCarloReturn(rewards,gamma,batch_size,state_size):
  T = rewards.shape[0]
  discount = np.array([gamma**i for i in range(T)])
  discount = torch.from_numpy(discount).unsqueeze(1).unsqueeze(1)
  discount = discount.repeat(1,batch_size, state_size)
  return torch.sum(discount * rewards,dim=0)

def create_discount_matrix(args):
  T = args.projection_horizon
  discount = np.array([args.gamma**i for i in range(T)])
  discount = torch.from_numpy(discount).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(args.DEVICE)
  discount = discount.repeat(1,args.ensemble_size,args.batch_size, 1)
  return discount


def Q_returns(rewards, values,args):
  T = rewards.shape[0]
  zeros = torch.zeros_like(rewards,device=rewards.device)
  zeros[0:T-1,:,:,:] = args.DISCOUNT_MATRIX[1:T,:,:,:] * values[1:T,:,:,:]
  returns = rewards + zeros
  return returns

def actor_critic_loss(state,dynamics_model, reward_model,actor,critic,args,verbose=False):
  pred_states, pred_rewards, pred_values, actions = imagine_forward(state, dynamics_model,reward_model,actor,critic,args.projection_horizon)
  #returns = MonteCarloReturn(rewards,gamma, batch_size, state_size)
  returns = Q_returns(pred_rewards, pred_values,args)
  actor_loss = -torch.mean(returns)
  if verbose:
    print("pred_states", pred_states.mean().item(),pred_states.var().item(),pred_states.min().item(), pred_states.max().item())
    print("pred_rewards", pred_rewards.mean().item(),pred_rewards.var().item(),pred_rewards.min().item(), pred_rewards.max().item())
    print("pred_values", pred_values.mean().item(),pred_values.var().item(),pred_values.min().item(), pred_values.max().item())
    print("actions", actions.mean().item(),actions.var().item(),actions.min().item(), actions.max().item())
    print("returns", returns.mean().item(),returns.var().item(),returns.min().item(), returns.max().item())

  value_loss = F.mse_loss(pred_values, returns.detach())
  return actor_loss, value_loss

def train_actor_critic(state, dynamics_model, reward_model,actor,critic,planning_horizon,gamma,projection_horizon):
  actor_opt = torch.optim.Adam(lr=1e-4)
  critic_opt = torch.optim.Adam(lr=1e-4)
  actor_loss, critic_loss = actor_critic_loss(state, dynamics_model, reward_model, actor,critic,gamma,projection_horizon)
  actor_opt.zero_grad()
  actor_opt.backwards()
  actor_loss.backward()
  critic_loss.backward()
  actor_opt.step()
  critic_opt.step()
