import gym
import torch
import numpy as np
import argparse
import time
import subprocess
from datetime import datetime

#from pmbrl.env import GymEnv, NoisyEnv
from pmbrl.normalizer import TransitionNormalizer
from pmbrl.buffer import Buffer
from pmbrl.models import EnsembleModel, RewardModel, EnsembleRewardModel, ActionModel, ValueModel
from pmbrl.planner import CEMPlanner, PIPlanner, RandomShootingPlanner
from pmbrl.agent import Agent
from pmbrl import tools
import pmbrl.actor_critic_utils as utils
try:
    import pybullet
except:
    print("Pybullet not installed. Oh well.")
    pass
try:
    import pybullet_envs
except:
    print("pybullet_envs not installed. Oh well.")
    pass

from baselines.envs import TorchEnv, NoisyEnv, const



def main(args):
    tools.log(" === Loading experiment ===")
    tools.log(args)
    args.save_model = True
    print("N SEED EPISODES: ", args.n_seed_episodes)
    print("ARGS RENDER: ", args.render)
    print("ARGS SAVE MODEL: ", args.save_model)
    print("ARGS EXPLORATION: ", args.use_exploration)
    # terible global, but oh well...

    if args.projection_horizon == -1:
        args.projection_horizon = args.plan_horizon # set this up as the default

    args.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.DISCOUNT_MATRIX = utils.create_discount_matrix(args)
    DEVICE = args.DEVICE
    if args.env_name == "SparseHalfCheetah" or args.env_name == "SparseCartpoleSwingup":
        try:
            import roboschool
        except:
            raise Exception("Cannot use these environments without roboschool, which failed to import")


    env = TorchEnv(args.env_name, args.max_episode_len, action_repeat=args.action_repeat, device=DEVICE)
    state_size = env.state_dims[0]
    action_size = env.action_dims[0]

    if args.save_path != "":
        subprocess.call(["mkdir","-p",str(args.save_path)])

    if args.env_std > 0.0:
        print("USING NOISE ENV!!!")
        env = NoisyEnv(env, args.env_std)

    normalizer = TransitionNormalizer()
    buffer = Buffer(
        state_size,
        action_size,
        args.ensemble_size,
        normalizer,
        buffer_size=args.buffer_size,
        device=DEVICE,
    )

    ensemble = EnsembleModel(
        state_size + action_size,
        state_size,
        args.hidden_size,
        args.ensemble_size,
        normalizer,
        device=DEVICE,
    ).to(DEVICE)
    if args.use_ensemble_reward_model:
        reward_model = EnsembleRewardModel(state_size, args.hidden_size, args.ensemble_size,device=DEVICE)
    else:
        reward_model = RewardModel(state_size, args.hidden_size,device=DEVICE)

    actor = ActionModel(state_size, args.hidden_size,action_size,device=DEVICE)
    critic = ValueModel(state_size, args.hidden_size, device=DEVICE)
    actor_params = list(actor.parameters())
    critic_params =list(critic.parameters())

    params = list(ensemble.parameters()) + list(reward_model.parameters())
    optim = torch.optim.Adam(params, lr=args.learning_rate, eps=args.epsilon)
    actor_optim = torch.optim.Adam(actor_params, lr=args.learning_rate, eps=args.epsilon)
    critic_optim = torch.optim.Adam(critic_params, lr=args.learning_rate, eps=args.epsilon)
    print("USE EXPLORATION: ", args.use_exploration)

    if args.planner == "CEM":
        planner = CEMPlanner(
            ensemble,
            reward_model,
            action_size,
            plan_horizon=args.plan_horizon,
            optimisation_iters=args.optimisation_iters,
            n_candidates=args.n_candidates,
            top_candidates=args.top_candidates,
            use_exploration=args.use_exploration,
            use_reward=args.use_reward,
            use_reward_info_gain = args.use_reward_info_gain,
            expl_scale=args.expl_scale,
            device=DEVICE,
        ).to(DEVICE)

    if args.planner == "PI":
        planner = PIPlanner(
        ensemble,
        reward_model,
        action_size,
        args.N_smaples,
        args.plan_horizon,
        args.lambda_,
        args.noise_mu,
        args.noise_sigma,
        env,
        args.use_exploration,
        args.use_reward,
        args.use_reward_info_gain,
        device=DEVICE
        )

    agent = Agent(env, planner,actor,args.use_epsilon_greedy, args.epsilon_greedy_value,args.use_actor)
    # I have to make a decision on whether to reset all my models or not, which is going to be really frustrating
    #ugh.


    if tools.logdir_exists(args.logdir) and True == False:
        tools.log("Loading existing _logdir_ at {}".format(args.logdir))
        if args.save_model:
            print("In saver!!!: ", args.save_model)
            try:
                normalizer = tools.load_normalizer(args.logdir)
                buffer = tools.load_buffer(args.logdir, buffer)
                buffer.set_normalizer(normalizer)
                metrics = tools.load_metrics(args.logdir)
                model_dict = tools.load_model_dict(args.logdir, metrics["last_save"])
                ensemble.load_state_dict(model_dict["ensemble"])
                ensemble.set_normalizer(normalizer)
                reward_model.load_state_dict(model_dict["reward"])
                optim.load_state_dict(model_dict["optim"])
            except:
                print("Failed to load metrics")
                metrics = tools.build_metrics()
                buffer = agent.get_seed_episodes(buffer, args.n_seed_episodes,render_flag = args.render)
                message = "Collected seeds: [{} episodes] [{} frames]"
                tools.log(message.format(args.n_seed_episodes, buffer.total_steps))
    else:
        tools.init_dirs(args.logdir)
        metrics = tools.build_metrics()
        buffer = agent.get_seed_episodes(buffer, args.n_seed_episodes,render_flag = args.render)
        message = "Collected seeds: [{} episodes] [{} frames]"
        tools.log(message.format(args.n_seed_episodes, buffer.total_steps))

    traj_dir_name = str(args.logdir) + "/" + str(args.trajectory_savedir)


    if args.collect_trajectories == True:
        tools.init_dirs(traj_dir_name)

    for episode in range(metrics["episode"], args.n_episodes):
        tools.log("\n === Episode {} ===".format(episode))
        start_time_episode = time.process_time()
        start_time_training = time.process_time()

        tools.log("Training on {} data points".format(buffer.total_steps))
        for epoch in range(args.n_train_epochs):
            e_losses = []
            r_losses = []
            a_losses = []
            c_losses = []

            for (states, actions, rewards, delta_states) in buffer.get_train_batches(
                args.batch_size
            ):
                ensemble.train()
                reward_model.train()
                actor.train()
                critic.train()
                optim.zero_grad()

                e_loss = ensemble.loss(states, actions, delta_states)
                r_loss = reward_model.loss(states, rewards)
                e_losses.append(e_loss.item())
                r_losses.append(r_loss.item())
                (e_loss + r_loss).backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm, norm_type=2)
                optim.step()

                if args.use_actor:
                    actor_optim.zero_grad()
                    critic_optim.zero_grad()
                    verbose = True if epoch % 10 == 0 else False
                    actor_loss, critic_loss = utils.actor_critic_loss(states, ensemble, reward_model, actor, critic, args,verbose=verbose)
                    #do actor loss first
                    actor_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(actor_params, args.grad_clip_norm, norm_type=2)
                    actor_optim.step()
                    #critic loss:
                    critic_optim.zero_grad()
                    actor_optim.zero_grad()
                    critic_loss.backward(retain_graph=False)
                    torch.nn.utils.clip_grad_norm_(critic_params, args.grad_clip_norm, norm_type=2)
                    critic_optim.step()
                    a_losses.append(actor_loss.item())
                    c_losses.append(critic_loss.item())




            if epoch % args.log_every == 0 and epoch > 0:
                message = "> Epoch {} [ ensemble {:.2f} | rew {:.2f} | actor {:.2f} | critic {:.2f}]"
                tools.log(
                    message.format(epoch, sum(e_losses) / epoch, sum(r_losses) / epoch, sum(a_losses)/epoch, sum(c_losses)/epoch)
                )
        metrics["ensemble_loss"].append(sum(e_losses))
        metrics["reward_loss"].append(sum(r_losses))
        metrics["actor_loss"].append(sum(a_losses))
        metrics["critic_loss"].append(sum(c_losses))
        message = "Losses: [ensemble {} | reward {} | actor {} | critic {}]"
        tools.log(message.format(sum(e_losses), sum(r_losses),sum(a_losses), sum(c_losses)))
        end_time_training = time.process_time() - start_time_training
        tools.log("Total training time: {:.2f}".format(end_time_training))

        if args.action_noise > 0.0:
            start_time_expl = time.process_time()
            if args.collect_trajectories == True:
                expl_reward, expl_steps, buffer,reward_stats,_,_,trajectories = agent.run_episode(
                    buffer=buffer, action_noise=args.action_noise,render_flag=args.render,collect_trajectories=args.collect_trajectories)
                tools._save_pickle(traj_dir_name+"traj_action_noise"+str(episode)+".npy", trajectories)
            else:
                expl_reward, expl_steps, buffer,reward_stats,_,_ = agent.run_episode(
                buffer=buffer, action_noise=args.action_noise,render_flag=args.render)
            metrics["train_rewards"].append(expl_reward)
            metrics["train_steps"].append(expl_steps)
            metrics["reward_stats"].append(reward_stats)
            message = "Exploration: [reward {:.2f} | steps {:.2f} ]"
            tools.log(message.format(expl_reward, expl_steps))
            end_time_expl = time.process_time() - start_time_expl
            tools.log("Total exploration time: {:.2f}".format(end_time_expl))

        start_time = time.process_time()
        if args.collect_trajectories == True:
            expl_reward, expl_steps, buffer,reward_stats,info_stats,reward_info_stats, trajectories = agent.run_episode(
                buffer=buffer, render_flag=args.render,collect_trajectories=args.collect_trajectories)
            tools._save_pickle(traj_dir_name+"traj_"+str(episode)+".npy", trajectories)
        else:
            expl_reward, expl_steps, buffer,reward_stats,info_stats,reward_info_stats = agent.run_episode(
            buffer=buffer, render_flag=args.render)
        #print("reward_stats: ", reward_stats)
        #print("info_stats: ", info_stats)
        metrics["test_rewards"].append(expl_reward)
        metrics["test_steps"].append(expl_steps)
        metrics["reward_stats"].append(reward_stats)
        metrics["information_stats"].append(info_stats)
        metrics["reward_info_stats"].append(reward_info_stats)
        message = "Exploitation: [reward {:.2f} | steps {:.2f} ]"
        tools.log(message.format(expl_reward, expl_steps))
        end_time = time.process_time() - start_time
        tools.log("Total exploitation time: {:.2f}".format(end_time))

        end_time_episode = time.process_time() - start_time_episode
        tools.log("Total episode time: {:.2f}".format(end_time_episode))
        metrics["episode"] += 1
        metrics["total_steps"].append(buffer.total_steps)
        metrics["episode_time"].append(end_time_episode)

        args.save_every == 1
        if episode % args.save_every == 0:
            metrics["episode"] += 1
            metrics["last_save"] = episode
            if args.save_model:
                tools.save_model(args.logdir, ensemble, reward_model, optim, episode)
                tools.save_normalizer(args.logdir, normalizer)
                tools.save_buffer(args.logdir, buffer)
                tools.save_metrics(args.logdir, metrics)

            # save to actual default file using rsync
            subprocess.call(['rsync','--archive','--update','--compress','--progress',str(args.logdir) + "/",str(args.save_path)])
            tools.log("Rsynced files from: " + str(args.logdir) + "/ " + " to" + str(args.save_path))
            #echo the date so I can straightforwardly see it in the logs if each run is taking an exponentially long time.
            #otherwise will just be super irritating to see what's going wrong here.
            #the pybullet stuff is also SUPER ANNOYING! I need to see if I can fix it somewhere... or rebuild pybullet
            now = datetime.now()
            current_time = str(now.strftime("%H:%M:%S"))

            subprocess.call(['echo', f" TIME OF SAVE: {current_time}"])

            #rsync --archive --update --compress --progress ${log_path}/ ${save_path}

if __name__ == "__main__":

    def boolcheck(x):
        return str(x).lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="test_pendulum")
    parser.add_argument("--env_name", type=str, default="Pendulum-v1")
    parser.add_argument("--max_episode_len", type=int, default=500)
    parser.add_argument("--action_repeat", type=int, default=3)
    parser.add_argument("--env_std", type=float, default=0.00)
    parser.add_argument("--action_noise", type=float, default=0.0)
    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--buffer_size", type=int, default=10 ** 6)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--plan_horizon", type=int, default=30)
    parser.add_argument("--n_candidates", type=int, default=700)
    parser.add_argument("--optimisation_iters", type=int, default=7)
    parser.add_argument("--top_candidates", type=int, default=70)
    parser.add_argument("--n_seed_episodes", type=int, default=20)
    parser.add_argument("--n_train_epochs", type=int, default=100)
    parser.add_argument("--n_episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--grad_clip_norm", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--use_reward", type=boolcheck, default=True)
    parser.add_argument("--use_exploration", type=boolcheck, default=False)
    parser.add_argument("--render", type=boolcheck, default=False)
    parser.add_argument("--expl_scale", type=float, default=1)
    parser.add_argument("--planner", type=str, default="CEM")
    parser.add_argument("--use_ensemble_reward_model", type=boolcheck, default=False)
    parser.add_argument("--use_reward_info_gain", type=boolcheck, default=False)
    parser.add_argument("--collect_trajectories",type=boolcheck, default=False)
    parser.add_argument("--trajectory_savedir", type=str,default="trajectories/")
    parser.add_argument("--save_path", type=str, default="/home/s1686853/default_save")
    parser.add_argument("--use_epsilon_greedy", type=boolcheck, default=False)
    parser.add_argument("--epsilon_greedy_value", type=float, default=0.0)
    # actor-critic arguments
    parser.add_argument("--use_actor", type=boolcheck, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--projection_horizon",type=int,default=-1)



    args = parser.parse_args()
    main(args)


    """
    parser.add_argument("--logdir", type=str, default="log-cheetah")
    parser.add_argument("--env_name", type=str, default="RoboschoolHalfCheetah-v1")
    parser.add_argument("--max_episode_len", type=int, default=10)
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--env_std", type=float, default=0.02)
    parser.add_argument("--action_noise", type=float, default=0.3)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--buffer_size", type=int, default=10 ** 6)
    parser.add_argument("--hidden_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--plan_horizon", type=int, default=2)
    parser.add_argument("--n_candidates", type=int, default=20)
    parser.add_argument("--optimisation_iters", type=int, default=2)
    parser.add_argument("--top_candidates", type=int, default=10)
    parser.add_argument("--n_seed_episodes", type=int, default=5)
    parser.add_argument("--n_train_epochs", type=int, default=10)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--grad_clip_norm", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--use_reward", type=bool, default=True)
    parser.add_argument("--use_exploration", type=bool, default=False)
    parser.add_argument("--expl_scale", type=int, default=1)
    """
