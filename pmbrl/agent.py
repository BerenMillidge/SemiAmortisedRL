# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
import numpy as np
from .tools import average_stats
import gc

from copy import deepcopy


class Agent(object):
    def __init__(self, env, planner,use_epsilon_greedy, epsilon):
        self.env = env
        self.planner = planner
        self.stats_sample_reward = 0.1
        self.use_epsilon_greedy = use_epsilon_greedy
        self.epsilon = epsilon
        self.reward_stats_samples = []
        self.info_stats_samples = []
        self.reward_IG_stats_samples = []

    def get_seed_episodes(self, buffer, n_episodes,render_flag=False,use_epsilon_greedy=False, epsilon=0.0):
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.sample_action()
                next_state, reward, done = self.env.step(action)
                if render_flag:
                    self.env.render()
                buffer.add(state, action, reward, next_state)
                state = deepcopy(next_state)
                if done:
                    break
        return buffer

    def run_episode(self, buffer=None, action_noise=0.0,render_flag = False, collect_trajectories = False):
        total_reward = 0
        total_steps = 0
        done = False
        if collect_trajectories:
            trajectories = []

        with torch.no_grad():
            state = self.env.reset()
            while not done:
                if self.use_epsilon_greedy:
                    rnd = np.random.uniform()
                    if rnd <= self.epsilon:
                        action = self.env.sample_action()
                    else:
                        r = np.random.uniform()
                        if r < self.stats_sample_reward:
                            self.planner.return_stats = True
                            action,reward_stats, info_stats,reward_IG_stats = self.planner(state)
                            self.planner.return_stats = False
                            self.reward_stats_samples.append(reward_stats)
                            self.info_stats_samples.append(info_stats)
                            self.reward_IG_stats_samples.append(reward_IG_stats)
                        else:
                            action = self.planner(state)

                        action = action.cpu().detach().numpy()
                else:
                    r = np.random.uniform()
                    if r < self.stats_sample_reward:
                        self.planner.return_stats = True
                        action,reward_stats, info_stats,reward_IG_stats = self.planner(state)
                        self.planner.return_stats = False
                        self.reward_stats_samples.append(reward_stats)
                        self.info_stats_samples.append(info_stats)
                        self.reward_IG_stats_samples.append(reward_IG_stats)
                    else:
                        action = self.planner(state)

                    action = action.cpu().detach().numpy()

                if action_noise > 0:
                    action = action + np.random.normal(0, action_noise, action.shape)

                next_state, reward, done = self.env.step(action)
                if render_flag:
                    self.env.render()
                total_reward += reward
                total_steps += 1

                if buffer is not None:
                    buffer.add(state, action, reward, next_state)
                if collect_trajectories:
                    trajectories.append(deepcopy(state))
                state = deepcopy(next_state)
                if done:
                    break

        #self.env.close()
        # this may be needed to prevent pybullet messing up every time.
        #fix from https://github.com/bulletphysics/bullet3/issues/2470
        #gc.collect()

        if buffer is not None:
            if collect_trajectories:
                return total_reward, total_steps, buffer,average_stats(self.reward_stats_samples), average_stats(self.info_stats_samples),average_stats(self.reward_IG_stats_samples),trajectories
            else:
                return total_reward, total_steps, buffer,average_stats(self.reward_stats_samples), average_stats(self.info_stats_samples),average_stats(self.reward_IG_stats_samples)
        else:
            if collect_trajectories:
                return total_reward, total_stepsc,average_stats(self.reward_stats_samples), average_stats(self.info_stats_samples),average_stats(self.reward_IG_stats_samples), trajectories
            else:
                return total_reward, total_stepsc,average_stats(self.reward_stats_samples), average_stats(self.info_stats_samples),average_stats(self.reward_IG_stats_samples)
