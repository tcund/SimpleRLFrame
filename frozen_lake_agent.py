#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : frozon_lake_agent.py
#       @date         : 2018/05/07 15:53
import sys
import random
import ann_agent
import ann_agent_ET
import ls_agent
import q_learning_replay
import q_learning
import sarsa
import mc_learning
import lspi_learning
import tensorflow as tf
import numpy as np
import gym
from gym.envs import register

class FrozenLakeAgentLSPI(ls_agent.LSAgent, lspi_learning.LSPILearning):
    def __init__(self, state_size, action_size, gamma, env):
        ls_agent.LSAgent.__init__(self, state_size, action_size)
        lspi_learning.LSPILearning.__init__(self, gamma, env)

    def calc_reward(self, reward, done):
        return reward
        if not done:
            pass
            return -0.01
        elif reward != 1:
            return -1
        return reward

    def feature_ext(self, state, action=None):
        if action is None:
            out = np.zeros((self.state_size))
            out[state] = 1
        else:
            out = np.zeros((self.state_size * self.action_size))
            out[action * self.state_size + state] = 1
        return out

register(
    id='FrozenLake-simple-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', "is_slippery": False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

if __name__ == '__main__':

    envs = [gym.make("FrozenLake-v0") for _ in range(1)]
    env = envs[0]
    state_size = env.observation_space.n
    action_size = env.action_space.n
    sess = tf.Session()
    agent = FrozenLakeAgentLSPI(state_size, action_size, 0.9, envs)
    init = tf.global_variables_initializer()
    sess.run(init)
    agent.learn(10000, 490, rand=1.0)
    #agent.eval(3000, True)
    for i in xrange(4):
        for j in xrange(4):
            print agent.choose_action(i*4 + j, False),
        print 
