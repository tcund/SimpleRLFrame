#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : cartpole_agent.py
#       @date         : 2018/05/03 11:11
import sys
import random
import ann_agent
import actor_critic_agent
import target_ann_agent
import ann_agent_ET
import ls_agent
import q_learning_replay
import value_learning_replay
import q_learning
import target_q_learning
import sarsa_learning
import mc_learning
import lspi_learning
import tensorflow as tf
import numpy as np
import gym
from gym.envs import register

class CartPoleAgentReplay(ann_agent.ANNAgent, q_learning_replay.QLearningReplay):
    def __init__(self, state_size, action_size, 
            sess, learning_rate,
            gamma, env, replay_size, batch_size):
        ann_agent.ANNAgent.__init__(self, state_size, action_size, sess, learning_rate)
        q_learning_replay.QLearningReplay.__init__(self, gamma, env, replay_size, batch_size)
        #sarsa.SARSA.__init__(self, gamma, env)
        #q_learning.QLearning.__init__(self, gamma, env)

    def calc_reward(self, reward, done):
        return -5 if done else reward

    def _update_value(self, state, action, error):
        if self.epsilon >= 0.6 or random.random() < 0.4 / (1 - self.epsilon):
            ann_agent.ANNAgent.update_value(self, state, action, error)

#class CartPoleAgent(ann_agent.ANNAgent, sarsa.SARSA):
class CartPoleAgent(ann_agent.ANNAgent, q_learning.QLearning):
    def __init__(self, state_size, action_size, 
            sess, learning_rate, 
            gamma, env):
        ann_agent.ANNAgent.__init__(self, state_size, action_size, sess, learning_rate)
        #sarsa.SARSA.__init__(self, gamma, env)
        q_learning.QLearning.__init__(self, gamma, env)

    def calc_reward(self, reward, done):
        return -3 if done else reward

class CartPoleAgentTarget(target_ann_agent.TargetANNAgent, target_q_learning.QLearning):
    def __init__(self, state_size, action_size, 
            sess, learning_rate, 
            gamma, env):
        target_ann_agent.TargetANNAgent.__init__(self, state_size, action_size, sess, learning_rate)
        target_q_learning.QLearning.__init__(self, gamma, env)

    def calc_reward(self, reward, done):
        return -3 if done else reward

class CartPoleAgentET(ann_agent_ET.ANNAgentET, sarsa_learning.SARSALearning):
    def __init__(self, state_size, action_size, 
            sess, learning_rate, lamb,
            gamma, env):
        ann_agent_ET.ANNAgentET.__init__(self, state_size, action_size, sess, learning_rate, lamb)
        sarsa_learning.SARSALearning.__init__(self, gamma, env)

    def calc_reward(self, reward, done):
        return -2 if done else reward

    def env_init(self):
        self.clear_et()
        return sarsa_learning.SARSALearning.env_init(self)

class CartPoleAgentMC(ann_agent.ANNAgent, mc_learning.MCLearning):
    def __init__(self, state_size, action_size, 
            sess, learning_rate,
            gamma, env):
        ann_agent.ANNAgent.__init__(self, state_size, action_size, sess, learning_rate)
        mc_learning.MCLearning.__init__(self, gamma, env)

    def calc_reward(self, reward, done):
        return -2 if done else reward

    def select_episode(self, episode):
        #return list(episode)[-10:]
        return random.sample(list(episode), min(10, len(episode)))

class CartPoleAgentLSPI(ls_agent.LSAgent, lspi_learning.LSPILearning):
    def __init__(self, state_size, action_size, gamma, env):
        ls_agent.LSAgent.__init__(self, state_size, action_size)
        lspi_learning.LSPILearning.__init__(self, gamma, env)

    def calc_reward(self, reward, done):
        return -2 if done else reward

    def feature_ext(self, state, action = None):
        if action is None:
            return state
        out = np.zeros((self.weight_size))
        out[action * self.state_size: (action + 1) * self.state_size] = state
        return out 

class CartPoleAgentAC(actor_critic_agent.ActorCriticAgent, value_learning_replay.ValueLearningReplay):
    def __init__(self, state_size, action_size,  sess, learning_rate,
            gamma, env, replay_size, batch_size):
        actor_critic_agent.ActorCriticAgent.__init__(self, state_size, action_size,  sess, learning_rate)
        value_learning_replay.ValueLearningReplay.__init__(self, gamma, env, replay_size, batch_size)

    def calc_reward(self, reward, done):
        return -10 if done else reward


if __name__ == '__main__':
    envs = [gym.make("CartPole-v1") for _ in range(10)]
    env = envs[0]
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    sess = tf.Session()
    #agent = CartPoleAgent(state_size, action_size, sess, 0.001, 
    #        0.95, envs)
    #agent = CartPoleAgentTarget(state_size, action_size, sess, 0.001, 
    #        0.95, envs)
    #agent = CartPoleAgentET(state_size, action_size, sess, 0.001, 0.9,
    #        0.95, envs)
    #agent = CartPoleAgentReplay(state_size, action_size, sess, 0.002, 
    #        0.95, envs, 2000, 64)
    agent = CartPoleAgentAC(state_size, action_size, sess, 0.001, 
            0.95, envs, 2000, 64)
    #agent = CartPoleAgentMC(state_size, action_size, sess, 0.002, 
    #        0.8, envs)
    #agent = CartPoleAgentLSPI(state_size, action_size, 0.95, envs)
    init = tf.global_variables_initializer()
    sess.run(init)
    eps = 1.0
    agent.learn(10000, 490)
    for i in range(10):
        step = agent.eval(None, True)
        print >> sys.stderr, step
