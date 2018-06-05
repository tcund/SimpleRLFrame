#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : ls_agent.py
#       @date         : 2018/05/06 15:17

import agent_interface
import numpy as np
import random

class LSAgent(agent_interface.AgentInterface):
    def __init__(self,  state_size, action_size):
        super(LSAgent, self).__init__(state_size, action_size)
        self.state_weight = np.ones((state_size * action_size))
        self.weight_size = state_size * action_size
        self.epsilon = 1.0

    def choose_action(self, state, rand=1.0):
        if rand > 0 and np.random.rand() <= rand:
            return random.randrange(self.action_size)
        return np.argmax(self.all_action_q_value(state))

    def q_value(self, state, action):
        return self.all_action_q_value(state)[action]

    def all_action_q_value(self, state):
        raw_state = state
        state = self.feature_ext(state)
        return np.matmul(state, self.state_weight.reshape((self.action_size, self.state_size)).T)

    def update_weight(self, weights):
        se = np.mean(np.square(self.state_weight - weights))
        self.state_weight = weights
        return se


if __name__ == '__main__':
    pass
