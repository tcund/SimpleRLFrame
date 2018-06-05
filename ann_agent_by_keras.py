#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : ann_agent.py
#       @date         : 2018/05/02 16:15

import agent_interface
import numpy as np
import random
import os
import tensorflow
Sequential = tensorflow.keras.models.Sequential
Dense = tensorflow.keras.layers.Dense
Adam = tensorflow.keras.optimizers.Adam

class ANNAgent(agent_interface.AgentInterface):
    def __init__(self, state_size, action_size, sess, learning_rate):
        super(ANNAgent, self).__init__(state_size, action_size)
        self.sess = sess
        self.learning_rate = learning_rate
        self.hidden_units_count = 24
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def choose_action(self, state, rand=1.0):
        if np.random.rand() <= rand:
            return random.randrange(self.action_size)
        return np.argmax(self.all_action_q_value(state))

    def q_value(self, state, action):
        return self.all_action_q_value(state)[action]

    def all_action_q_value(self, state):
        state = np.array([state])
        return self.model.predict(state)[0]

    def update_value_target(self, state, action, target):
        self.model.fit(np.array(state), np.array(target), epochs=64, verbose=0)

if __name__ == '__main__':
    pass
