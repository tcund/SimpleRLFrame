#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : td_learing_replay.py
#       @date         : 2018/06/02 16:09
import numpy as np
import q_learning_replay
import random

class ValueLearningReplay(q_learning_replay.QLearningReplay):
    def __init__(self, gamma, env, replay_size, batch_size):
        super(ValueLearningReplay, self).__init__(gamma, env, replay_size, batch_size)

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        s_i = [None] * self.batch_size
        s_a = [None] * self.batch_size
        s_e = [None] * self.batch_size
        for idx,  (state, action, reward, next_state, done) in enumerate(minibatch):
            s_i[idx] = state
            s_a[idx] = action
            target = reward + (
                    self.gamma * self.value(next_state)[0] if not done else 0)
            error = target - self.value(state)[0]
            s_e[idx] = error
        self.update_value(s_i, s_a, s_e)

if __name__ == '__main__':
    pass
