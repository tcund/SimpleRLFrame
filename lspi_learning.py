#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : q_learning_agent.py
#       @date         : 2018/04/20 17:54
import numpy as np
import learning_interface
import learning_interface_multi_env
import sys
from collections import deque

class LSPILearning(learning_interface_multi_env.LearningInterface):
    def __init__(self, gamma, env):
        super(LSPILearning, self).__init__(gamma, env)

    def learn(self, episode, max_step = None, display = False):
        l_episode = deque()
        for e in xrange(episode):
            state = self.env_init()
            step = 0
            while True:
                if max_step is not None and step >= max_step:
                    break
                if display:
                    self.env_render()
                action = self.choose_action(state, 1.0)
                next_state, reward, done, _ = self.env_step(action)
                reward = self.calc_reward(reward, done)
                l_episode.append([state, action, reward, next_state, done])

                state = next_state
                if done:
                    break

                step += 1
            print >> sys.stderr, "processing... %i, score: %d" % (e, reward)
        count = 0
        while True:
            se = self.lspi_update(l_episode)
            if se < 0.003:
                break
        loss = 0.0
        for  state, action, reward, next_state, done in l_episode:
            next_action = self.choose_action(next_state, False)
            loss += (reward + self.gamma * self.q_value(next_state, next_action) - self.q_value(state, action)) ** 2
        #print loss / len(l_episode)

    def lspi_update(self, episodes):
        A = np.zeros((self.weight_size, self.weight_size))
        b = np.zeros((self.weight_size))
        for state, action, reward, next_state, done in episodes:
            qv = self.feature_ext(state, action)
            next_action = self.choose_action(next_state, False)
            nqv = self.feature_ext(next_state, next_action)
            A += np.outer(qv, qv - self.gamma * nqv) if not done else np.outer(qv, qv)
            b += reward * qv
        w = np.dot(np.linalg.pinv(A), b)    
        se = self.update_weight(w)
        return se


if __name__ == '__main__':
    pass
