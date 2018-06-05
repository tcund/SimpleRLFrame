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

class MCLearning(learning_interface_multi_env.LearningInterface):
    def __init__(self, gamma, env):
        super(MCLearning, self).__init__(gamma, env)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_delay = 0.995

    def learn(self, episode, max_step = None, display = False):
        for e in xrange(episode):
            state = self.env_init()
            step = 0
            l_episode = deque()
            while True:
                if max_step is not None and step >= max_step:
                    break
                if display:
                    self.env_render()
                action = self.choose_action(state, rand=self.epsilon)
                next_state, reward, done, _ = self.env_step(action)
                reward = self.calc_reward(reward, done)
                l_episode.append([state, action, reward, done])

                state = next_state
                if done:
                    break

                step += 1
            self.mc_update(l_episode)
            print >> sys.stderr, "processing... %i, score: %d" % (e, step)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_delay

    def select_episode(self, episode):
        return episode

    def mc_update(self, episode):
        for i in range(len(episode)-2, -1, -1):
            episode[i][2] += episode[i+1][2] * self.gamma
        episode = self.select_episode(episode)
        s_i = [None] * len(episode)
        s_a = [None] * len(episode)
        s_e = [None] * len(episode)

        for idx, (state, action, mc_reward, done) in enumerate(episode):
            error = mc_reward - self.q_value(state, action)
            s_i[idx] = state
            s_a[idx] = action
            s_e[idx] = error
        self.update_value(s_i, s_a, s_e)

if __name__ == '__main__':
    pass
