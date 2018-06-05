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

class QLearning(learning_interface_multi_env.LearningInterface):
    def __init__(self, gamma, env):
        super(QLearning, self).__init__(gamma, env)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_delay = 0.995

    def learn(self, episode, max_step = None, display = False):
        for e in xrange(episode):
            state = self.env_init()
            step = 0
            while True:
                if max_step is not None and step >= max_step:
                    break
                if display:
                    self.env_render()
                action = self.choose_action(state, rand=self.epsilon)
                next_state, reward, done, _ = self.env_step(action)
                reward = self.calc_reward(reward, done)
                target = reward + (
                        self.gamma * np.amax(self.all_action_q_value(next_state)) if not done else 0
                        )
                error = target - self.q_value(state, action)

                self.update_value([state], [action], [error])
                state = next_state
                if done:
                    break

                step += 1
            print >> sys.stderr, "processing... %i, score: %d" % (e, step)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_delay

if __name__ == '__main__':
    pass
