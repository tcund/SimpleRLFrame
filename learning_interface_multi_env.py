#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : learning_interface.py
#       @date         : 2018/04/20 18:19
import random

class LearningInterface(object):
    def __init__(self, gamma, env):
        self.gamma = gamma
        self.envs = env
        self.cur_env = random.sample(env, 1)[0]

    def env_init(self):
        self.cur_env = random.sample(self.envs, 1)[0]
        return self.cur_env.reset()

    def env_render(self):
        self.cur_env.render()

    def env_step(self, action):
        return self.cur_env.step(action)

    def calc_reward(self, reward, done):
        return reward

    def learn(self, episode, max_step = None, display = False):
        pass

    def eval(self, max_step = None, display = False):
        state = self.env_init()
        self.cur_env.seed()
        step = 0
        while True:
            if max_step is not None and step >= max_step:
                break
            if display:
                self.env_render()
            action = self.choose_action(state, 0.0)
            next_state, reward, done, _ = self.env_step(action)
            state = next_state
            if done:
                break
            step += 1
        return step

if __name__ == '__main__':
    pass
