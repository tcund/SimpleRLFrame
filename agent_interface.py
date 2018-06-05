#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : agent_interface.py
#       @date         : 2018/04/20 16:32

class AgentInterface(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def choose_action(self, state):
        pass

    def q_value(self, state, action):
        pass

    def all_action_q_value(self, state):
        pass

    def value(self, state):
        pass

    def update_value(self, state, action, error):
        pass

    def update_value_target(self, state, action, target):
        pass

if __name__ == '__main__':
    pass
