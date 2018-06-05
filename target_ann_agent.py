#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : target_ann_agent.py
#       @date         : 2018/05/12 22:22
import tensorflow as tf
import numpy as np
import ann_agent

class TargetANNAgent(ann_agent.ANNAgent):
    def __init__(self, state_size, action_size, sess, learning_rate):
        super(TargetANNAgent, self).__init__(state_size, action_size, sess, learning_rate)
        [self.t_state_input, _], [self.t_logits, _], self.t_train_var = self._build_model("target_ann_agent_net")
        ops = []
        self.r = tf.placeholder(tf.float32, (1))
        for idx in xrange(len(self.train_var)):
            ops.append(self.t_train_var[idx].assign(self.r * self.train_var[idx] + (1-self.r) * self.t_train_var[idx]))
        self.sync_ops = tf.group(*ops)

    def all_action_q_value(self, state):
        state = np.array(state)
        if len(state.shape) < 2:
            state = [state]
        out = self.sess.run(self.t_logits, feed_dict={self.t_state_input:state})[0]
        return out

    def sync_weight(self, rate = 1.0):
        self.sess.run(self.sync_ops, {self.r: [rate]})


if __name__ == '__main__':
    pass
