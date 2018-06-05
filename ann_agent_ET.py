#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : ann_agent_ET.py
#       @date         : 2018/05/04 17:47

import ann_agent
import tensorflow as tf
import numpy as np
import os

class ANNAgentET(ann_agent.ANNAgent):
    def __init__(self, state_size, action_size, sess, learning_rate, lamb):
        self.gamma = 0
        self.lamb = lamb
        super(ANNAgentET, self).__init__(state_size, action_size, sess, learning_rate)

    def _build_model(self):
        self.state_input = tf.placeholder(tf.float32, (1, self.state_size))
        self.td_error = tf.placeholder(tf.float32, (1))
        self.action = tf.placeholder(tf.int32, (1))
        dense1 = tf.layers.dense(inputs = self.state_input, units = self.hidden_units_count, activation = tf.nn.relu)
        dense2 = tf.layers.dense(inputs = dense1, units = self.hidden_units_count, activation = tf.nn.relu)
        self.logits = tf.layers.dense(inputs = dense2, units = self.action_size)
        self.logit = tf.gather(self.logits, self.action, axis=1)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads = tf.gradients(self.logit, tf.trainable_variables())

        self.et = [tf.placeholder(tf.float32, i.shape) for i in grads]
        self.et_value = [np.zeros(i.shape) for i in grads] # et 初始化
        self.out_et = [self.et[i] * self.lamb * self.gamma + grads[i] for i in xrange(len(grads))]
        #td_error_cut = tf.minimum(tf.maximum(self.td_error, -1.0), 1.0)
        #self.out_et[0] = tf.Print(self.out_et[0], [td_error_cut])
        self.grads = [-2 * self.out_et[i] * self.td_error for i in xrange(len(grads))]
        grads_and_vars = list(zip(self.grads, tf.trainable_variables()))
        self.train_op = optimizer.apply_gradients(grads_and_vars)

        self.writer = tf.summary.FileWriter("/tmp/tbFile/%d" % os.getpid())
        tf.summary.scalar('Total_Loss', tf.reduce_mean(tf.square(self.td_error)))
        self.write_op = tf.summary.merge_all()
        self.write_step = 0

    def clear_et(self):
        for i in self.et_value:
            i.fill(0)

    def update_value(self, state, action, error):
        assert len(action) == 1
        assert len(error) == 1
        fd = {self.state_input:state, self.td_error:error, self.action:action}
        fd.update(zip(self.et, self.et_value))
        _, summary, self.et_value = self.sess.run([self.train_op, self.write_op, self.out_et], feed_dict=fd)
        self.write_step += 1
        if self.write_step % 1 == 0:
            self.writer.add_summary(summary, self.write_step)
            self.writer.flush()

if __name__ == '__main__':
    pass
