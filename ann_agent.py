#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : ann_agent.py
#       @date         : 2018/05/02 16:15

import agent_interface
import numpy as np
import tensorflow as tf
import random
import os

class ANNAgent(agent_interface.AgentInterface):
    def __init__(self, state_size, action_size, sess, learning_rate):
        super(ANNAgent, self).__init__(state_size, action_size)
        self.sess = sess
        self.learning_rate = learning_rate
        self.hidden_units_count = 24
        [self.state_input, self.td_error], [self.logits, self.train_op], self.train_var = self._build_model("ann_agent_net")

    def _build_model(self, scope):
        with tf.variable_scope(scope):
            state_input = tf.placeholder(tf.float32, (None, self.state_size))
            td_error = tf.placeholder(tf.float32, (None, self.action_size))
            dense1 = tf.layers.dense(inputs = state_input, units = self.hidden_units_count, activation = tf.nn.relu)
            dense2 = tf.layers.dense(inputs = dense1, units = self.hidden_units_count, activation = tf.nn.relu)
            logits = tf.layers.dense(inputs = dense2, units = self.action_size)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads = tf.gradients(logits, tf.trainable_variables(scope), grad_ys = -td_error * 2)
            #grads, _ = tf.clip_by_global_norm(grads, 1)
            grads_and_vars = list(zip(grads, tf.trainable_variables(scope)))
            train_op = optimizer.apply_gradients(grads_and_vars)
            return [state_input, td_error], [logits, train_op], tf.trainable_variables(scope)

        #self.writer = tf.summary.FileWriter("/tmp/tbFile/%d" % os.getpid())
        #tf.summary.scalar('Total_Loss', tf.reduce_mean(tf.square(self.td_error)))
        #for i in xrange(len(self.grads)):
        #    tf.summary.histogram('var_grad_%d' % i, self.grads[i])
        #self.write_op = tf.summary.merge_all()
        #self.write_step = 0

    def choose_action(self, state, rand=1.0):
        if np.random.rand() <= rand:
            return random.randrange(self.action_size)
        return np.argmax(self.all_action_q_value(state))

    def q_value(self, state, action):
        return self.all_action_q_value(state)[action]

    def all_action_q_value(self, state):
        state = np.array(state)
        if len(state.shape) < 2:
            state = [state]
        out = self.sess.run(self.logits, feed_dict={self.state_input:state})[0]
        return out

    def update_value(self, state, action, error):
        # 只在action方向有误差，其他action误差为0
        error_input = np.zeros((len(error), self.action_size))
        for idx, i in enumerate(error):
            error_input[idx][action[idx]] = i
        self.sess.run([self.train_op], feed_dict={self.state_input:state, self.td_error:error_input})
        #self.write_step += 1 
        #if self.write_step % 1 == 0:
        #    self.writer.add_summary(summary, self.write_step)
        #    self.writer.flush()

if __name__ == '__main__':
    pass
