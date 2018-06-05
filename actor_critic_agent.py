#!/usr/bin/env python
# encoding: utf8
#
#       @author       : tcund@126.com
#       @file         : actor_critic_agent.py
#       @date         : 2018/06/02 14:51
import agent_interface
import numpy as np
import tensorflow as tf
import random
import os

class ActorCriticAgent(agent_interface.AgentInterface):
    def __init__(self, state_size, action_size, sess, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess
        self.learning_rate = learning_rate
        self.hidden_units_count = 24
        [self.state_input, self.td_error], [self.logits, self.actor_train_op], self.train_var = self._build_actor_model("actor")
        _, [self.v_out, self.critic_train_op], self.critic_train_var = self._build_critic_model("critic")

    def _build_actor_model(self, scope):
        with tf.variable_scope(scope):
            state_input = tf.placeholder(tf.float32, (None, self.state_size))
            td_error = tf.placeholder(tf.float32, (None, self.action_size))  # 只有在action上存在td error，其他维度上为0
            dense1 = tf.layers.dense(inputs = state_input, units = self.hidden_units_count, activation = tf.nn.relu)
            dense2 = tf.layers.dense(inputs = dense1, units = self.hidden_units_count, activation = tf.nn.relu)
            logits = tf.layers.dense(inputs = dense2, units = self.action_size)
            ps = tf.nn.softmax(logits)
            grads = tf.gradients(tf.nn.log_softmax(logits), tf.trainable_variables(scope), grad_ys = -td_error)
            reg_grads = tf.gradients(tf.reduce_sum(ps * tf.log(ps) * 0.01), tf.trainable_variables(scope))
            grads_and_vars = list((grads[idx] + reg_grads[idx], v) for idx, v in enumerate(tf.trainable_variables(scope)) )
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            train_op = optimizer.apply_gradients(grads_and_vars)
            return [state_input, td_error], [logits, train_op], tf.trainable_variables(scope)

    def _build_critic_model(self, scope):
        with tf.variable_scope(scope):
            state_input = self.state_input
            td_error = tf.reshape(tf.reduce_sum(self.td_error, 1), [-1, 1])
            dense1 = tf.layers.dense(inputs = state_input, units = self.hidden_units_count, activation = tf.nn.relu)
            #dense2 = dense1
            dense2 = tf.layers.dense(inputs = dense1, units = self.hidden_units_count, activation = tf.nn.relu)
            logits = tf.layers.dense(inputs = dense2, units = 1)
            grads = tf.gradients(logits, tf.trainable_variables(scope), grad_ys = -td_error)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = list(zip(grads, tf.trainable_variables(scope)))
            train_op = optimizer.apply_gradients(grads_and_vars)
            return [state_input, td_error], [logits, train_op], tf.trainable_variables(scope)

    def choose_action(self, state, rand=None):
        state = np.array(state)
        if len(state.shape) == 1:
            state = state[np.newaxis, :]
        return np.argmax(self.sess.run(self.logits, feed_dict={self.state_input:state})[0])

    def q_value(self, state, action):
        raise

    def all_action_q_value(self, state):
        raise

    def value(self, state):
        state = np.array(state)
        if len(state.shape) == 1:
            state = state[np.newaxis, :]
        return self.sess.run(self.v_out, feed_dict={self.state_input:state})[0]

    def update_value(self, state, action, error):
        state = np.array(state)
        if len(state.shape) == 1:
            state = state[np.newaxis, :]
        #print np.average(np.square(error))
        error_input = np.zeros((len(error), self.action_size))
        for idx, i in enumerate(error):
            error_input[idx][action[idx]] = i

        self.sess.run([self.actor_train_op, self.critic_train_op], feed_dict={self.state_input:state, self.td_error:error_input})


if __name__ == '__main__':
    pass
