import os
import sys

import tensorflow as tf
import numpy as np
import math
import threading
import signal

from a3c_network import A3CFFNetwork, A3CLSTMNetwork
from a3c_actor_thread import A3CActorThread

from config import *


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


class A3C(object):

    def __init__(self):
        self.device = '/gpu:0' if USE_GPU else '/cpu:0'
        self.stop_requested = False
        self.global_t = 0
        if USE_LSTM:
            self.global_network = A3CLSTMNetwork(STATE_DIM, STATE_CHN, ACTION_DIM, self.device, -1)
        else:
            self.global_network = A3CFFNetwork(STATE_DIM, STATE_CHN, ACTION_DIM, self.device)

        self.initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW, INITIAL_ALPHA_HIGH, INITIAL_ALPHA_LOG_RATE)
        self.learning_rate_input = tf.placeholder('float')
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_input,
                                                   decay=RMSP_ALPHA, momentum=0.0, epsilon=RMSP_EPSILON)

        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))

        self.reward_input = tf.placeholder(tf.float32)
        tf.scalar_summary('reward', self.reward_input)

        self.time_input = tf.placeholder(tf.float32)
        tf.scalar_summary('living_time', self.time_input)

        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(LOG_FILE, self.sess.graph)

        self.actor_threads = []
        for i in range(PARALLEL_SIZE):
            actor_thread = A3CActorThread(i, self.global_network, self.initial_learning_rate,
                                          self.learning_rate_input, self.optimizer, MAX_TIME_STEP, self.device)
            actor_thread.set_log_parmas(self.summary_writer, self.summary_op, self.reward_input, self.time_input)
            self.actor_threads.append(actor_thread)

        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        self.restore()

        self.action_value_list = [None] * PARALLEL_SIZE
        return

    def restore(self):
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("checkpoint loaded:", checkpoint.model_checkpoint_path)
            tokens = checkpoint.model_checkpoint_path.split("-")
            # set global step
            self.global_t = int(tokens[1])
            print(">>> global step set: ", self.global_t)
        else:
            print("Could not find old checkpoint")
        return

    def backup(self):
        if not os.path.exists(CHECKPOINT_DIR):
            os.mkdir(CHECKPOINT_DIR)

        self.saver.save(self.sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step=self.global_t)
        return

    # def train_function(self, thread_id, state, reward, next_state, terminal):
    #     actor_thread = self.actor_threads[thread_id]
    #     action_id = actor_thread.process(self.sess, self.global_t, state, reward, terminal)
    #     self.global_t += 1
    #     if self.global_t % 1000000 < LOCAL_T_MAX:
    #         self.backup()
    #     return action_id
         
    def train_function(self, thread_id, state, reward, next_state, terminal, start_frame):
        actor_thread = self.actor_threads[thread_id]
        next_action_id, next_value = actor_thread.get_action(self.sess, next_state)
        if not start_frame:
            # for the start_frame, just use it to get value_
            action_id = self.action_value_list[thread_id][0]
            value = self.action_value_list[thread_id][1]
            actor_thread.process(self.sess, self.global_t, state, action_id, value, reward, terminal)

        # delay update of previous action and value
        self.action_value_list[thread_id] = (next_action_id, next_value)
        self.global_t += 1
        if self.global_t % 100000 == 0:
            self.backup()
        return next_action_id


if __name__ == '__main__':
    print 'a3c.py'
    # net = A3C()
    # net.train_function(0)

