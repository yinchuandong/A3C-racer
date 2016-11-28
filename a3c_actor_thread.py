import tensorflow as tf
import numpy as np
import random
import time

from accum_trainer import AccumTrainer
from a3c_network import A3CFFNetwork, A3CLSTMNetwork
from config import *


def timestamp():
    return time.time()


class A3CActorThread(object):

    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 optimizer,
                 max_global_time_step,
                 device
                 ):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        if USE_LSTM:
            self.local_network = A3CLSTMNetwork(STATE_DIM, STATE_CHN, ACTION_DIM, device, thread_index)
        else:
            self.local_network = A3CFFNetwork(STATE_DIM, STATE_CHN, ACTION_DIM, device)
        self.local_network.create_loss(ENTROPY_BETA)
        self.trainer = AccumTrainer(device)
        self.trainer.create_minimize(self.local_network.total_loss, self.local_network.get_vars())
        self.accum_gradients = self.trainer.accumulate_gradients()
        self.reset_gradients = self.trainer.reset_gradients()

        clip_accum_grads = [tf.clip_by_norm(accum_grad, 40.0) for accum_grad in self.trainer.get_accum_grad_list()]
        self.apply_gradients = optimizer.apply_gradients(zip(clip_accum_grads, global_network.get_vars()))

        self.sync = self.local_network.sync_from(global_network)

        self.local_t = 0
        self.initial_learning_rate = initial_learning_rate

        # for log
        self.episode_reward = 0.0
        self.episode_start_time = 0.0
        self.prev_local_t = 0

        # for pull mode, like brower based game
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.start_lstm_state = None
        return

    def set_log_parmas(self, summary_writer, summary_op, reward_input, time_input):
        '''
        notes: need to be called after initializing the class
        '''
        self.summary_writer = summary_writer
        self.summary_op = summary_op
        self.reward_input = reward_input
        self.time_input = time_input
        return

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * \
            (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, policy_output):
        sum_pi = []
        sum = 0.0
        for rate in policy_output:
            sum += rate
            sum_pi.append(sum)

        r = random.random() * sum
        for i in range(len(sum_pi)):
            if sum_pi[i] >= r:
                return i
        return len(sum_pi) - 1

    def _record_log(self, sess, global_t, reward, living_time):
        summary_str = sess.run(self.summary_op, feed_dict={
            self.reward_input: reward,
            self.time_input: living_time
        })
        self.summary_writer.add_summary(summary_str, global_t)
        return

    def process(self, sess, global_t, state, reward, terminal):
        # reduce the influence of socket connecting time
        if self.episode_start_time == 0.0:
            self.episode_start_time = timestamp()
            # copy weight from global network
            sess.run(self.reset_gradients)
            sess.run(self.sync)
            if USE_LSTM:
                self.start_lstm_state = self.local_network.lstm_state_out

        policy_, value_ = self.local_network.run_policy_and_value(sess, state)
        if self.thread_index == 0 and self.local_t % 1000 == 0:
            print 'policy=', policy_
            print 'value=', value_

        action_id = self.choose_action(policy_)

        self.states.append(state)
        self.actions.append(action_id)
        self.values.append(value_)

        self.episode_reward += reward
        self.rewards.append(np.clip(reward, -1.0, 1.0))

        self.local_t += 1

        if terminal:
            episode_end_time = timestamp()
            living_time = episode_end_time - self.episode_start_time

            self._record_log(sess, global_t, self.episode_reward, living_time)

            print ("global_t=%d / reward=%.2f / living_time=%.4f") % (global_t, self.episode_reward, living_time)

            # reset variables
            self.episode_reward = 0.0
            self.episode_start_time = episode_end_time
            if USE_LSTM:
                self.local_network.reset_lstm_state()
        elif self.local_t % 2000 == 0:
            # save log per 2000 episodes
            living_time = timestamp() - self.episode_start_time
            self._record_log(sess, global_t, self.episode_reward, living_time)
        # -----------end of batch (LOCAL_T_MAX)--------------------

        # do training
        if self.local_t % LOCAL_T_MAX == 0 or terminal:
            R = 0.0
            if not terminal:
                R = self.local_network.run_value(sess, state)

            self.states.reverse()
            self.actions.reverse()
            self.rewards.reverse()
            self.values.reverse()

            batch_state = []
            batch_action = []
            batch_td = []
            batch_R = []

            for (ai, ri, si, Vi) in zip(self.actions, self.rewards, self.states, self.values):
                R = ri + GAMMA * R
                td = R - Vi
                action = np.zeros([ACTION_DIM])
                action[ai] = 1

                batch_state.append(si)
                batch_action.append(action)
                batch_td.append(td)
                batch_R.append(R)

            if USE_LSTM:
                batch_state.reverse()
                batch_action.reverse()
                batch_td.reverse()
                batch_R.reverse()
                sess.run(self.accum_gradients, feed_dict={
                    self.local_network.state_input: batch_state,
                    self.local_network.action_input: batch_action,
                    self.local_network.td: batch_td,
                    self.local_network.R: batch_R,
                    self.local_network.step_size: [len(batch_state)],
                    self.local_network.initial_lstm_state: self.start_lstm_state
                })
                self.start_lstm_state = self.local_network.lstm_state_out
            else:
                sess.run(self.accum_gradients, feed_dict={
                    self.local_network.state_input: batch_state,
                    self.local_network.action_input: batch_action,
                    self.local_network.td: batch_td,
                    self.local_network.R: batch_R
                })

            cur_learning_rate = self._anneal_learning_rate(global_t)
            sess.run(self.apply_gradients, feed_dict={
                self.learning_rate_input: cur_learning_rate
            })

            print len(self.states), len(self.actions), len(self.values)
            # reste temporal buffer
            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []

            sess.run(self.reset_gradients)
            sess.run(self.sync)

        return action_id


if __name__ == '__main__':
    # game_state = GameState()
    # game_state.process(1)
    # print np.shape(game_state.s_t)
    print timestamp()
    print time.time()
