"""
This part of code is the Deep Q Network (DQN) brain.
view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: r1.2
"""

import numpy as np
import tensorflow as tf
import h5py

from tensorflow.python import debug as tf_debug

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features_x,
            n_features_y,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=50,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features_x = n_features_x
        self.n_features_y = n_features_y
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features_y * n_features_x * 2 + 2))
        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.sess, "./network/net_work.ckpt")
        except:
            print "Failed to RESTORE network"

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("./logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features_y, self.n_features_x], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features_y, self.n_features_x], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 1.), tf.random_normal_initializer(.0, 1.)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, self.n_features_x * self.n_features_y, tf.nn.relu,
                                 kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, self.n_features_x, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            e3 = tf.layers.dense(e2, self.n_features_y, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e3')
            e4 = tf.layers.dense(e3, self.n_actions * self.n_features_x * self.n_features_y, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e4')
            e5 = tf.layers.dense(e4, self.n_actions, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e5')
            self.q_eval = tf.layers.dense(e5, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, self.n_features_x * self.n_features_y, tf.nn.relu,
                                 kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, self.n_features_x, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            t3 = tf.layers.dense(t2, self.n_features_y, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t3')
            t4 = tf.layers.dense(t3, self.n_actions * self.n_features_x * self.n_features_y, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t4')
            t5 = tf.layers.dense(t4, self.n_actions, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t5')
            self.q_next = tf.layers.dense(t5, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t')
        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(?, 2)
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        stack = np.column_stack((s, s_))
        stack = np.insert(stack.reshape(1, -1), 0, (a, r))
        transition = np.hstack(stack)
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def save_memory(self):
        with h5py.File('data.h5', 'w') as f:
            f.create_dataset('memory', data=self.memory)

    def load_memory(self):
        with h5py.File('data.h5', 'r') as f:
            memory = f['memory'][:]
            empty_memory = np.zeros((self.memory_size - memory.shape[1], self.n_features_y * self.n_features_x * 2 + 2))
            self.memory = np.concatenate((memory, empty_memory))

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.choice([0,2,2,1,3,3,4], 1)[0]
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')
        if self.learn_step_counter % self.replace_target_iter * 100 == 0:
            self.saver.save(self.sess, './network/net_work.ckpt')
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        fa = batch_memory[:, 0]
        fr = batch_memory[:, 1]
        ss_list = batch_memory[:, 2:]
        ss_reshape = [x.reshape((self.n_features_y, self.n_features_x * 2)) for x in ss_list]
        fs, fs_ = [], []
        for x in ss_reshape:
            t, t_ = np.split(x, 2, 1)
            fs.append(t)
            fs_.append(t_)

        # debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=self.sess)
        # debug_sess.run([self._train_op, self.loss],
        #     feed_dict={
        #         self.s: fs,
        #         self.a: fa,
        #         self.r: fr,
        #         self.s_: fs_,
        #     })
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: fs,
                self.a: fa,
                self.r: fr,
                self.s_: fs_,
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


if __name__ == '__main__':
    DQN = DeepQNetwork(4, 1, 2, output_graph=True)
    s = [3,4]
    a = 2
    r = 0
    s_= [5,6]
    DQN.store_transition(s, a, r, s_)
    DQN.store_transition(s, a, r, s_)
    DQN.learn()
