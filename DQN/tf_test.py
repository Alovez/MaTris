import numpy as np
import tensorflow as tf


a = tf.Variable([[1,2,3,4],[5,6,7,8]])
b = tf.Variable([[1],[3]])
c = tf.one_hot(tf.transpose(b), depth=a.shape[1], on_value=1, off_value=0)
d = a * c
e = np.max(a, axis=1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print sess.run(e)