# %%
from importlib import reload

import tensorflow as tf
import numpy as np
import mnn

reload(mnn)

# %%
a = np.matrix([[1, 2],
               [4, 5],
               [7, 8]])


b = np.matrix([[10, 11, 12],
               [13, 14, 15]])

a = tf.constant(a)
b = tf.constant(b)
v = tf.constant([[10, 20, 30]])

c = mnn.morph_matmul(a, b, mnn.bmax)
d = mnn.morph_matmul(a, tf.transpose(v), mnn.bmax)


with tf.Session() as sess:
    assert np.array_equal(
        np.array(
            [[15, 16, 17],
             [18, 19, 20],
             [21, 22, 23]]).T,
        sess.run(c))

    assert np.array_equal(
        np.array([[12, 25, 38]]).T,
        sess.run(d))
