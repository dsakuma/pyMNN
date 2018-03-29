# %%
import tensorflow as tf
import numpy as np

# %%
def _mdot(a, b, fn):
    assert a.shape == b.T.shape
    assert len(a.shape) == 1

    return fn(a + b.T)


class bmax:
    def mdot(a, b): return _mdot(a, b, np.max)


class bmin:
    def mdot(a, b): return _mdot(a, b, np.min)


def _mmatmul_g(a, b, body):
    dims = a.shape
    assert dims == b.T.shape

    for k in range(dims[0]):
        row_a = a[k]

        for l in range(dims[0]):
            col_b = b.T[l]
            yield body.mdot(row_a.A1, col_b.T.A1)


def mmatmul(a, b, body):
    gen = _mmatmul_g(a, b, body)
    return np.matrix(list(gen)).reshape((a.shape[0], b.shape[1]))

# %%
a = np.matrix([[1, 2],
               [4, 5],
               [7, 8]])

b = np.matrix([[10, 11, 12],
               [13, 14, 15]])

# %%
def _morph_matmul_elem(a, b, i, j):
    c = tf.gather(a, i)
    d = tf.gather(b, j)
    return tf.add(c, d)

def morph_matmul(a, b):
    i, j = a.shape
    outs = [a.shape[0], b.shape[1]]

    indices = tf.range(i * j)

    mat_i = tf.tile(indices, [i])
    mat_j = tf.reshape(tf.transpose(tf.reshape(tf.tile(jdx, [i]), [-1, j, i]), perm=[2, 0, 1]), [-1])

    sums = _morph_matmul_elem(tf.reshape(a, [-1]), tf.reshape(b, [-1]), mat_i, mat_j)
    to_reduce = tf.reshape(sums, [-1, 2])
    data = tf.reduce_max(to_reduce, axis=1)
    return tf.transpose(tf.reshape(data, outs))

a = tf.convert_to_tensor(a)
b = tf.convert_to_tensor(b)
c = morph_matmul(a, b)

with tf.Session() as sess:
    print(sess.run(c))
