# %%
import tensorflow as tf

# %%
x = tf.Variable(tf.random_uniform([]))
y = tf.Variable(tf.random_uniform([]))

res = tf.cond(x < y, lambda: x + y, lambda: x - y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(x.eval(), y.eval(), res.eval())

# %%
def random_int():
    return tf.random_uniform(
        [],
        minval=1,
        maxval=6,
        dtype=tf.int32)

x = tf.Variable(random_int())
y = tf.Variable(random_int())

res = tf.case(
    [(x < y, lambda: x + y ** 2),
     (x > y, lambda: x - y)],
    default=lambda: tf.constant(0))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(x.eval(), y.eval(), res.eval())

# %%
x = tf.constant([-1, -2, -3, 0, 1, 2], shape=[2, 3], dtype=tf.int32)
y = tf.zeros_like(x, dtype=tf.int32)
z = tf.equal(x, y)

with tf.Session() as sess:
    out = sess.run(z)
    print(out)
