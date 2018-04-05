# %%
import tensorflow as tf
import mnn
import matplotlib.pyplot as plt
import numpy as np

from importlib import reload
reload(mnn)

# %%
def load_image(path, dsize):
    data = tf.read_file(path)
    img = tf.image.decode_png(data, channels=1) / 255

    return tf.reshape(img, [1, dsize])

# %%
dsize = 32*32
letters = ['A', 'X']
images = [load_image('data/{0}.png'.format(l), dsize) for l in letters]

X = tf.reshape(tf.stack(images), [len(letters), dsize])
M = mnn.morph_matmul(tf.transpose(X), X, mnn.bmax)

X_p = images[1]
Y_p = mnn.morph_matmul(X_p, M, mnn.bmax)

# %%
with tf.Session() as sess:
    recall = sess.run(Y_p)
    plt.imshow(recall.reshape([32, 32]))
