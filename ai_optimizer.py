import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

# Set random seed for reproducibility
tf.set_random_seed(1)
np.random.seed(1)  # Set the seed for NumPy random number generation

# Hyper Parameters
BATCH_SIZE = 64
LR = 0.002  # initial learning rate
N_TEST_IMG = 5
EARLY_STOPPING_THRESHOLD = 0.0001  # threshold for early stopping

# Mnist digits
mnist = input_data.read_data_sets('./mnist', one_hot=False)
test_x = mnist.test.images[:200]
test_y = mnist.test.labels[:200]

# Plot one example
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0]))
plt.show()

# TensorFlow placeholders
tf_x = tf.placeholder(tf.float32, [None, 28 * 28])

# Encoder
en0 = tf.layers.dense(tf_x, 128, tf.nn.tanh)
en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
en2 = tf.layers.dense(en1, 12, tf.nn.tanh)
encoded = tf.layers.dense(en2, 3)

# Decoder
de0 = tf.layers.dense(encoded, 12, tf.nn.tanh)
de1 = tf.layers.dense(de0, 64, tf.nn.tanh)
de2 = tf.layers.dense(de1, 128, tf.nn.tanh)
decoded = tf.layers.dense(de2, 28 * 28, tf.nn.sigmoid)

# Loss and optimizer
loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LR, global_step, decay_steps=100, decay_rate=0.96, staircase=True)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# AI-driven early stopping mechanism
previous_loss = None
early_stopping_count = 0

# Initialize NumPy random number generator
rng = np.random.default_rng(seed=1)

# AI-driven data augmentation function
def augment_data(batch_x):
    augmented_batch = []
    for img in batch_x:
        if rng.random() > 0.5:
            img = np.flipud(img.reshape(28, 28)).flatten()  # flip image upside down
        if rng.random() > 0.5:
            img = np.fliplr(img.reshape(28, 28)).flatten()  # flip image left to right
        augmented_batch.append(img)
    return np.array(augmented_batch)

# Session and initialization
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()

# Original data (first row) for viewing
view_data = mnist.test.images[:N_TEST_IMG]
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

# Training loop with AI-driven enhancements
for step in range(8000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    b_x = augment_data(b_x)  # Apply data augmentation
    _, encoded_, decoded_, loss_ = sess.run([train, encoded, decoded, loss], {tf_x: b_x})

    # Early stopping check
    if previous_loss is not None and abs(previous_loss - loss_) < EARLY_STOPPING_THRESHOLD:
        early_stopping_count += 1
        if early_stopping_count >= 10:
            print("Early stopping triggered at step:", step)
            break
    else:
        early_stopping_count = 0
    previous_loss = loss_

    # Dynamic learning rate adjustment
    if step % 100 == 0:
        print('Train loss: %.4f, Learning rate: %.6f' % (loss_, sess.run(learning_rate)))
        decoded_data = sess.run(decoded, {tf_x: view_data})
        for i in range(N_TEST_IMG):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        plt.draw()
        plt.pause(0.01)

plt.ioff()

# Visualize in 3D plot
view_data = test_x[:200]
encoded_data = sess.run(encoded, {tf_x: view_data})
fig = plt.figure(2)
ax = Axes3D(fig)
X, Y, Z = encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2]
for x, y, z, s in zip(X, Y, Z, test_y):
    c = cm.rainbow(int(255 * s / 9))
    ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
