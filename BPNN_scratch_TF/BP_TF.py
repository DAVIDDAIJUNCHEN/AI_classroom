#! /usr/bin/env python3
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# build input placeholder
x = tf.placeholder(tf.float32, [None, 3])
y = tf.placeholder(tf.float32, [None, 1])

# build model
W1 = tf.Variable(tf.random_normal([3, 2]))
b1 = tf.Variable(tf.zeros([2]))
W2 = tf.Variable(tf.random_normal([2, 1]))
b2 = tf.Variable(tf.zeros([1]))

f1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
f2 = tf.nn.sigmoid(tf.matmul(f1, W2) + b2)

# cost function
cost = tf.reduce_mean(tf.reduce_sum(tf.square(f2 - y), axis=1))

# optimizer
learning_rate = 1e-1
optimizer = tf.train.GradientDecentOptimizer(learning_rate).minimize(cost)

# train model
training_epochs = 10
batch_size = 100
display_step = 1

train_X = np.random.normal(0, 1, [100, 3])
train_Y = 2 * train_X[:, 0] + 5 * train_X[:, 1] + 10 * train_X[:, 2]
train_Y = np.array([[y] for y in train_Y])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={x:train_X, y:train_Y})
        avg_cost = c
    if (epoch + 1) % display_step == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

    print('Finished')
