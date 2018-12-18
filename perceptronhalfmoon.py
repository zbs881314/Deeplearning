# Perceptron for Moon Classification Problem
# Hayin, Neural Network and Learning Machine, Chapter 1
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Step 1: generate halfmoon data
# parameters
rad = 10
width = 6
dist = 1  # distance between two half moon
num_train = 1000
num_test = 2000
num_data = num_train + num_test
epochs = 50


def generate_halfmoon_data(r, w, d, N):
    r = r + w / 2  # generate data for top moon
    R = (r - w) * np.ones((N, 1), dtype=np.float32) + np.random.rand(N, 1) * w  # randomly pick radial between r-w and r
    theta = np.random.rand(N, 1) * np.pi  # randomly pick the angular coordinate
    X = np.concatenate((R * np.cos(theta), R * np.sin(theta)), axis=1)
    Y = np.ones((N, 1), dtype=np.int)  # class 1

    R = (r - w) * np.ones((N, 1), dtype=np.float32) + np.random.rand(N, 1) * w  # generate data for bottom moon
    theta = np.pi + np.random.rand(N, 1) * np.pi
    del_x = r - (w / 2)  # move x coordinate by y coordinate down by d
    del_y = -d
    x = np.concatenate((R * np.cos(theta) + del_x, R * np.sin(theta) + del_y), axis=1)
    y = -np.ones((N, 1), dtype=np.int)  # class 2

    X = np.concatenate((X, x), axis=0)  # put two half-moon data set together
    Y = np.concatenate((Y, y), axis=0)
    X, Y = shuffle(X, Y, random_state=0)  # shuffle the data randomly

    if 1 == 1:  # diplay data in figure, set 1==0 to avoid diplaying
        plt.plot(X[np.where(Y == 1)[0], 0], X[np.where(Y == 1)[0], 1], 'r+')
        plt.plot(X[np.where(Y == -1)[0], 0], X[np.where(Y == -1)[0], 1], 'bx')
        plt.title('2 Half Moons Dataset')
        plt.show()
    return X, Y


X, Y = generate_halfmoon_data(rad, width, dist, num_data // 2)  # call function to generate data

# step 2: Initialize Perceptron
num_input = 2  # number of input neuron
b = dist / 2  # bias
err = 0  # a counter to denote the number of error outputs
eta = 0.9  # learning rate
w = np.zeros((num_input + 1, 1), dtype=np.float32)  # initial weights
w[0] = b  #
x = np.zeros((num_input + 1, 1), dtype=np.float32)
mse = np.zeros((epochs, 1), dtype=np.float32)  # store learning curve

# Step 2: training
for epoch in range(epochs):  # epochs: number of epochs
    ee = np.zeros((num_train, 1), dtype=np.float32)  # store error
    for i in range(num_train):  # in each epoch, update w for each training data
        x[0, 0] = 1
        x[1:num_input + 1, 0] = X[i, :]
        y = np.sign(w.T.dot(x))
        ee[i, 0] = Y[i, :] - y
        w = w + eta * ee[i, 0] * x
    mse[epoch, 0] = np.mean(np.square(ee))

# step 3: display learning result
plt.plot(mse, 'b-'), plt.title('Learning Curve'), plt.xlabel('Number of epochs'),
plt.ylabel('MSE'), plt.grid(True), plt.show()

# testing
Yt = np.zeros((num_test, 1), dtype=np.int)
for i in range(num_test):
    x[0, 0] = 1
    x[1:num_input + 1, 0] = X[num_train + i, :]
    Yt[i, 0] = np.sign(w.T.dot(x))

# plot testing results
plt.plot(X[num_train + np.where(Yt == 1)[0], 0], X[num_train + np.where(Yt == 1)[0], 1], 'r+')
plt.plot(X[num_train + np.where(Yt == -1)[0], 0], X[num_train + np.where(Yt == -1)[0], 1], 'bx')
plt.title('Testing Results')
x2 = np.arange(-15, 25, 1)  # draw classification line
x1 = -(w[0, 0] + w[1, 0] * x2) / w[2, 0]
plt.plot(x2, x1, 'c--')
plt.show()

