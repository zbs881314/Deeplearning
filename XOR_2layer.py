# Exercise 6.1 of EECE680C Lecture 06
# implement 2-layer ANN to learn the XOR problem
import numpy as np
import tensorflow as tf

# Step 1. define training data
Xtraindata = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])  # input training array X
Ytraindata = np.array([[0.], [1.], [1.], [0.]])  # output training array Y

# Step 2. define tensorflow input/output and variables
X = tf.placeholder(tf.float32, shape=[None, 2])  # tensorflow input: 4x2
Y = tf.placeholder(tf.float32, shape=[None, 1])  # tensorflow output: 4x1

W1 = tf.Variable(tf.truncated_normal([2, 2], stddev=.1))  # weight W1 2x2 matrix
B1 = tf.Variable(tf.zeros([1, 2]))  # bias B1: 1x2
W2 = tf.Variable(tf.truncated_normal([2, 1], stddev=.1))  # weight W2 1x2 matrix
B2 = tf.Variable(tf.zeros([1, 1]))  # bias B2: 1x1

# Step 3. define ANN graph with tensorflow
Z1 = tf.matmul(X, W1) + B1  # first layer: Z=ReLU(X*W1+B1)
Z2 = tf.nn.relu(Z1)
Z3 = tf.matmul(Z2, W2) + B2  # second layer: Z=ReLU(X*W2+B2)
Z = tf.nn.relu(Z3)  # this is the ANN output

# Step 3. Define tensorflow training parameters:
Err = tf.reduce_mean(tf.square(Z - Y))
train = tf.train.AdamOptimizer(0.005).minimize(Err)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Step 4. Training ANN
for epoch in range(1000):
    train.run(feed_dict={X: Xtraindata, Y: Ytraindata})
    train_err = Err.eval(feed_dict={X: Xtraindata, Y: Ytraindata})
    print("epoch %d, err %f" % (epoch, train_err))

# Step 5. Output results
Youtput = Z.eval(feed_dict={X: Xtraindata})
W1output = W1.eval()
B1output = B1.eval()
W2output = W2.eval()
B2output = B2.eval()

print("ANN Final output =")
print(Youtput.T)

# a set of reference values for comparision purpose:
# W1=np.array([[1.0, 1.0],[1.0, 1.0]])  # weight matrix W1
# B1=np.array([[0,-1]])  # bias B1=[b1,b2]
# W2=np.array([[1.0],[-2.0]]) # weight matrix W2
# B2=np.array([[0]])   # bias B2=b3