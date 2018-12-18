
# By Fan Zhang for eece680 hw3
from scipy.io import loadmat 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)
'''
data=loadmat('imagedata.mat') # load the data set: 240 images, 3 flowers (80 images per flower)
                              # each image is 128*128*3.
Xtrain=data['Xtrain']         # training data set: 210 images (70 images per flower). 210*128*128*3
Ytrain=data['Ytrain']         # one-hot vector [1,0,0], [0,1,0], [0,0,1].             210*3
Xtest=data['Xtest']           # testing data set: 30 images (10 images per flower).    30*128*128*3
Ytest=data['Ytest']           # one-hot vector [1,0,0], [0,1,0], [0,0,1]               30*3


plt.imshow(Xtrain[1,:,:,:])   # example: display an image and its label
print(Ytrain[1,:])
plt.show()
'''

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 128, 128, 3])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=3)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=labels, logits=logits)
  #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=tf.argmax(labels, axis=1), predictions=predictions["classes"]),
      "recall" : tf.metrics.recall(
          labels=tf.argmax(labels, axis=1), predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data
    data = loadmat('imagedata.mat')  # load the data set: 240 images, 3 flowers (80 images per flower)
    # each image is 128*128*3.
    Xtrain = np.float32(data['Xtrain'])  # training data set: 210 images (70 images per flower). 210*128*128*3
    Ytrain = np.float32(data['Ytrain'])  # one-hot vector [1,0,0], [0,1,0], [0,0,1].             210*3
    Xtest = np.float32(data['Xtest'])  # testing data set: 30 images (10 images per flower).    30*128*128*3
    Ytest = np.float32(data['Ytest'])  # one-hot vector [1,0,0], [0,1,0], [0,0,1]               30*3


    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(log_device_placement=True,
                                      gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4)))	
    # Create the Estimator
    flower_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./convnet_model",config = run_config)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": Xtrain},
        y=Ytrain,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    flower_classifier.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": Xtest},
        y=Ytest,
        num_epochs=1,
        shuffle=False)
    eval_results = flower_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)

    predictions = list(flower_classifier.predict(input_fn=eval_input_fn))
    predicted_classes = [p["classes"] for p in predictions]

    print(
        "New Samples, Class Predictions:    {}\n"
        .format(predicted_classes))

if __name__ == "__main__":
    tf.app.run()
