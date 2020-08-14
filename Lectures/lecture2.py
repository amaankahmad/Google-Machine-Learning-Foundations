# Import modules
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Load in MNIST data
mnist = tf.keras.datasets.fashion_mnist
# Load in images for training and testing
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Print a training image and label to check
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

# Normalise grayscale values
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Design the model
# Sequential: Defines a sequence of layers in the neural network
# Flatten: Takes square image and turns into a 1D set
# Dense: Adds a layer of neurons
# Each layer of neurons needs an ACTIVATION FUNCTION to tell them what to do:
# Relu: If X is > 0, return X else return 0., therefore passes values greater than 0 onto the next layer
# Softmax: Sets the biggest value as 1 and the others to 0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Build the model, using Adam optimizer
model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the trianing data to the labels
model.fit(training_images, training_labels, epochs=5)

# Test model with unseen data
model.evaluate(test_images, test_labels)