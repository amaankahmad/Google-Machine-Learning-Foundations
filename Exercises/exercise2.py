# Import modules
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if logs.get('accuracy')>0.99:
            print("\nReached 99% accuracy.")
            self.model.stop_training = True

# Load in MNIST data
mnist = tf.keras.datasets.mnist
# Load in images for training and testing
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Normalise grayscale values
x_train  = x_train / 255.0
x_test = x_test / 255.0

# Design the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Build the model, using Adam optimizer
model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the trianing data to the labels
model.fit(x_train, y_train, epochs=10, callbacks=[myCallback()])

# Test model with unseen data
model.evaluate(x_test, y_test)