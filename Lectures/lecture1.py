# Import modules
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Create (smallest possible) Neural Network with 1 layer, 1 nueron and input shape of 1 value
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# Specify 2 functions, a loss and an optimiser
model.compile(optimizer='sgd', loss='mean_squared_error')

# Provide the data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
# Train the Neural Network
model.fit(xs, ys, epochs=500)

# Model can predict value
print(model.predict([10.0]))