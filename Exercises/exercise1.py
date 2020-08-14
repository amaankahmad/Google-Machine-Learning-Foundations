# Import modules
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Create (smallest possible) Neural Network with 1 layer, 1 nueron and input shape of 1 value
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# Specify 2 functions, a loss and an optimiser
model.compile(optimizer='sgd', loss='mean_squared_error')

# Provide the data
bedrooms = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
price = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0], dtype=float)

# Train the Neural Network
model.fit(bedrooms, price, epochs=500)

# Model can predict value
print(model.predict([7.0]))