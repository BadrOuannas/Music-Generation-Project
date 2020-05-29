import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activ

from tensorflow import keras

"""
    Architecture based on:
        https://arxiv.org/pdf/1908.05551.pdf
"""


class Generator(keras.Model):
    def __init__(self, pitch_range, time_steps, noise_dim, depth):
        super(Generator, self).__init__()
        self.depth = depth
        self.pitch_range = pitch_range
        self.time_steps = time_steps

        self.lrelu = layers.LeakyReLU(alpha=0.3)  # leaky relu
        self.batch_norm = lambda x: \
            layers.BatchNormalization(epsilon=1e-05, momentum=0.9, scale=True)(x)

        self.relu = layers.Dense(400, activation="relu", input_shape=(time_steps, pitch_range))
        self.lstm = layers.LSTM(400, activation="tanh", return_sequences=True)
        self.linear = layers.Dense(pitch_range, activation="linear")

    def call(self, noise, **kwargs):
        noise = tf.cast(noise, dtype=float)
        # x = tf.cast(x, dtype=float)
        # x = tf.convert_to_tensor(x)
        # noise = tf.convert_to_tensor(noise)

        # lstm_input = layers.Concatenate()([noise, x])

        o = self.batch_norm(self.relu(noise))
        for _ in range(self.depth):
            o = self.batch_norm(self.lstm(o))

        o = self.linear(o)
        o = tf.reshape(o, [-1, self.time_steps, self.pitch_range])
        return o


class Discriminator(keras.Model):
    def __init__(self, pitch_range, time_steps, depth):
        super(Discriminator, self).__init__()
        self.depth = depth
        self.pitch_range = pitch_range
        self.time_steps = time_steps
        self.units = 400

        self.lrelu = layers.LeakyReLU(alpha=0.3)  # leaky relu
        self.batch_norm = lambda x: \
            layers.BatchNormalization(epsilon=1e-05, momentum=0.9, scale=True)(x)

        self.lstm1 = layers.LSTM(400, activation="tanh", return_sequences=True)
        self.lstm2 = layers.LSTM(400, activation="tanh", return_sequences=True)
        self.linear = layers.Dense(1, activation="sigmoid")

    def call(self, x, **kwargs):
        x = tf.cast(x, dtype=float)

        o = self.batch_norm(self.lstm1(x))
        for _ in range(self.depth - 1):
            o = self.batch_norm(self.lstm2(o))
        shape = o.get_shape()
        o = tf.reshape(o, [-1, shape[1]*shape[2]])
        return self.linear(o)
