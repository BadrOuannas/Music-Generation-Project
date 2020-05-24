import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activ

from tensorflow import keras


"""
    Architecture based on:
        https://arxiv.org/pdf/1908.05551.pdf
"""


class Generator(keras.Model):
    def __init__(self, pitch_range, batch_size):
        super(Generator, self).__init__()
        self.relu = layers.Dense(400, activation="relu", input_shape=(batch_size, 1, 128, 16))
        self.lstm1 = layers.LSTM(400, activation="tanh", return_sequences=True)
        self.lstm2 = layers.LSTM(400, activation="tanh", return_sequences=True)
        self.linear = layers.Dense(3, activation="linear")

    def call(self, inputs, **kwargs):
        x, noise = inputs

        noise = tf.cast(noise, dtype=float)
        x = tf.cast(x, dtype=float)
        x = tf.convert_to_tensor(x)
        noise = tf.convert_to_tensor(noise)

        lstm_input = layers.Concatenate()([noise, x])
        return self.linear(self.lstm2(self.lstm1(self.relu(lstm_input))))


class Discriminator(keras.Model):
    def __init__(self, pitch_range, batch_size):
        super(Discriminator, self).__init__()
        self.lstm1 = layers.LSTM(400, activation="tanh", return_sequences=True, input_shape=(batch_size, 1, 128, 16))
        self.lstm2 = layers.LSTM(400, activation="tanh", return_sequences=True)
        self.output = layers.Dense(2, activation="sigmoid")


    def call(self, inputs, **kwargs):
        x = inputs

        x = tf.cast(x, dtype=float)
        
        return self.output(self.lstm2(self.lstm1(x)))