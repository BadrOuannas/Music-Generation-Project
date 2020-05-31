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
        self.dense = layers.Dense(pitch_range, activation="relu", input_shape=(pitch_range,))

        self.relu = layers.Dense(pitch_range, activation="relu", input_shape=(1, time_steps*pitch_range + 13)) # output was 400 and activ = relu
        self.lstm = layers.LSTM(pitch_range, activation="relu", return_sequences=True) # output was 400
        self.linear = layers.Dense(pitch_range, activation="linear")

    def call(self, inputs, **kwargs):
        chords, noise = inputs
        noise = tf.cast(noise, dtype=float)
        chords = tf.cast(chords, dtype=float)
        # x = tf.convert_to_tensor(x)
        # noise = tf.convert_to_tensor(noise)

        # lstm_input = layers.Concatenate()([noise, x])

        # x_input = tf.reshape(x, [-1, self.pitch_range * self.time_steps])
        # x_input = tf.concat([x, noise], axis=1)

        z = tf.concat([noise, chords], axis=1)
        z = tf.reshape(z, [-1, 1, self.pitch_range*self.time_steps + chords.get_shape()[-1]])

        o = self.batch_norm(self.relu(z))
        o = layers.Dropout(0.3)(o)
        for _ in range(self.depth):
            o = self.batch_norm(self.dense(self.lstm(o)))
            o = layers.Dropout(0.3)(o)

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

        self.dense = layers.Dense(pitch_range, activation="relu",
                                  input_shape=(time_steps, pitch_range))  # output was 400 and activ = relu

        self.lstm1 = layers.LSTM(400, activation="relu", return_sequences=True)
        self.lstm2 = layers.LSTM(400, activation="relu", return_sequences=True)
        self.linear = layers.Dense(1, activation="sigmoid")

    def call(self, x, **kwargs):
        x = tf.cast(x, dtype=float)

        o = self.batch_norm(self.dense(self.lstm1(x)))
        for _ in range(self.depth - 1):
            o = self.batch_norm(self.dense(self.lstm2(o)))
            # o = layers.Dropout(0.3)(o)
        shape = o.get_shape()
        o = tf.reshape(o, [-1, shape[1]*shape[2]])
        return self.linear(o)
