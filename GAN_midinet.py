import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activ

from tensorflow import keras


# from MidiNet.ops import *


def conv_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv_prev_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    if x_shapes[:2] == y_shapes[:2]:
        return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)
    else:
        print(x_shapes[:2])
        print(y_shapes[:2])

class Generator(keras.Model):
    def __init__(self, pitch_range, batch_size):
        super(Generator, self).__init__()
        self.gf_dim = 64
        self.y_dim = 13
        self.n_channel = 256
        self.batch_size = batch_size

        self.relu = layers.ReLU()
        self.lrelu = layers.LeakyReLU(alpha=0.2)
        self.batch_norm = lambda x: \
            layers.BatchNormalization(epsilon=1e-05, momentum=0.9, scale=True)(x)

        self.h1 = layers.Conv2DTranspose(pitch_range, kernel_size=(2, 1), strides=(2, 2), output_padding=0)
        self.h2 = layers.Conv2DTranspose(pitch_range, kernel_size=(2, 1), strides=(2, 2), output_padding=0)
        self.h3 = layers.Conv2DTranspose(pitch_range, kernel_size=(2, 1), strides=(2, 2), output_padding=0)
        self.h4 = layers.Conv2DTranspose(1, kernel_size=(1, pitch_range), strides=(1, 2), output_padding=0)

        self.h0_prev = layers.Conv2D(16, kernel_size=(1, pitch_range), strides=(1, 2))
        self.h1_prev = layers.Conv2D(16, kernel_size=(2, 1), strides=(2, 2))
        self.h2_prev = layers.Conv2D(16, kernel_size=(2, 1), strides=(2, 2))
        self.h3_prev = layers.Conv2D(16, kernel_size=(2, 1), strides=(2, 2))

        self.linear1 = lambda z, dim: layers.Dense(1024, input_dim=dim)(z)
        self.linear2 = lambda h0, dim: layers.Dense(self.gf_dim * 2 * 2 * 1, input_dim=dim)(h0)

    def call(self, inputs, **kwargs):
        z, prev_x, y = inputs

        z = tf.convert_to_tensor(z)
        z = tf.cast(z, dtype=float)

        prev_x = tf.convert_to_tensor(prev_x)
        prev_x = tf.cast(prev_x, dtype=float)

        if y is not None:
            y = tf.convert_to_tensor(y)
            y = tf.cast(y, dtype=float)

        h0_prev = self.lrelu(self.batch_norm(self.h0_prev(prev_x)))  # [72, 16, 16, 1]
        h1_prev = self.lrelu(self.batch_norm(self.h1_prev(h0_prev)))  # [72, 16, 8, 1]
        h2_prev = self.lrelu(self.batch_norm(self.h2_prev(h1_prev)))  # [72, 16, 4, 1]
        h3_prev = self.lrelu(self.batch_norm(self.h3_prev(h2_prev)))  # [72, 16, 2, 1]

        if y is None:
            h0 = self.relu(self.batch_norm(self.linear1(z, z.get_shape()[1])))  # (72,1024)

            h1 = self.relu(self.batch_norm(self.linear2(h0, h0.get_shape()[1])))  # (72, 256)
            h1 = tf.reshape(h1, [self.batch_size, self.gf_dim * 2, 2, 1])  # (72,128,2,1)
            h1 = conv_prev_concat(h1, h3_prev)  # (72, 157, 2, 1)

            h2 = self.relu(self.batch_norm(self.h1(h1)))  # (72, 128, 4, 1)
            h2 = conv_prev_concat(h2, h2_prev)  # ([72, 157, 4, 1])

            h3 = self.relu(self.batch_norm(self.h2(h2)))  # ([72, 128, 8, 1])
            h3 = conv_prev_concat(h3, h1_prev)  # ([72, 157, 8, 1])

            h4 = self.relu(self.batch_norm(self.h3(h3)))  # ([72, 128, 16, 1])
            h4 = conv_prev_concat(h4, h0_prev)  # ([72, 157, 16, 1])

            g_x = activ.sigmoid(self.h4(h4))  # ([72, 1, 16, 128])

        else:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])  # (72,13,1,1)

            z = tf.concat([z, y], 1)  # (72,113)

            h0 = self.relu(self.batch_norm(self.linear1(z, z.get_shape()[1])))  # (72,1024)
            h0 = tf.concat([h0, y], 1)  # (72,1037)

            h1 = self.relu(self.batch_norm(self.linear2(h0, h0.get_shape()[1])))  # (72, 256)
            h1 = tf.reshape(h1, [self.batch_size, 2, 1, self.gf_dim * 2])  # (72,128,2,1)
            h1 = conv_concat(h1, yb)  # (72,141,2,1)
            h1 = conv_prev_concat(h1, h3_prev)  # (72, 157, 2, 1)

            h2 = self.relu(self.batch_norm(self.h1(h1)))  # (72, 128, 4, 1)
            h2 = conv_concat(h2, yb)  # ([72, 141, 4, 1])
            h2 = conv_prev_concat(h2, h2_prev)  # ([72, 157, 4, 1])

            h3 = self.relu(self.batch_norm(self.h2(h2)))  # ([72, 128, 8, 1])
            h3 = conv_concat(h3, yb)  # ([72, 141, 8, 1])
            h3 = conv_prev_concat(h3, h1_prev)  # ([72, 157, 8, 1])

            h4 = self.relu(self.batch_norm(self.h3(h3)))  # ([72, 128, 16, 1])
            h4 = conv_concat(h4, yb)  # ([72, 141, 16, 1])
            h4 = conv_prev_concat(h4, h0_prev)  # ([72, 157, 16, 1])

            g_x = activ.sigmoid(self.h4(h4))  # ([72, 1, 16, 128])

        return g_x


class Discriminator(keras.Model):
    def __init__(self, pitch_range, batch_size):
        super(Discriminator, self).__init__()

        self.batch_size = batch_size
        self.df_dim = 64
        self.dfc_dim = 1024
        self.y_dim = 13
        self.c_dim = 1  # number of midi tracks

        self.lrelu = layers.LeakyReLU(alpha=0.2)  # leaky relu
        self.batch_norm = lambda x: \
            layers.BatchNormalization(epsilon=1e-05, momentum=0.9, scale=True)(x)

        self.h0_prev = layers.Conv2D(self.c_dim + self.y_dim, kernel_size=(2, pitch_range), strides=(2, 2))
        # out channels = y_dim +1
        self.h1_prev = layers.Conv2D(self.df_dim + self.y_dim, kernel_size=(4, 1), strides=(2, 2))
        # out channels = df_dim + y_dim
        self.linear1 = lambda h1, dim: self.lrelu(layers.Dense(self.dfc_dim, input_dim=dim)(h1))
        self.linear2 = lambda h2, dim: layers.Dense(1, input_dim=dim)(h2)

    def call(self, inputs, **kwargs):

        x, y = inputs

        x = tf.convert_to_tensor(x)
        x = tf.cast(x, dtype=float)

        if y is not None:
            y = tf.convert_to_tensor(y)
            y = tf.cast(y, dtype=float)


        if y is None:
            h0 = self.lrelu(self.h0_prev(x))
            fm = h0

            h1 = self.lrelu(self.batch_norm(self.h1_prev(h0)))
            h1 = tf.reshape(h1, [self.batch_size, -1])

            h2 = self.lrelu(self.batch_norm(self.linear1(h1, h1.get_shape()[1])))
            h3 = self.linear2(h2, h2.get_shape()[1])
            h3_sigmoid = activ.sigmoid(h3)
        else:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_concat(x, yb)

            h0 = self.lrelu(self.h0_prev(x))
            fm = h0
            h0 = conv_concat(h0, yb)

            h1 = self.lrelu(self.batch_norm(self.h1_prev(h0)))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = tf.concat([h1, y], 1)

            h2 = self.lrelu(self.batch_norm(self.linear1(h1, h1.get_shape()[1])))
            h2 = tf.concat([h2, y], 1)

            h3 = self.linear2(h2, h2.get_shape()[1])
            h3_sigmoid = activ.sigmoid(h3)

        return h3_sigmoid, h3, fm


class Sampler(keras.Model):
    def __init__(self, pitch_range, batch_size):
        super(Sampler, self).__init__()
        self.gf_dim = 64
        self.y_dim = 13
        self.n_channel = 256
        self.batch_size = batch_size

        self.lrelu = layers.LeakyReLU(alpha=0.2)
        self.relu = layers.ReLU()
        self.batch_norm = lambda x: \
            layers.BatchNormalization(epsilon=1e-05, momentum=0.9, scale=True)(x)

        self.h1 = layers.Conv2DTranspose(pitch_range, kernel_size=(2, 1), strides=(2, 2), data_format="channels_first")
        self.h2 = layers.Conv2DTranspose(pitch_range, kernel_size=(2, 1), strides=(2, 2), data_format="channels_first")
        self.h3 = layers.Conv2DTranspose(pitch_range, kernel_size=(2, 1), strides=(2, 2), data_format="channels_first")
        self.h4 = layers.Conv2DTranspose(1, kernel_size=(1, pitch_range), strides=(1, 2), data_format="channels_first")

        self.h0_prev = layers.Conv2D(16, kernel_size=(1, pitch_range), strides=(1, 2), data_format="channels_first")
        self.h1_prev = layers.Conv2D(16, kernel_size=(2, 1), strides=(2, 2), data_format="channels_first")
        self.h2_prev = layers.Conv2D(16, kernel_size=(2, 1), strides=(2, 2), data_format="channels_first")
        self.h3_prev = layers.Conv2D(16, kernel_size=(2, 1), strides=(2, 2), data_format="channels_first")

        self.linear1 = lambda z, dim: self.lrelu(layers.Dense(1024, input_dim=dim)(z))  # (113, 1024)
        self.linear2 = lambda h0, dim: self.lrelu(
            layers.Dense(self.gf_dim * 2 * 2 * 1, input_dim=dim)(h0))  # (1037, self.gf_dim * 2 * 2 * 1)

    def call(self, inputs, **kwargs):
        z, prev_x, y = inputs

        h0_prev = self.lrelu(self.batch_norm(self.h0_prev(prev_x)))  # [72, 16, 16, 1]
        h1_prev = self.lrelu(self.batch_norm(self.h1_prev(h0_prev)))  # [72, 16, 8, 1]
        h2_prev = self.lrelu(self.batch_norm(self.h2_prev(h1_prev)))  # [72, 16, 4, 1]
        h3_prev = self.lrelu(self.batch_norm(self.h3_prev(h2_prev)))  # [72, 16, 2, 1]

        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])  # (72,13,1,1)

        z = tf.concat([z, y], 1)  # (72,113)

        h0 = self.relu(self.batch_norm(self.linear1(z)))  # (72,1024)
        h0 = tf.concat([h0, y], 1)  # (72,1037)

        h1 = self.relu(self.batch_norm(self.linear2(h0)))  # (72, 256)
        h1 = tf.reshape(h1, [self.batch_size, self.gf_dim * 2, 2, 1])  # (72,128,2,1)
        h1 = conv_concat(h1, yb)  # (b,141,2,1)
        h1 = conv_prev_concat(h1, h3_prev)  # (72, 157, 2, 1)

        h2 = self.relu(self.batch_norm(self.h1(h1)))  # (72, 128, 4, 1)
        h2 = conv_concat(h2, yb)  # ([72, 141, 4, 1])
        h2 = conv_prev_concat(h2, h2_prev)  # ([72, 157, 4, 1])

        h3 = self.relu(self.batch_norm(self.h2(h2)))  # ([72, 128, 8, 1])
        h3 = conv_concat(h3, yb)  # ([72, 141, 8, 1])
        h3 = conv_prev_concat(h3, h1_prev)  # ([72, 157, 8, 1])

        h4 = self.relu(self.batch_norm(self.h3(h3)))  # ([72, 128, 16, 1])
        h4 = conv_concat(h4, yb)  # ([72, 141, 16, 1])
        h4 = conv_prev_concat(h4, h0_prev)  # ([72, 157, 16, 1])

        g_x = activ.sigmoid(self.h4(h4))  # ([72, 1, 16, 128])

        return g_x
