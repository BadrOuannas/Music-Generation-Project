import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optim
import tensorflow.keras.activations as activ

from MidiNet.ops import *


class generator(layers.Layer):
    def __init__(self, pitch_range):
        super(generator, self).__init__()
        self.gf_dim = 64
        self.y_dim = 13
        self.n_channel = 256

        self.lrelu = layers.LeakyReLU(alpha=0.2)
        self.batch_norm = lambda x, axis: layers.BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.9, scale=True)(
            x)

        self.h1 = layers.Conv2DTranspose(kernel_size=(2, 1), strides=(2, 2))
        self.h2 = layers.Conv2DTranspose(kernel_size=(2, 1), strides=(2, 2))
        self.h3 = layers.Conv2DTranspose(kernel_size=(2, 1), strides=(2, 2))
        self.h4 = layers.Conv2DTranspose(kernel_size=(1, pitch_range), strides=(1, 2))

        self.h0_prev = layers.Conv2D(kernel_size=(1, pitch_range), strides=(1, 2))
        self.h1_prev = layers.Conv2D(kernel_size=(2, 1), strides=(2, 2))
        self.h2_prev = layers.Conv2D(kernel_size=(2, 1), strides=(2, 2))
        self.h3_prev = layers.Conv2D(kernel_size=(2, 1), strides=(2, 2))

        self.linear1 = lambda z, shape: layers.Dense(1024, input_dim=shape, activation='relu')(z)
        self.linear2 = lambda h0, shape: layers.Dense(self.gf_dim * 2 * 2 * 1, input_dim=shape, activation='relu')(h0)

    def forward(self, z, prev_x, y, batch_size, pitch_range):
        h0_prev = self.lrelu(self.batch_norm(self.h0_prev(prev_x), self.h0_prev(prev_x).shape[1]))  # [72, 16, 16, 1]
        h1_prev = self.lrelu(self.batch_norm(self.h1_prev(h0_prev), self.h0_prev(prev_x).shape[1]))  # [72, 16, 8, 1]
        h2_prev = self.lrelu(self.batch_norm(self.h2_prev(h1_prev), self.h0_prev(prev_x).shape[1]))  # [72, 16, 4, 1]
        h3_prev = self.lrelu(self.batch_norm(self.h3_prev(h2_prev), self.h0_prev(prev_x).shape[1]))  # [72, 16, 2, 1])

        yb = y.view(batch_size, self.y_dim, 1, 1)  # (72,13,1,1)

        z = tf.concat((z, y), 1)  # (72,113)

        # h0 = activ.relu(self.linear1(z))  # (72,1024)
        h0 = self.linear1(z, z.shape[1])  # layer.Dense(1024, input_dim=z.shape[1], activation='relu')(z)
        h0 = tf.concat((h0, y), 1)  # (72,1037)

        # h1 = activ.relu(self.linear2(h0))  # (72, 256)
        h1 = self.linear2(h0, h0.shape[
            1])  # layer.Dense(self.gf_dim * 2 * 2 * 1, input_dim=h0.shape[1], activation='relu')(h0)
        h1 = h1.view(batch_size, self.gf_dim * 2, 2, 1)  # (72,128,2,1)
        h1 = conv_cond_concat(h1, yb)  # (b,141,2,1)
        h1 = conv_prev_concat(h1, h3_prev)  # (72, 157, 2, 1)

        h2 = activ.relu(self.batch_norm(self.h1(h1)))  # (72, 128, 4, 1)
        h2 = conv_cond_concat(h2, yb)  # ([72, 141, 4, 1])
        h2 = conv_prev_concat(h2, h2_prev)  # ([72, 157, 4, 1])

        h3 = activ.relu(self.batch_norm(self.h2(h2)))  # ([72, 128, 8, 1])
        h3 = conv_cond_concat(h3, yb)  # ([72, 141, 8, 1])
        h3 = conv_prev_concat(h3, h1_prev)  # ([72, 157, 8, 1])

        h4 = activ.relu(self.batch_norm(self.h3(h3)))  # ([72, 128, 16, 1])
        h4 = conv_cond_concat(h4, yb)  # ([72, 141, 16, 1])
        h4 = conv_prev_concat(h4, h0_prev)  # ([72, 157, 16, 1])

        g_x = activ.sigmoid(self.h4(h4))  # ([72, 1, 16, 128])

        return g_x


class generator2(layers.Layer):
    def __init__(self, pitch_range):
        super(generator, self).__init__()
        # need to add ex: "input_shape=(16,), not sure what our input dimension is."
        self.relu = layers.Dense(400, activation="relu", input_shape=)
        self.lrelu = layers.LeakyReLU(alpha=0.2)
        self.lstm = layers.LSTM(400, activation="tanh", return_sequences=True)
        self.linear = layers.Dense(3, activation="linear")
    
    def forward(self, z, prev_x, y, batch_size, pitch_range):
        # TODO

class discriminator(layers.Layer):
    def __init__(self, pitch_range):
        super(discriminator, self).__init__()

        self.df_dim = 64
        self.dfc_dim = 1024
        self.y_dim = 13

        self.lrelu = layers.LeakyReLU(alpha=0.2)  # leaky relu

        self.h0_prev = layers.Conv2D(kernel_size=(2, pitch_range), stride=(2, 2))
        # out channels = y_dim +1
        self.h1_prev = layers.Conv2D(kernel_size=(4, 1), stride=(2, 2))
        # out channels = df_dim + y_dim
        self.linear1 = lambda h1, shape: self.lrelu(layers.Dense(self.dfc_dim, input_dim=shape)(h1))
        self.linear2 = lambda h2, shape: layers.Dense(1, input_dim=shape, activation='sigmoid')(h2)

    def forward(self, x, y, batch_size, pitch_range):
        yb = y.view(batch_size, self.y_dim, 1, 1)
        x_input = conv_cond_concat(x, yb)  # x.shape torch.Size([72, 14, 16, 128])

        h0 = self.lrelu(self.h0_prev(x_input), 0.2)
        fm = h0
        h0 = conv_cond_concat(h0, yb)  # torch.Size([72, 27, 8, 1])

        h1 = self.lrelu(self.batch_norm(self.h1_prev(h0)), 0.2)  # torch.Size([72, 77, 3, 1])
        h1 = h1.view(batch_size, -1)  # torch.Size([72, 231])
        h1 = tf.concat((h1, y), 1)  # torch.Size([72, 244])

        h2 = self.linear1(h1, h1.shape[1])
        h2 = tf.concat((h2, y), 1)  # torch.Size([72, 1037])

        h3 = self.linear2(h2, h2.shape[1])
        h3_sigmoid = activ.sigmoid(h3)

        return h3_sigmoid, h3, fm


class discriminator2(layers.Layer):
    def __init__(self, pitch_range):
        super(discriminator, self).__init__()

        # need to add ex: "input_shape=(16,), not sure what our input dimension is."
        self.lstm1 = layers.LSTM(400, activation="tanh", return_sequences=True, input_shape=)
        self.lstm2 = layers.LSTM(400, activation="tanh", return_sequences=True)
        self.output = layers.Dense(2, activation="sigmoid")                                     # real or fake => 2

    def forward(self, x, y, batch_size, pitch_range):
        # TODO