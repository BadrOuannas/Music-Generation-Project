from tensorflow import keras
from GAN_lstm import *

"""
Model for midinet construct the GAN model
"""


class LSTMidi(keras.Model):
    def __init__(self, pitch_range=128, time_steps=16, noise_dim=50, depth=3):
        super(LSTMidi, self).__init__()
        self.generator = Generator(pitch_range, time_steps, noise_dim, depth)
        self.discriminator = Discriminator(pitch_range, time_steps, depth)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(LSTMidi, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
