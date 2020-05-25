from tensorflow import keras
from GAN_midinet import *

"""
Model for midinet construct the GAN model
"""


class MidiNet(keras.Model):
    def __init__(self, pitch_range=128, batch_size=72):
        super(MidiNet, self).__init__()
        self.batch_size = batch_size
        self.generator = Generator(pitch_range, batch_size)
        self.discriminator = Discriminator(pitch_range, batch_size)
        self.sampler = Sampler(pitch_range, batch_size)
        # self.chords = chords

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(MidiNet, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, inputs, **kwargs):
        x, prev_x, y = inputs
        # y = self.chords
        # Sample random points in the latent space
        batch_size = tf.shape(x)[0]
        z = tf.random.normal(shape=tf.shape(x))

        # Decode them to fake images
        gen_midi = self.generator([z, prev_x, y])

        # Combine them with real images
        combined_midi = tf.concat([gen_midi, x], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels todo (maybe?)
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator([combined_midi, y])
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        z = tf.random.normal(shape=tf.shape(x))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator([self.generator(z, prev_x, y), y])
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}
