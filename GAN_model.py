import os
import PIL
import time
import numpy as np
import collections
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers


def make_generator():
    model = tf.keras.Sequential()
    #setup correct structure, example:
    model.add(layers.LSTM(350))
    model.add(layers.LSTM(350))

    return model

def generator_loss(fake_output):
    # change to the wanted loss function
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def make_discriminator():
    model = tf.keras.Sequential()
    # Set up correct structure

    return model

def discriminator_loss(real_output, fake_output):
    # change to the wanted loss function
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


@tf.function
def train_step(images):
    # Example:
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_melody = generator(noise, training=True)

        real_output = discriminator(melody, training=True)
        fake_output = discriminator(generated_melody, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

    for image_batch in dataset:
        train_step(image_batch)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


if __name__ == "__main__":
    # short tests without training:
    generator = make_generator()

    noise = tf.random.normal([1, 100])                  # change these depending on architecture
    generated_melody = generator(noise, training=False)

    discriminator = make_discriminator()

    decision = discriminator(generated_melody)

    # set optimizers before training:
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


    # setup training and then train
