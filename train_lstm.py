import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# from random import random, seed, randint
from sklearn.utils import shuffle
from LSTMModel import *
from tensorflow import keras

"""
Code for training the GAN model for Midinet
"""


def load_dataset(x_file):
    x = np.load(x_file, allow_pickle=True)
    return x


def main():
    batch_size = 72
    pitch_range = 128
    time_steps = 16
    depth = 3

    lr = 0.0001
    beta1 = 0.75
    noise_dim = 128
    num_epochs = 50

    x = load_dataset('./data/data_x.npy')
    x = np.transpose(x, (0, 3, 2, 1))
    x = np.reshape(x, [-1, time_steps, pitch_range])
    print(x.shape)

    d_optim = keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)
    g_optim = keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)

    d_loss_list = []
    g_loss_list = []

    model = LSTMidi(pitch_range, time_steps, noise_dim, depth)
    model.compile(d_optim, g_optim, tf.nn.sigmoid_cross_entropy_with_logits)

    for epoch in range(num_epochs):
        d_loss_avg = tf.keras.metrics.Mean()
        g_loss_avg = tf.keras.metrics.Mean()
        x = shuffle(x)

        batch_idxs = x.shape[0] // batch_size

        for idx in range(0, batch_idxs):
            batch_x = x[idx * batch_size:(idx + 1) * batch_size]

            labels_real = tf.ones((batch_size, 1))
            labels_fake = tf.zeros((batch_size, 1))

            # Train the discriminator on real data
            with tf.GradientTape() as tape:
                pred = model.discriminator(batch_x)
                d_loss_real = model.loss_fn(labels_real, pred)
            grads = tape.gradient(d_loss_real, model.discriminator.trainable_weights)
            model.d_optimizer.apply_gradients(
                zip(grads, model.discriminator.trainable_weights)
            )

            # Train the discriminator on data from generator
            noise = tf.random.normal(shape=(batch_size, time_steps, noise_dim))
            gen_midi = model.generator(noise)

            with tf.GradientTape() as tape:
                pred_ = model.discriminator(gen_midi)
                d_loss_fake = model.loss_fn(labels_fake, pred_)
            grads = tape.gradient(d_loss_fake, model.discriminator.trainable_weights)
            model.d_optimizer.apply_gradients(
                zip(grads, model.discriminator.trainable_weights)
            )

            # train generator
            with tf.GradientTape() as tape:
                pred_ = model.discriminator(model.generator(noise))
                pred = model.discriminator(batch_x)

                g_loss0 = tf.math.reduce_mean(model.loss_fn(pred_, labels_real))

                # Feature Matching
                mean_image_from_g = tf.math.reduce_mean(gen_midi, axis=0)
                mean_image_from_i = tf.math.reduce_mean(batch_x, axis=0)
                fm_g_loss = tf.math.multiply(
                    tf.nn.l2_loss(mean_image_from_g - tf.cast(mean_image_from_i, dtype=float)), 0.01) # lambda1 = 0.01

                g_loss = g_loss0 + fm_g_loss

            grads = tape.gradient(g_loss, model.generator.trainable_weights)
            model.d_optimizer.apply_gradients(
                zip(grads, model.generator.trainable_weights)
            )

            # train generator again!
            with tf.GradientTape() as tape:
                pred_ = model.discriminator(model.generator(noise))
                pred = model.discriminator(batch_x)

                g_loss0 = tf.math.reduce_mean(model.loss_fn(pred_, labels_real))

                # Feature Matching
                mean_image_from_g = tf.math.reduce_mean(gen_midi, axis=0)
                mean_image_from_i = tf.math.reduce_mean(batch_x, axis=0)
                fm_g_loss = tf.math.multiply(
                    tf.nn.l2_loss(mean_image_from_g - tf.cast(mean_image_from_i, dtype=float)), 0.01)

                g_loss = g_loss0 + fm_g_loss

            grads = tape.gradient(g_loss, model.generator.trainable_weights)
            model.d_optimizer.apply_gradients(
                zip(grads, model.generator.trainable_weights)
            )

            d_loss = d_loss_real + d_loss_fake

            d_loss_avg.update_state(d_loss)
            g_loss_avg.update_state(g_loss)

        # End epoch
        d_loss_list.append(d_loss_avg.result())
        g_loss_list.append(g_loss_avg.result())

        print("Epoch {:02d}/{:02d}: D_loss: {:.3f} G_loss: {:.3f}".format(epoch, num_epochs, d_loss_avg.result(),
                                                                          g_loss_avg.result()))

    # inputs = keras.Input(shape=x.shape[1:], batch_size=batch_size)
    # model._set_inputs(inputs)
    # saving the model's generator
    path = "./models/lstm_noise{}_depth{}".format(noise_dim, depth)

    try:
        if not os.path.exists(path):
            os.mkdir(path)
        model.generator.save_weights(path+"/gen_weights", save_format='tf')
    except OSError:
        print("Creation of the directory %s failed" % path)

    plt.plot(np.arange(len(d_loss_list)), d_loss_list, 'cx-', label="d_loss")
    plt.plot(np.arange(len(g_loss_list)), g_loss_list, 'bx-', label="g_loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
