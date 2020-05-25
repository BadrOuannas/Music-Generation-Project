from MidiModel import MidiNet
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle
from tensorflow import keras
import matplotlib.pyplot as plt


"""
Code for training the GAN model for Midinet
"""


def load_datasets(datasets):
    x_file, prev_x_file, y_file = datasets
    x = np.load(x_file, allow_pickle=True)
    prev_x = np.load(prev_x_file, allow_pickle=True)
    n = x.shape[0]
    y = []
    for i in range(n):
        y.append([1] + [0] * 11 + [1])
    y = np.vstack(y)
    # y = np.load(y_file, allow_pickle=True)
    return x, prev_x, y


def main():
    batch_size = 72
    pitch_range = 128
    lr = 0.00005
    beta1 = 0.5
    noise_dim = 100
    num_epochs = 20

    datasets = ['data_x.npy', 'prev_x.npy', 'chords.npy']
    x, prev_x, chords = load_datasets(datasets)
    x = np.transpose(x, (0, 3, 2, 1))
    prev_x = np.transpose(prev_x, (0, 3, 2, 1))
    print(x.shape)
    print(chords.shape)

    d_optim = keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)
    g_optim = keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)

    d_loss_list = []
    g_loss_list = []

    model = MidiNet(pitch_range, batch_size)
    model.compile(d_optim, g_optim, tf.nn.sigmoid_cross_entropy_with_logits)

    for epoch in range(num_epochs):
        d_loss_avg = tf.keras.metrics.Mean()
        g_loss_avg = tf.keras.metrics.Mean()
        x, prev_x, chords = shuffle(x, prev_x, chords)

        batch_idxs = x.shape[0] // batch_size

        for idx in range(0, batch_idxs):
            batch_x = x[idx * batch_size:(idx + 1) * batch_size]
            batch_prev_x = prev_x[idx * batch_size:(idx + 1) * batch_size]
            chord_cond = chords[idx * batch_size:(idx + 1) * batch_size]

            labels_real = tf.ones((batch_size, 1))
            labels_fake = tf.zeros((batch_size, 1))

            # Train the discriminator on real data
            with tf.GradientTape() as tape:
                _, pred, fm = model.discriminator([batch_x, chord_cond])
                d_loss_real = model.loss_fn(labels_real, pred)
            grads = tape.gradient(d_loss_real, model.discriminator.trainable_weights)
            model.d_optimizer.apply_gradients(
                zip(grads, model.discriminator.trainable_weights)
            )

            # Train the discriminator on data from generator
            noise = tf.random.normal(shape=(batch_size, noise_dim))
            gen_midi = model.generator([noise, batch_prev_x, chord_cond])

            with tf.GradientTape() as tape:
                _, pred_, fm_ = model.discriminator([gen_midi, chord_cond])
                d_loss_fake = model.loss_fn(labels_fake, pred_)
            grads = tape.gradient(d_loss_fake, model.discriminator.trainable_weights)
            model.d_optimizer.apply_gradients(
                zip(grads, model.discriminator.trainable_weights)
            )

            # train generator
            with tf.GradientTape() as tape:
                _, pred_, fm_ = model.discriminator([model.generator([noise, batch_prev_x, chord_cond]), chord_cond])
                _, pred, fm = model.discriminator([batch_x, chord_cond])

                g_loss0 = tf.math.reduce_mean(model.loss_fn(pred_, labels_real))

                # Feature Matching
                features_from_g = tf.math.reduce_mean(fm_, axis=0)
                features_from_i = tf.math.reduce_mean(fm, axis=0)
                fm_g_loss1 = tf.math.multiply(tf.nn.l2_loss(features_from_g - features_from_i), 0.1)

                mean_image_from_g = tf.math.reduce_mean(gen_midi, axis=0)
                mean_image_from_i = tf.math.reduce_mean(batch_x, axis=0)
                fm_g_loss2 = tf.math.multiply(tf.nn.l2_loss(mean_image_from_g - tf.cast(mean_image_from_i, dtype=float)), 0.01)

                g_loss = g_loss0 + fm_g_loss1 + fm_g_loss2

            grads = tape.gradient(g_loss, model.generator.trainable_weights)
            model.d_optimizer.apply_gradients(
                zip(grads, model.generator.trainable_weights)
            )

            # train generator again!
            with tf.GradientTape() as tape:
                _, pred_, fm_ = model.discriminator([model.generator([noise, batch_prev_x, chord_cond]), chord_cond])
                _, pred, fm = model.discriminator([batch_x, chord_cond])

                g_loss0 = tf.math.reduce_mean(model.loss_fn(pred_, labels_real))

                # Feature Matching
                features_from_g = tf.math.reduce_mean(fm_, axis=0)
                features_from_i = tf.math.reduce_mean(fm, axis=0)
                fm_g_loss1 = tf.math.multiply(tf.nn.l2_loss(features_from_g - features_from_i), 0.1)

                mean_image_from_g = tf.math.reduce_mean(gen_midi, axis=0)
                mean_image_from_i = tf.math.reduce_mean(batch_x, axis=0)
                fm_g_loss2 = tf.math.multiply(
                    tf.nn.l2_loss(mean_image_from_g - tf.cast(mean_image_from_i, dtype=float)), 0.01)

                g_loss = g_loss0 + fm_g_loss1 + fm_g_loss2

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
    # model.generator.save_weights("./weights/gen_weights", save_format='tf')
    plt.plot(np.arange(len(d_loss_list)), d_loss_list, 'cx', label="d_loss")
    plt.plot(np.arange(len(g_loss_list)), g_loss_list, 'bx', label="g_loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
