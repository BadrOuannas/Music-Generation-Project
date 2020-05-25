import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from random import random, randint, seed
from pypianoroll import Track, Multitrack
from sklearn.utils import shuffle
from MidiModel import MidiNet


def load_model(path, pitch_range, batch_size):
    model = MidiNet(pitch_range, batch_size)
    model.generator.load_weights(path)
    return model


def construct_random_chords(rs, batch_size):
    seed(rs)
    chords = np.zeros((batch_size, 13))
    for i in range(batch_size):
        u = random()
        if u < 0.5:
            chords[i, -1] = 1
        chords[i, randint(0, 11)] = 1
    return chords


def a_minor(batch_size):
    chords = np.zeros((batch_size, 13))
    chords[:, 0] = 1
    chords[:, -1] = 1
    return chords


def main():
    pitch_range = 128
    batch_size = 8
    noise_dim = 100
    num_max_songs = 10

    # choosing model
    model = load_model("./weights_random_100epochs/gen_weights.index", pitch_range, batch_size)

    prev_x = np.load('prev_x.npy')
    prev_x = np.transpose(prev_x, (0, 3, 2, 1))
    prev_x = shuffle(prev_x)
    # chords_cond = construct_random_chords(2020, batch_size)

    data_shape = prev_x.shape
    batch_idxs = prev_x.shape[0] // batch_size
    songs = []
    chords = []

    for idx in range(0, min(batch_idxs, num_max_songs)):
        seed = 2020
        batch_prev_x = prev_x[idx * batch_size:(idx + 1) * batch_size]
        chord_cond = construct_random_chords(seed, batch_size)
        # chord_cond = a_minor(batch_size)
        noise = tf.random.normal(shape=(batch_size, noise_dim))

        current_song = model.generator([noise, batch_prev_x, chord_cond])
        songs.append(tf.reshape(current_song, [-1, data_shape[2]]))
        chords.append(chord_cond)
    songs = np.asarray(songs)
    chords = np.asarray(chords)
    np.save('./samples/songs.npy', songs)
    np.save('./samples/chords.npy', chords)


if __name__ == '__main__':
    main()
