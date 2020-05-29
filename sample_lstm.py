import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from random import random, randint, seed
from pypianoroll import Track, Multitrack
from sklearn.utils import shuffle
from LSTMModel import *


def load_model(path, pitch_range, time_steps, noise_dim, depth):
    model = LSTMidi(pitch_range, time_steps, noise_dim, depth)
    model.generator.load_weights(path)
    return model

def main():
    pitch_range = 128
    batch_size = 8
    noise_dim = 100
    num_max_songs = 10

    # choosing model
    model = load_model("./models/lstm_noise100_depth3/gen_weights.index", pitch_range, batch_size)

    songs = []

    for idx in range(0, num_max_songs):
        noise = tf.random.normal(shape=(batch_size, noise_dim))

        current_song = model.generator(noise)
        songs.append(tf.reshape(current_song, [-1, data_shape[2]]))
        chords.append(chord_cond)
    songs = np.asarray(songs)
    chords = np.asarray(chords)
    np.save('./samples/songs_lstm.npy', songs)
    # np.save('./samples/chords.npy', chords)


if __name__ == '__main__':
    main()
