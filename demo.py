import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from random import random, randint, seed
from pypianoroll import Track, Multitrack
from sklearn.utils import shuffle
from MidiModel import MidiNet


# building the chord map see paper table 2
def build_chord_map():
        c_maj = [60, 64, 67]
        c_min = [60, 63, 67]
        chord_map = []
        chord_list_maj = []
        chord_list_min = []
        chord_list_maj.append(c_maj)
        chord_list_min.append(c_min)
        for i in range(11):
            chord = [x + 1 for x in c_maj]
            c_maj = chord
            chord_list_maj.append(chord)
            chord_min = [x + 1 for x in c_min]
            chord_list_min.append(chord_min)
            c_min = chord_min
        chord_map.append(chord_list_maj)
        chord_list_min[:] = chord_list_min[9:] + chord_list_min[0:9]
        chord_map.append(chord_list_min)
        return chord_map


def find_pitch(song, volume):  # song shape(128,128), which is (time step, pitch)
    for time in range(song.shape[0]):
        step = song[time, :]
        max_index = np.argmax(step)
        for i in range(len(step)):
            if i == max_index:
                song[time, i] = volume
            else:
                song[time, i] = 0
    return song


def make_chord_track(chord, instrument, volume):
    pianoroll = np.zeros((128, 128))
    for i in range(len(chord)):
        st = 16 * i
        ed = st + 16
        chord_pitch = chord[i]
        pianoroll[st:ed, chord_pitch] = volume
    track = Track(pianoroll=pianoroll, program=instrument, is_drum=False,
                  name='chord')
    return track


def main():
    songs = np.load('./samples/songs.npy') # array of shape (N, 128, 128) N number of songs sampled with the generator
    chords = np.load('./samples/chords.npy')

    volume = 200
    instrument = 25  # guitar; see https://www.midi.org/specifications/item/gm-level-1-sound-set for more info
    chord_map = build_chord_map()

    for i in range(songs.shape[0]):
        current_song = find_pitch(songs[i], volume).astype(int)
        current_chords = chords[i]

        song_track = Track(pianoroll=current_song, program=instrument, is_drum=False)
        # fig, axs = song_track.plot()
        # plt.show()

        maj_min = list(current_chords[:, -1].astype(int))
        keys = [int(np.where(current_chords[i, :-1] == 1)[0][0]) for i in range(current_chords.shape[0])]

        style = []
        for m, k in zip(maj_min, keys):
            style.append(chord_map[m][k])

        chord_track = make_chord_track(style, instrument, volume)

        multitrack = Multitrack(tracks=[song_track, chord_track], tempo=120.0, beat_resolution=4)
        multitrack.write("./songs/gen_midi{}_instrument{}.mid".format(i, instrument))


    # shape = prev_x.shape
    # pianorolls = []
    #
    #
    # # for i in range(1,2):
    # for _ in range(num_samples):
    #     sample = model.generator([noise, , chords])
    #     for i in range(batch_size):
    #         pianorolls.append(sample[i, :, :, :].reshape(shape[1], shape[2])*200)
    #
    # pianoroll = np.vstack(pianorolls)
    # print(pianoroll.shape)
    # track = Track(pianoroll=current_song, program=0, is_drum=False)
    #
    # multitrack = Multitrack(tracks=[track])
    # multitrack.write('test.mid')
    #
    # fig, axs = multitrack.plot()
    # plt.show()


if __name__ == '__main__':
    main()