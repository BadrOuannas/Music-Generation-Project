from tensorflow import keras
import tensorflow as tf


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, prev, chord, num_songs=3):
        self.prev = prev
        self.num_songs = num_songs
        self.chord = chord

    def on_epoch_end(self, epoch, logs=None):
        shape = self.prev.get_shape() # todo shapes are not compatible between z and prev
        random_latent_vectors = tf.random.normal(shape=(self.num_songs, shape[1], shape[2], shape[3]))
        generated_songs = self.model.generator([random_latent_vectors, self.prev, self.chord])
        generated_songs.numpy()
        for i in range(self.num_songs):
            img = keras.preprocessing.image.array_to_img(generated_songs[i])
            img.save("generated_song_{i}_{epoch}.png".format(i=i, epoch=epoch))
