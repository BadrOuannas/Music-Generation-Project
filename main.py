from MidiModel import MidiNet
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.activations import sigmoid


def load_datasets(datasets):
    x_file, prev_x_file, y_file = datasets
    x = np.load(x_file, allow_pickle=True)
    prev_x = np.load(prev_x_file, allow_pickle=True)
    n = x.shape[0]
    y = []
    for i in range(n):
        y.append([1]+[0]*11+[1])
    y = np.vstack(y)
    # y = np.load(y_file, allow_pickle=True)
    return x, prev_x, y


batch_size = 72
pitch_range = 128
lr = 0.00005
beta1 = 0.5

datasets = ['data_x.npy', 'prev_x.npy', 'chords.npy']
x, prev_x, chords = load_datasets(datasets)
x = tf.transpose(x, (0, 1, 3, 2))
prev_x = tf.transpose(prev_x, (0, 1, 3, 2))
# todo shuffle the datasets
print(x.shape)
print(chords.shape)
chords = tf.convert_to_tensor(chords)

dataset = tf.data.Dataset.from_tensor_slices([x, prev_x])
dataset = dataset.batch(batch_size)

# x = tf.data.Dataset.from_tensor_slices(x)
# prev_x = tf.data.Dataset.from_tensor_slices(prev_x)
chords = tf.data.Dataset.from_tensor_slices(chords)
#
# x = x.batch(batch_size)
# prev_x = prev_x.batch(batch_size)
chords = chords.batch(batch_size)

epochs = 20
gan = MidiNet(chords, pitch_range, batch_size)

gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=beta1),
    g_optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=beta1),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

gan.fit(
    [x, prev_x],
    epochs=epochs
)
