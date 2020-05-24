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

#
# class DataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
#                  n_classes=10, shuffle=True):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()
#
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))
#
#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#
#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]
#
#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)
#
#         return X, y
#
#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
#
#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size), dtype=int)
#
#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # Store sample
#             X[i,] = np.load('data/' + ID + '.npy')
#
#             # Store class
#             y[i] = self.labels[ID]
#
#         return X, keras.utils.to_categorical(y, num_classes=self.n_classes)



batch_size = 72
pitch_range = 128
lr = 0.00005
beta1 = 0.5

datasets = ['data_x.npy', 'prev_x.npy', 'chords.npy']
x, prev_x, chords = load_datasets(datasets)
# x = tf.transpose(x, (0, 1, 3, 2))
# prev_x = tf.transpose(prev_x, (0, 1, 3, 2))
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
