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



def train():
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


# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 20

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss


  # End epoch
  train_loss_results.append(epoch_loss_avg.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
