#These scripts are refer to "https://github.com/carpedm20/DCGAN-tensorflow"
import os
import scipy.misc
import numpy as np
from GAN_midinet.model import MidiNet
from GAN_midinet.utils import pp, to_json, generation_test

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 20, "Epoch to train [20]")
flags.DEFINE_float("learning_rate", 0.00005, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 72, "The size of batch [72]")
flags.DEFINE_integer("output_w", 16, "The size of the output segs to produce [16]")
flags.DEFINE_integer("output_h", 128, "The size of the output note to produce [128]")
flags.DEFINE_integer("c_dim", 1, "Number of Midi track. [1]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("dataset", "MidiNet_v1", "The name of dataset ")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("generation_test", False, "True for generation_test, False for nothing [False]")
flags.DEFINE_string("gen_dir", "gen", "Directory name to save the generate samples [samples]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.gen_dir):
        os.makedirs(FLAGS.gen_dir)

    with tf.Session() as sess:
        if FLAGS.dataset == 'MidiNet_v1':
            model = MidiNet(sess,  batch_size=FLAGS.batch_size,y_dim=13, output_w=FLAGS.output_w, output_h=FLAGS.output_h, c_dim=FLAGS.c_dim,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir, sample_dir=FLAGS.sample_dir, gen_dir=FLAGS.gen_dir)
        
        if FLAGS.is_train:
            model.train(FLAGS)
        else:
            model.load(FLAGS.checkpoint_dir)


if __name__ == '__main__':
    tf.app.run()