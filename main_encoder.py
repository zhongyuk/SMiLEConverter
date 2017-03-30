from __future__ import print_function

__author__ = "zhongyu kuang"
"""
Tensorflow implementation of Wasserstein GAN
"""
import numpy as np
import tensorflow as tf
from models.Encoder import *

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_string("gen_logs_dir", "logs/CelebA_GAN_logs/", "path to generator logs directory")
tf.flags.DEFINE_string("logs_dir", "encoder_logs/", "path to save encoder logs directory")
tf.flags.DEFINE_string("data_dir", "/home/paperspace/Downloads/", "path to dataset")
tf.flags.DEFINE_integer("z_dim", "100", "size of input vector to generator")
tf.flags.DEFINE_float("learning_rate", "2e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("optimizer_param", "0.5", "beta1 for Adam optimizer / decay for RMSProp")
tf.flags.DEFINE_float("iterations", "1e5", "No. of iterations to train model")
tf.flags.DEFINE_string("image_size", "128,128", "Size of actual images, Size of images to be generated at.")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer to use for training")
tf.flags.DEFINE_integer("gen_dimension", "32", "dimension of first layer in generator") #defualt 16
tf.flags.DEFINE_string("mode", "train", "train / visualize model")
tf.flags.DEFINE_integer("num_cls", "2", "number of classes")
tf.flags.DEFINE_integer("num_iter", "50", "iteration specifying ACGAN model ckpt")


def main(argv=None):
    gen_dim = FLAGS.gen_dimension
    generator_dims = [64 * gen_dim, 64 * gen_dim // 2, 64 * gen_dim // 4, 64 * gen_dim // 8, 64 * gen_dim // 16, 3]
    encoder_dims = [3, 64, 64 * 2, 64 * 4, 64 * 8, 64 * 16, 100]

    crop_image_size, resized_image_size = map(int, FLAGS.image_size.split(','))
    model = Encoder_Network(FLAGS.z_dim, FLAGS.num_cls, crop_image_size, resized_image_size, FLAGS.batch_size, FLAGS.data_dir)

    model.create_network(generator_dims, encoder_dims, FLAGS.gen_logs_dir, FLAGS.num_iter, FLAGS.optimizer, FLAGS.learning_rate, FLAGS.optimizer_param)

    model.initialize_network(FLAGS.logs_dir, FLAGS.iterations)

    if FLAGS.mode == "train":
        model.train_model(int(1 + FLAGS.iterations))
    elif FLAGS.mode == "visualize":
        model.visualize_model(FLAGS.iterations)


if __name__ == "__main__":
    tf.app.run()
