from __future__ import print_function

__author__ = "shekkizh"
"""
Tensorflow implementation of Wasserstein GAN
"""
import numpy as np
import tensorflow as tf
from models.GAN_models import *

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/CelebA_GAN_logs/", "path to logs directory")
#tf.flags.DEFINE_string("data_dir", "/Users/Zhongyu/Downloads/celebA/", "path to dataset")
tf.flags.DEFINE_string("data_dir", "/home/paperspace/Downloads/", "path to dataset")
tf.flags.DEFINE_integer("z_dim", "100", "size of input vector to generator")
tf.flags.DEFINE_float("learning_rate", "2e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("optimizer_param", "0.5", "beta1 for Adam optimizer / decay for RMSProp")
tf.flags.DEFINE_float("iterations", "1e5", "No. of iterations to train model")
tf.flags.DEFINE_string("image_size", "128,128", "Size of actual images, Size of images to be generated at.")
#tf.flags.DEFINE_string("image_size", "108,64", "Size of actual images, Size of images to be generated at.")
tf.flags.DEFINE_integer("model", "0", "Model to train. 0 - GAN, 1 - WassersteinGAN, 2 - AuxiliaryConditoinalGAN")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer to use for training")
tf.flags.DEFINE_integer("gen_dimension", "32", "dimension of first layer in generator") #defualt 16
tf.flags.DEFINE_string("mode", "train", "train / visualize model")
tf.flags.DEFINE_integer("num_cls", "2", "number of classes for ACGAN")
tf.flags.DEFINE_boolean("feature_matching", "True", "use improved gan loss/feature matching or not")


def main(argv=None):
    gen_dim = FLAGS.gen_dimension
    generator_dims = [64 * gen_dim, 64 * gen_dim // 2, 64 * gen_dim // 4, 64 * gen_dim // 8, 64 * gen_dim // 16, 3]
    discriminator_dims = [3, 64, 64 * 2, 64 * 4, 64 * 8, 64 * 16, 1]

    crop_image_size, resized_image_size = map(int, FLAGS.image_size.split(','))
    if FLAGS.model == 0:
        model = GAN(FLAGS.z_dim, crop_image_size, resized_image_size, FLAGS.batch_size, FLAGS.data_dir)
    elif FLAGS.model == 1:
        model = WassersteinGAN(FLAGS.z_dim, crop_image_size, resized_image_size, FLAGS.batch_size, FLAGS.data_dir,
                               clip_values=(-0.01, 0.01), critic_iterations=5)
    elif FLAGS.model == 2:
        model = ACGAN(FLAGS.z_dim, FLAGS.num_cls, crop_image_size, resized_image_size, FLAGS.batch_size, FLAGS.data_dir)
    elif FLAGS.model == 3:
        model = WassersteinACGAN(FLAGS.z_dim, FLAGS.num_cls, crop_image_size, resized_image_size, FLAGS.batch_size, FLAGS.data_dir,
                               clip_values=(-0.01, 0.01), critic_iterations=5)

    else:
        raise ValueError("Unknown model identifier - FLAGS.model=%d" % FLAGS.model)

    model.create_network(generator_dims, discriminator_dims, FLAGS.optimizer, FLAGS.learning_rate,
                         FLAGS.optimizer_param, FLAGS.feature_matching)

    model.initialize_network(FLAGS.logs_dir, FLAGS.iterations)

    if FLAGS.mode == "train":
        model.train_model(int(1 + FLAGS.iterations))
    elif FLAGS.mode == "visualize":
        model.visualize_model(FLAGS.iterations, generator_dims)


if __name__ == "__main__":
    tf.app.run()
