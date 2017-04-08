__author__ = 'zhongyu kuang'

import tensorflow as tf
import numpy as np
import os, sys, inspect
import time

utils_folder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)

import utils as utils
import Dataset_Reader.read_celebADataset as celebA
import Read_Generator
from six.moves import xrange

class Encoder_Network(object):
    def __init__(self, z_dim, num_cls, crop_image_size, resized_image_size, batch_size, data_dir):
        self.num_cls = num_cls
        celebA_dataset = celebA.read_dataset(data_dir)
        self.z_dim = z_dim
        self.crop_image_size = crop_image_size
        self.resized_image_size = resized_image_size
        self.batch_size = batch_size
        
        label_dict = celebA.create_label_dict(data_dir)
        imgfn_list = label_dict.keys()
        label_list = [label_dict[imgfn] for imgfn in imgfn_list]

        images = tf.convert_to_tensor(imgfn_list)
        labels = tf.convert_to_tensor(label_list, dtype=np.int32)
        input_queue = tf.train.slice_input_producer([images, labels], shuffle=True)
        self.images, self.labels = self._read_input_queue(input_queue)

    def _read_input(self, queue):
        class DataRecord(object):
            pass
        label = queue[1]
        image = tf.read_file(queue[0])
        record = DataRecord()
        decoded_image = tf.image.decode_jpeg(image, channels=3)
        cropped_image = tf.cast(
            tf.image.crop_to_bounding_box(decoded_image, 55, 35, self.crop_image_size, self.crop_image_size),
            tf.float32)
        decoded_image_4d = tf.expand_dims(cropped_image, 0)
        resized_image = tf.image.resize_bilinear(decoded_image_4d, [self.resized_image_size, self.resized_image_size])
        record.input_image = tf.squeeze(resized_image, squeeze_dims=[0])
        record.input_label = label
        return record

    def _read_input_queue(self, input_queue):
        print("Setting up image reader...")
        read_input = self._read_input(input_queue)
        num_preprocess_threads = 4
        num_examples_per_epoch = 800
        min_queue_examples = int(0.1 * num_examples_per_epoch)
        print("Shuffling")
        input_image, input_label = tf.train.batch([read_input.input_image, read_input.input_label],
                                     batch_size=self.batch_size,
                                     num_threads=num_preprocess_threads,
                                     capacity=min_queue_examples + 2 * self.batch_size)
        input_image = utils.process_image(input_image, 127.5, 127.5)
        input_label = tf.one_hot(input_label, self.num_cls)
        return input_image, input_label

    def _setup_placeholder(self):
        self.train_phase = tf.placeholder(tf.bool)
        self.class_num = tf.placeholder(tf.int32)

    def _get_optimizer(self, optimizer_name, learning_rate, optimizer_param):
        self.learning_rate = learning_rate
        if optimizer_name == "Adam":
            return tf.train.AdamOptimizer(learning_rate, beta1=optimizer_param)
        else:
            raise ValueError("Unknown optimizer %s" % optimizer_name)

    def _train(self, loss, var_list, optimizer):
    	grads = optimizer.compute_gradients(loss, var_list=var_list)
    	for grad, var in grads:
    		utils.add_gradient_summary(grad, var)
    	return optimizer.apply_gradients(grads)

    def _encoder(self, dims, train_phase, activation=tf.nn.relu, scope_name="encoder"):
    	N = len(dims)
    	with tf.variable_scope(scope_name) as scope:
    		h = self.images
    		skip_bn = True
    		for index in range(N-2):
    			W = utils.weight_variable([3, 3, dims[index], dims[index+1]], name="W_%d" % index)
    			b = utils.bias_variable([dims[index+1]], name="b_%d" % index)
    			h_conv = utils.conv2d_strided(h, W, b)
    			if skip_bn:
    				h_bn = h_conv
    				skip_bn = False
    			else:
    				h_bn = utils.batch_norm(h_conv, dims[index + 1], train_phase, scope="disc_bn%d" % index)
    			h = activation(h_bn, name="h_%d" % index)
    			utils.add_activation_summary(h)

	    	shape = h.get_shape().as_list()
	    	image_size = self.resized_image_size // (2**(N-2))
	    	h_reshaped = tf.reshape(h, [self.batch_size, image_size*image_size*shape[3]])

	    	W_z = utils.weight_variable([image_size*image_size*shape[3], dims[-1]], name="W_z")
	    	b_z = utils.bias_variable([dims[-1]], name='b_z')
	    	h_z = tf.matmul(h_reshaped, W_z) + b_z
        return tf.nn.sigmoid(h_z)

    def _load_generator_data(self, num_iter=50):
        gen_params = Read_Generator.load_generator(self.gen_logs_dir, num_iter)
        return gen_params

    def _generator(self, z, dims, train_phase, num_iter, activation=tf.nn.relu, scope_name="generator"):
        N = len(dims)
        image_size = self.resized_image_size // (2 ** (N - 1))

        input_labels = tf.cond(train_phase, lambda: self.labels, 
            lambda: tf.stack([self.labels[:, 1], self.labels[:, 0]], axis=1)) # translate expression

        gen_params = self._load_generator_data(num_iter)

        with tf.name_scope(scope_name) as scope:
        	W_ebd = tf.Variable(initial_value=gen_params[scope_name+'/W_ebd:0'], name='W_ebd', trainable=False)
        	b_ebd = tf.Variable(initial_value=gen_params[scope_name+'/b_ebd:0'], name='b_ebd', trainable=False)
        	h_ebd = tf.matmul(input_labels, W_ebd) + b_ebd

        	h_bnebd_avg = tf.Variable(initial_value=gen_params[scope_name+'/gen_bnebd/moving_mean:0'], name='gen_bnebd/moving_mean', trainable=False)
        	h_bnebd_var = tf.Variable(initial_value=gen_params[scope_name+'/gen_bnebd/moving_variance:0'], name='gen_bnebd/moving_variance', trainable=False)
        	h_bnebd_beta = tf.Variable(initial_value=gen_params[scope_name+'/gen_bnebd/beta:0'], name='gen_bnebd/beta', trainable=False)
        	h_bnebd_gamma = tf.Variable(initial_value=gen_params[scope_name+'/gen_bnebd/gamma:0'], name='gen_bnebd/gamma', trainable=False)
        	h_bnebd = tf.nn.batch_normalization(h_ebd, h_bnebd_avg, h_bnebd_var, h_bnebd_beta, h_bnebd_gamma, variance_epsilon=0.001)
        	h_ebd = activation(h_bnebd, name='h_bnebd')
        	utils.add_activation_summary(h_ebd)

        	h_zebd = tf.multiply(h_ebd, z)

        	W_z = tf.Variable(initial_value=gen_params[scope_name+'/W_z:0'], name='W_z', trainable=False)
        	b_z = tf.Variable(initial_value=gen_params[scope_name+'/b_z:0'], name='b_z', trainable=False)
        	h_z = tf.matmul(h_zebd, W_z) + b_z
        	h_z = tf.reshape(h_z, [-1, image_size, image_size, dims[0]])
        	h_bnz_avg = tf.Variable(initial_value=gen_params[scope_name+'/gen_bnz/moving_mean:0'], name='gen_bnz/moving_mean', trainable=False)
        	h_bnz_var = tf.Variable(initial_value=gen_params[scope_name+'/gen_bnz/moving_variance:0'], name='gen_bnz/moving_variance', trainable=False)
        	h_bnz_beta = tf.Variable(initial_value=gen_params[scope_name+'/gen_bnz/beta:0'], name='gen_bnz/beta', trainable=False)
        	h_bnz_gamma = tf.Variable(initial_value=gen_params[scope_name+'/gen_bnz/gamma:0'], name='gen_bnz/gamma', trainable=False)
        	h_bnz = tf.nn.batch_normalization(h_z, h_bnz_avg, h_bnz_var, h_bnz_beta, h_bnz_gamma, variance_epsilon=0.001)
        	h = activation(h_bnz, name='h_z')
        	utils.add_activation_summary(h)

        	for index in range(N - 2):
        		image_size *= 2
        		wt_var_name = scope_name+"/W_"+str(index)+':0'
        		bi_var_name = scope_name+'/b_'+str(index)+':0'
        		W = tf.Variable(initial_value=gen_params[wt_var_name], name='W_%d'%index, trainable=False)
        		b = tf.Variable(initial_value=gen_params[bi_var_name], name='b_%d'%index, trainable=False)
        		deconv_shape = tf.stack([tf.shape(h)[0], image_size, image_size, dims[index + 1]])
        		h_conv_t = utils.conv2d_transpose_strided(h, W, b, output_shape=deconv_shape)

        		bn_avg_name = scope_name+'/gen_bn'+str(index)+'/moving_mean:0'
        		bn_var_name = scope_name+'/gen_bn'+str(index)+'/moving_variance:0'
        		bn_beta_name = scope_name+'/gen_bn'+str(index)+'/beta:0'
        		bn_gamma_name = scope_name+'/gen_bn'+str(index)+'/gamma:0'
        		bn_avg = tf.Variable(initial_value=gen_params[bn_avg_name], name='gen_bn'+str(index)+'/moving_mean', trainable=False)
        		bn_var = tf.Variable(initial_value=gen_params[bn_var_name], name='gen_bn'+str(index)+'/moving_variance', trainable=False)
        		bn_beta = tf.Variable(initial_value=gen_params[bn_beta_name], name='gen_bn'+str(index)+'/beta', trainable=False)
        		bn_gamma = tf.Variable(initial_value=gen_params[bn_gamma_name], name='gen_bn'+str(index)+'/gamma', trainable=False)
        		h_bn = tf.nn.batch_normalization(h_conv_t, bn_avg, bn_var, bn_beta, bn_gamma, variance_epsilon=0.001)
        		h = activation(h_bn, name='h_%d'%index)
        		utils.add_activation_summary(h)

        	image_size *= 2
        	W_pred = tf.Variable(initial_value=gen_params[scope_name+'/W_pred:0'], name='W_pred', trainable=False)
        	b_pred = tf.Variable(initial_value=gen_params[scope_name+'/b_pred:0'], name='b_pred', trainable=False)
        	deconv_shape = tf.stack([tf.shape(h)[0], image_size, image_size, dims[-1]])
        	h_conv_t = utils.conv2d_transpose_strided(h, W_pred, b_pred, output_shape=deconv_shape)
        	pred_image = tf.nn.tanh(h_conv_t, name='pred_image')
        	utils.add_activation_summary(pred_image)

        return pred_image#, input_labels

    def _encoder_loss(self):
    	#self.loss = tf.reduce_mean(tf.square(tf.subtract(self.gen_images, self.images)))
    	self.loss = tf.reduce_mean(tf.abs(self.gen_images - self.images))
    	tf.summary.scalar("Encoder_loss", self.loss)

    def initialize_network(self, logs_dir, iterations):
        print("Initializing network...")
        self.logs_dir = logs_dir
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.5)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=int(iterations//1000))
        #print(self.logs_dir) 
	self.summary_writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer(), feed_dict = {self.train_phase: True})
        ckpt = tf.train.get_checkpoint_state(self.logs_dir)
        if ckpt:
            ckpt_filename_splits = ckpt.model_checkpoint_path.split('-')
            ckpt_filename_splits[1] = str(int(iterations))
            ckpt_filename = '-'.join(ckpt_filename_splits)
            if ckpt_filename:
                print(ckpt_filename)
                self.saver.restore(self.sess, ckpt_filename)
                print("Model restored...")
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(self.sess, self.coord)

    def create_network(self, generator_dims, encoder_dims, gen_logs_dir, num_iter, optimizer="Adam", learning_rate=2e-4, optimizer_param=0.9):
    	print("Setting up model...")
        self.gen_logs_dir = gen_logs_dir 
		self._setup_placeholder()	
		self.z = self._encoder(encoder_dims, self.train_phase)
    	self.gen_images = self._generator(self.z, generator_dims, self.train_phase, num_iter)

    	tf.summary.image("image_real", self.images, max_outputs=4)
    	tf.summary.image("image_generated", self.gen_images, max_outputs=4)

    	self._encoder_loss()

    	self.train_variables = tf.trainable_variables()
    	for v in self.train_variables:
    		utils.add_to_regularization_and_summary(var=v)

    	optim = self._get_optimizer(optimizer, learning_rate, optimizer_param)
    	self.encoder_train_op = self._train(self.loss, self.train_variables, optim)

    def train_model(self, max_iterations):
    	try:
            print("Training model...")
            start_time = time.time()
            for itr in xrange(1, max_iterations):
                batch_z = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
                feed_dict = {self.train_phase: True, self.class_num: -1}

                self.sess.run(self.encoder_train_op, feed_dict=feed_dict)

                if itr % 200 == 0:
                    stop_time = time.time()
                    duration = (stop_time - start_time) / 200.
                    start_time = time.time()
                    encoder_loss, summary_str = self.sess.run(
                        [self.loss, self.summary_op], feed_dict=feed_dict)
                    print("Time: %g/itr, Step: %d, Encoder loss: %g" % (duration, itr, encoder_loss))
                    self.summary_writer.add_summary(summary_str, itr)

                if itr %1000 == 0:
                    self.saver.save(self.sess, self.logs_dir + "encoder.ckpt", global_step=itr)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        except KeyboardInterrupt:
            print("Ending Training...")
        finally:
            self.coord.request_stop()
            self.coord.join(self.threads)  # Wait for threads to finish.

    def visualize_model(self, iterations):
        print("Sampling images from model...")
        feed_dict = {self.train_phase: False, self.class_num: -1}
        origin_images, gen_images = self.sess.run([self.images, self.gen_images], feed_dict=feed_dict)
        origin_images = utils.unprocess_image(origin_images, 127.5, 127.5).astype(np.uint8)
        gen_images = utils.unprocess_image(gen_images, 127.5, 127.5).astype(np.uint8)
        shape = [2, 2]
        save_img_fn = "encoder_"+str(int(iterations))+'.png'
        utils.save_encoder_img(origin_images, gen_images, self.logs_dir, save_img_fn, shape=shape)

