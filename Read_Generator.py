__author__ = 'zhongyu kuang'
import tensorflow as tf
import numpy as np
import os, sys, inspect
from six.moves import cPickle as pickle

def extract_generator(save_dir, ckpt_fn):
	gen_vars = {}
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(ckpt_fn + ".meta")
		saver.restore(sess, ckpt_fn)
		all_vars = tf.global_variables()
		for var in all_vars:
			if var.name.startswith("generator"):
				if var.name.endswith('ExponentialMovingAverage:0'):
					name_list = var.name.split('/')
					if name_list[-2]=="mean":
						new_name = name_list[0]+'/'+name_list[1]+'/moving_mean:0'
					elif name_list[-2]=='variance':
						new_name = name_list[0]+'/'+name_list[1]+'/moving_variance:0'
					else:
						raise ValueError("Check generator namespaces!")
					gen_vars[new_name] = var.eval()
				else:
					gen_vars[var.name] = var.eval()
	ckpt_lst = ckpt_fn.split('-')
	gen_fn = save_dir + 'generator-'+ckpt_lst[-1]
	print("Pickling...")
	with open(gen_fn, 'wb') as f:
		pickle.dump(gen_vars, f, pickle.HIGHEST_PROTOCOL)
	print("Done pickling...")
	return gen_vars

def load_generator(logs_dir, num_iter, ckpt_bname='model.ckpt-'):
	dir_path = os.getcwd()
	pickle_filepath = dir_path + logs_dir + "generator-" + str(num_iter)
	if not os.path.exists(pickle_filepath):
		print("Extract generator from model ckpt...")
		ckpt_fn = dir_path + logs_dir + ckpt_bname + str(num_iter)
		if not os.path.exists(ckpt_fn+'.meta'):
			raise ValueError("Model ckpt not found")
		gen_params = extract_generator(dir_path+logs_dir, ckpt_fn)
	else:
		print("Found pickle file")
		with open(pickle_filepath, 'rb') as f:
			gen_params = pickle.load(f)
	return gen_params

if __name__ == '__main__':
	logs_dir = '/logs/acgan_l2/'
	num_iter = 40000
	g = load_generator(logs_dir, num_iter)
