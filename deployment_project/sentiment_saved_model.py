import os
print('Curr Dir:',os.getcwd())
import sys

# This is a placeholder for a Google-internal import.

import tensorflow as tf

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS


def main(_):
	if FLAGS.training_iteration <= 0:
		print 'Please specify a positive value for training iteration.'
		sys.exit(-1)
	if FLAGS.model_version <= 0:
		print 'Please specify a positive value for version number.'
		sys.exit(-1)

	# Train model
	print 'Training model...'
  
	# coding: utf-8

	# In[ ]:

	import numpy as np
	import tensorflow as tf
	import re
	from collections import Counter
	import json
	from pprint import pprint
	from tensorflow.contrib import learn
	import re
	import csv
	import pickle
	import time


	# # Load Chat Dataset From Pickle

	# In[ ]:

	pkl_file = open('train_x.pkl', 'rb')
	train_x = pickle.load(pkl_file)

	pkl_file = open('test_x.pkl', 'rb')
	test_x = pickle.load(pkl_file)

	pkl_file = open('val_x.pkl', 'rb')
	val_x = pickle.load(pkl_file)

	pkl_file = open('train_y.pkl', 'rb')
	train_y = pickle.load(pkl_file)

	pkl_file = open('test_y.pkl', 'rb')
	test_y = pickle.load(pkl_file)

	pkl_file = open('val_y.pkl', 'rb')
	val_y = pickle.load(pkl_file)


	# # Util Functions

	# In[ ]:

	#Write To csv File
	def file_writer(file_name,row_):
	    with open(file_name, 'a') as f:
		spamwriter = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(row_ )
		
	#Batching
	def get_batches(x, y, batch_size=100):
	    n_batches = len(x)//batch_size
	    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
	    for ii in range(0, len(x), batch_size):
		yield x[ii:ii+batch_size], y[ii:ii+batch_size]
		



	# # Construct Possible Hyperparameter List

	# In[ ]:

	lstm_size_list = [64] #[16,32,64]
	lstm_layers_list = [1]
	batch_size_list = [128] #[16,32,64,128]
	n_epochs_list = [1] #[10,15,20]

	all_hyperpara_list = []
	for a in lstm_size_list:
	    for b in lstm_layers_list:
		for c in batch_size_list:
		     for d in n_epochs_list:
		        all_hyperpara_list.append((a,b,c,d))


	# # Start Hyperparameter Tuning

	# In[ ]:

	hyperpara_index = 0 #Hyperparameter S.No.

	hyperameter_tuple = all_hyperpara_list[0] 
	#Current set of Hyperparmeters
	lstm_size = hyperameter_tuple[0]
	lstm_layers = hyperameter_tuple[1]
	batch_size = hyperameter_tuple[2]
	n_epochs = hyperameter_tuple[3]
	#file_writer('hyper_'+str(hyperpara_index)+'.csv',['lstm_size= '+str(lstm_size), 'lstm_layers= '+str(lstm_layers),'batch_size= '+str(batch_size),'n_epochs= '+str(n_epochs)])    

	#Reset Graph 
	from tensorflow.python.framework import ops
	ops.reset_default_graph()

	#Placeholder
	X = tf.placeholder(tf.float32, [None, None, 300], name = 'inputs')
	Y = tf.placeholder(tf.float32, [None, 1], name = 'labels')

	#Build Network
	lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
	cell = tf.contrib.rnn.MultiRNNCell([lstm]*lstm_layers)
	
	initial_state = cell.zero_state(tf.shape(X)[0] , tf.float32)
	outputs, final_state = tf.nn.dynamic_rnn(cell, X, initial_state = initial_state)
	predictions = tf.contrib.layers.fully_connected(outputs[:, -1],1, activation_fn=tf.tanh)

	#Optimisation
	loss = tf.reduce_mean(tf.square(Y - predictions))
	optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
	
	#Saving Model Vars	
	values, indices = tf.nn.top_k( tf.reshape(predictions ,[tf.shape(predictions)[0]] ), 10)
	table = tf.contrib.lookup.index_to_string_table_from_tensor(
      	tf.constant([str(i) for i in xrange(10)]))
	prediction_classes = table.lookup(tf.to_int64(indices))
	

	#Accuracy
	correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.float32), Y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	#Start Time
	start_time = time.time()

	#Temp Variables
	x_ = [] #Accounts for batch size
	loss_count = 0
	train_loss = []
	val_loss = []

	#START SESSION:
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	#Serialisation vars:
	serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
	feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
	tf_example = tf.parse_example(serialized_tf_example, feature_configs)
	
	
	#For every Epoch
	for e in range(n_epochs):
		count_=0
		batch_index = 1 #represents index of batch

		#For every batch
		for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
		    if(count_==0):
			state = sess.run(initial_state, feed_dict =  {X: x, Y: np.reshape(y,(len(y),1))})

		    feed = {X: x, Y: np.reshape(y,(len(y),1)), initial_state: state}
		    state, loss_,  _ = sess.run([final_state, loss, optimizer], feed_dict=feed)

		    #represents index of batch
		    batch_index +=1
		   
		    count_+=1
		#All epochs completed
		print('Training Completed')

  
  
	print 'Done training!'
	

	# Export model
	# WARNING(break-tutorial-inline-code): The following code snippet is
	# in-lined in tutorials, please update tutorial documents accordingly
	# whenever code changes.
	export_path_base = sys.argv[-1]
	export_path = os.path.join(tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(str(FLAGS.model_version)))
	import shutil
	try:
		shutil.rmtree(export_path)
	except OSError, e:
		print ("Error: %s - %s." % (e.filename,e.strerror))
	print 'Exporting trained model to', export_path
	builder = tf.saved_model.builder.SavedModelBuilder(export_path)

	# Build the signature_def_map.
	classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
	classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
	classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

	classification_signature = (tf.saved_model.signature_def_utils.build_signature_def(
	  inputs={
	      tf.saved_model.signature_constants.CLASSIFY_INPUTS:
		  classification_inputs
	  },
	  outputs={
	      tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
		  classification_outputs_classes,
	      tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
		  classification_outputs_scores
	  },
	  method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

	tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
	tensor_info_y = tf.saved_model.utils.build_tensor_info(predictions)

	prediction_signature = (
	tf.saved_model.signature_def_utils.build_signature_def(
	  inputs={'sentence': tensor_info_x},
	  outputs={'scores': tensor_info_y},
	  method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

	legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
	builder.add_meta_graph_and_variables(
	sess, [tf.saved_model.tag_constants.SERVING],
	signature_def_map={
	  'predict_sentiment':
	      prediction_signature,
	  tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
	      classification_signature,
	},
	legacy_init_op=legacy_init_op)

	builder.save()

	print 'Done exporting!'


if __name__ == '__main__':
  tf.app.run()
