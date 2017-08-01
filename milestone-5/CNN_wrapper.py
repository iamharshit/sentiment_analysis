import tensorflow as tf
import numpy as np

class CNN:
	def  __init__(self, seq_len, n_classes, vocab_size):
		self.seq_len = seq_len
		self.n_classes = n_classes
		self.vocab_size = vocab_size
		
		self.epoch_n = 1		
		self.embedding_sz	= 100
		self.filter_sizes = [4,10,15]
		self.batch_size = 4
		self.lr = 0.01
		self.global_iteration = 0

	def create_placeholders(self,):	
		self.x = tf.placeholder(dtype=tf.int32, shape=[None,self.seq_len])
		self.y = tf.placeholder(dtype=tf.float32, shape=[None,1])
				

		self.embedding_table = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_sz],-1.0,1.0) )

	def build_graph(self):
		#Layer1: Embedding Layer
		self.embedded_sentence = tf.nn.embedding_lookup(self.embedding_table, self.x)
		self.embedded_sentence = tf.expand_dims(self.embedded_sentence, -1)

		#Layer2: Convolution
		#Layer3: ReLU
		#Layer4: Maxpool
		pooling_outputs = []
		num_filters = 2#filter of particular shape
		for i,filter_size in enumerate(self.filter_sizes):
			filter_shape = [filter_size, self.embedding_sz,1, num_filters]
			W = tf.Variable(tf.random_uniform(shape=filter_shape))
			B = tf.Variable(tf.random_uniform(shape=[num_filters]))
			
			conv_output = tf.nn.bias_add(tf.nn.conv2d(self.embedded_sentence,W, strides=[1, 1, 1, 1],
padding="VALID"), B)
			
			non_linearity_output = tf.nn.relu(conv_output)
			
			pool_output = tf.nn.max_pool(non_linearity_output, 
								ksize = [1,self.seq_len-filter_size+1,1,1],strides=[1, 1, 1, 1],padding='VALID')
			pooling_outputs.append(pool_output)
		self.pool_outputs = tf.stack(pooling_outputs)
		shp = tf.shape(self.pool_outputs)
		
		#Converting to shape : [batch_size, total_filters]
		self.pool_outputs = tf.reshape(tf.transpose(self.pool_outputs, (1,0,2,3,4)) ,[shp[1],shp[0]*shp[4] ])		

		#Layer5: Fully-Connected Layer	
		total_filters = num_filters*len(self.filter_sizes)
		w = tf.Variable(tf.random_uniform(shape=[total_filters,self.n_classes]))
		b = tf.Variable(tf.random_uniform(shape=[self.n_classes]))
		self.score = tf.nn.softmax( tf.nn.xw_plus_b(self.pool_outputs, w, b) )
		
		#Prediction:
		
		
		#Loss & Optimisation:
		self.loss_train = tf.reduce_mean( tf.square(self.score - self.y) )
		self.loss_valid = tf.reduce_mean( tf.square(self.score - self.y) )		

		tf.summary.scalar("Train_Loss", self.loss_train)
		tf.summary.scalar("Valid_Loss", self.loss_valid)
		self.merged_summary_op = tf.summary.merge_all()
	
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_train)

	def get_batches(self, inp):
			batch_size = self.batch_size
			#inp = np.array(inp)
			x, y = inp 
			x,y = x.tolist(), y.tolist()
			n_batches = len(x)//batch_size
			x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
			for ii in range(0, len(x), batch_size):
				yield x[ii:ii+batch_size], y[ii:ii+batch_size]
	
	def eval_valid_loss(self, sess, valid_set):
			losses = []
			for valid_batch in self.get_batches(valid_set):
				batchX, batchY = valid_batch
				batchY = np.reshape(batchY, [-1,1])
				feed_dict = {self.x:batchX, self.y:batchY}
				summary, score_, loss = sess.run([self.merged_summary_op, self.score, self.loss_valid], feed_dict)
				self.summary_writer.add_summary(summary, self.global_iteration)
				self.global_iteration+=1
		
				losses.append(loss)

			print 'X: ',batchX[0]
			print 'Prediction: ',score_ [0]
			print 'Y: ',batchY[0]
			print 

			return np.mean(losses)

	def train_batch(self, sess, train_set_batch):        
			batchX, batchY = train_set_batch
			batchY = np.reshape(batchY, [-1,1])
			feed_dict = {self.x:batchX, self.y:batchY}

			summary, train_loss ,_ = sess.run([self.merged_summary_op, self.loss_train, self.optimizer], feed_dict)
			self.summary_writer.add_summary(summary, self.global_iteration)
			return train_loss
	
	def train(self, train_set, valid_set):
		self.create_placeholders()
		self.build_graph()       

		sess = tf.Session()
		sess.run(tf.global_variables_initializer() )
		self.summary_writer = tf.summary.FileWriter('/tmp/CNN_sentiment_analysis', graph=tf.get_default_graph())

		print('Start Training...')
		for epoch_i in range(self.epoch_n):
			for iteration,train_set_batch in enumerate(self.get_batches(train_set), 1):
				 train_loss = self.train_batch(sess, train_set_batch)

				 # Print Results:
				 if(iteration%5==0):
				     print("Iteration:  ",iteration)
				     print("Train Loss: ",train_loss)
				     print("Valid Loss: ",self.eval_valid_loss(sess, valid_set))
				     print 
				 self.global_iteration+=1

		print('Training Completed...')

		return sess 			

