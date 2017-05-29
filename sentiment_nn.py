import tensorflow as tf

#Hyperparameters
input_dim =  #vocab size
hidden_dim = 100
output_dim = 1 

#Network Architecture
X = tf.placeholder(tf.float32, [None, input_dim])
Y = tf.placeholder(tf.float32, [None, 1])

weights_0_1 = tf.Variable( tf.zeros([input_dim, hidden_dim]) )
weights_1_2 = tf.Variable( tf.zeros([hidden_dim, output_dim]) ) 

layer_1 = tf.matmul(X, weights_0_1)
layer_2 = tf.sigmoid( tf.matmul(layer_1, weights_1_2) )

loss = Y-layer_2
minimise = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#Training 
sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_epoch = 10
batch_size = #
n_batch = /batch_size	#
for epoch in range(n_epoch):
	start=0
	for batch in range(n_batch):
		inp,out = [start:start+batch_size] #	
		_, loss_ = sess.run([minimize, loss],feed_dict={X: , Y: })
		start += batch_size 
	print 'Epoch={}  Loss={} '.format(epoch, loss_)







