import tensorflow as tf

def model1(X,y,seed): 							# simple three layer feedforward network!
	initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
	W1 = tf.get_variable('W1',shape=[2304,100],initializer=initializer)
	b1 = tf.get_variable('b1',shape=[100],initializer=initializer)
	W2 = tf.get_variable('W2',shape=[100,100],initializer=initializer)
	b2 = tf.get_variable('b2',shape=[100],initializer=initializer)
	W3 = tf.get_variable('W3',shape=[100,7],initializer=initializer)
	b3 = tf.get_variable('b3',shape=[7],initializer=initializer)

	out1 = tf.nn.relu(tf.matmul(X,W1) + b1)
	out2 = tf.nn.relu(tf.matmul(out1,W2) + b2)
	y_out = tf.matmul(out2,W3) + b3

	return y_out

def model2(X,y,seed):
	initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
	W1 = tf.get_variable('W1',shape=[2304,5000],initializer=initializer)
	b1 = tf.get_variable('b1',shape=[5000],initializer=initializer)
	W2 = tf.get_variable('W2',shape=[5000,1000],initializer=initializer)
	b2 = tf.get_variable('b2',shape=[1000],initializer=initializer)
	W3 = tf.get_variable('W3',shape=[1000,100],initializer=initializer)
	b3 = tf.get_variable('b3',shape=[100],initializer=initializer)
	W4 = tf.get_variable('W4',shape=[100,100],initializer=initializer)
	b4 = tf.get_variable('b4',shape=[100],initializer=initializer)
	W5 = tf.get_variable('W5',shape=[100,7],initializer=initializer)
	b5 = tf.get_variable('b5',shape=[7],initializer=initializer)

	out1 = tf.nn.relu(tf.matmul(X,W1) + b1)
	out2 = tf.nn.relu(tf.matmul(out1,W2) + b2)
	out3 = tf.nn.relu(tf.matmul(out2,W3) + b3)
	out4 = tf.nn.relu(tf.matmul(out3,W4) + b4)
	y_out = tf.matmul(out4,W5) + b5

	return y_out

def model3(X,y,seed):
	initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
	W1 = tf.get_variable('W1',shape=[2304,5000],initializer=initializer)
	b1 = tf.get_variable('b1',shape=[5000],initializer=initializer)
	W2 = tf.get_variable('W2',shape=[5000,300],initializer=initializer)
	b2 = tf.get_variable('b2',shape=[300],initializer=initializer)
	W3 = tf.get_variable('W3',shape=[300,100],initializer=initializer)
	b3 = tf.get_variable('b3',shape=[100],initializer=initializer)
	W4 = tf.get_variable('W4',shape=[100,7],initializer=initializer)
	b4 = tf.get_variable('b4',shape=[7],initializer=initializer)

	out1 = tf.nn.relu(tf.matmul(X,W1) + b1)
	out2 = tf.nn.relu(tf.matmul(out1,W2) + b2)
	out3 = tf.nn.relu(tf.matmul(out2,W3) + b3)
	y_out = tf.matmul(out3,W4) + b4

	return y_out


def model4(X,y,seed):
	initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
	
	W1 = tf.get_variable('W1',shape=[2304,5000],initializer=initializer)
	b1 = tf.get_variable('b1',shape=[5000],initializer=initializer)
	gamma1 = tf.get_variable('gamma1',shape=[5000],initializer=initializer)
	beta1 = tf.get_variable('beta1',shape=[5000],initializer=initializer)

	W2 = tf.get_variable('W2',shape=[5000,300],initializer=initializer)
	b2 = tf.get_variable('b2',shape=[300],initializer=initializer)
	gamma2 = tf.get_variable('gamma2',shape=[300],initializer=initializer)
	beta2 = tf.get_variable('beta2',shape=[300],initializer=initializer)

	W3 = tf.get_variable('W3',shape=[300,100],initializer=initializer)
	b3 = tf.get_variable('b3',shape=[100],initializer=initializer)
	gamma3 = tf.get_variable('gamma3',shape=[100],initializer=initializer)
	beta3 = tf.get_variable('beta3',shape=[100],initializer=initializer)

	W4 = tf.get_variable('W4',shape=[100,7],initializer=initializer)
	b4 = tf.get_variable('b4',shape=[7],initializer=initializer)

	out1 = tf.nn.relu(tf.matmul(X,W1) + b1)
	mean1,var1 = tf.nn.moments(out1,axes=[0])
	out1_bn = tf.nn.batch_normalization(out1,mean1,var1,beta1,gamma1,1e-7)

	out2 = tf.nn.relu(tf.matmul(out1_bn,W2) + b2)
	mean2,var2 = tf.nn.moments(out2,axes=[0])
	out2_bn = tf.nn.batch_normalization(out2,mean2,var2,beta2,gamma2,1e-7)

	out3 = tf.nn.relu(tf.matmul(out2_bn,W3) + b3)
	mean3,var3 = tf.nn.moments(out3,axes=[0])
	out3_bn = tf.nn.batch_normalization(out3,mean3,var3,beta3,gamma3,1e-7)

	y_out = tf.matmul(out3_bn,W4) + b4

	return y_out

def model5(X,y,seed):
	initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)

	beta0 = tf.get_variable('beta0',shape=[2304],initializer=initializer)
	gamma0 = tf.get_variable('gamma0',shape=[2304],initializer=initializer)

	W1 = tf.get_variable('W1',shape=[2304,5000],initializer=initializer)
	b1 = tf.get_variable('b1',shape=[5000],initializer=initializer)
	gamma1 = tf.get_variable('gamma1',shape=[5000],initializer=initializer)
	beta1 = tf.get_variable('beta1',shape=[5000],initializer=initializer)

	W2 = tf.get_variable('W2',shape=[5000,300],initializer=initializer)
	b2 = tf.get_variable('b2',shape=[300],initializer=initializer)
	gamma2 = tf.get_variable('gamma2',shape=[300],initializer=initializer)
	beta2 = tf.get_variable('beta2',shape=[300],initializer=initializer)

	W3 = tf.get_variable('W3',shape=[300,100],initializer=initializer)
	b3 = tf.get_variable('b3',shape=[100],initializer=initializer)
	gamma3 = tf.get_variable('gamma3',shape=[100],initializer=initializer)
	beta3 = tf.get_variable('beta3',shape=[100],initializer=initializer)

	W4 = tf.get_variable('W4',shape=[100,7],initializer=initializer)
	b4 = tf.get_variable('b4',shape=[7],initializer=initializer)

	mean0,var0 = tf.nn.moments(X,axes=[0])
	X_bn = tf.nn.batch_normalization(X,mean0,var0,beta0,gamma0,1e-7)

	out1 = tf.nn.relu(tf.matmul(X_bn,W1) + b1)
	mean1,var1 = tf.nn.moments(out1,axes=[0])
	out1_bn = tf.nn.batch_normalization(out1,mean1,var1,beta1,gamma1,1e-7)

	out2 = tf.nn.relu(tf.matmul(out1_bn,W2) + b2)
	mean2,var2 = tf.nn.moments(out2,axes=[0])
	out2_bn = tf.nn.batch_normalization(out2,mean2,var2,beta2,gamma2,1e-7)

	out3 = tf.nn.relu(tf.matmul(out2_bn,W3) + b3)
	mean3,var3 = tf.nn.moments(out3,axes=[0])
	out3_bn = tf.nn.batch_normalization(out3,mean3,var3,beta3,gamma3,1e-7)

	y_out = tf.matmul(out3_bn,W4) + b4

	return y_out

def model6(X,y,is_training,seed):
	"""Model function for CNN."""

	initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
	# Input Layer
	input_layer = tf.reshape(X, [-1, 48, 48, 1])

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=8,                 # change number of filters!!
		kernel_size=[3, 3],
		strides=(1,1),
		padding="same",
		activation=tf.nn.relu,
		use_bias=True,
		kernel_initializer=initializer)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=10,               # change number of filters!!
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu,
		kernel_initializer=initializer,
		use_bias=True,
		strides=(1,1))

	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	  # Dense Layer
	pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 10])    # put 12 X 12 X {the value of number of filters of last layer}
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu,use_bias=True,kernel_initializer=initializer)
	dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=is_training,seed=10)

	  # Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=7,use_bias=True,kernel_initializer=initializer)
	return logits

def model7(X,y,seed,is_training):
	initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=tf.float32)
	# Input Layer
	input_layer = tf.reshape(X, [-1, 48, 48, 1])

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,                 # change number of filters!!
		kernel_size=[3, 3],
		strides=(1,1),
		padding="same",
		activation=tf.nn.relu,
		use_bias=True,
		kernel_initializer=initializer)

	batch_norm1 = tf.layers.batch_normalization(conv1,training=is_training) 

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=batch_norm1, pool_size=[2, 2], strides=2)

	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=32,               # change number of filters!!
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu,
		kernel_initializer=initializer,
		use_bias=True,
		strides=(1,1))

	batch_norm2 = tf.layers.batch_normalization(conv2,training=is_training)
    
	conv3 = tf.layers.conv2d(
		inputs=batch_norm2,
		filters=32,               # change number of filters!!
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu,
		kernel_initializer=initializer,
		use_bias=True,
		strides=(1,1))

	batch_norm3 = tf.layers.batch_normalization(conv3,training=is_training)
    
	conv4 = tf.layers.conv2d(
		inputs=batch_norm3,
		filters=32,               # change number of filters!!
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu,
		kernel_initializer=initializer,
		use_bias=True,
		strides=(1,1))

	batch_norm4 = tf.layers.batch_normalization(conv4,training=is_training)
    
	conv5 = tf.layers.conv2d(
		inputs=batch_norm4,
		filters=32,               # change number of filters!!
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu,
		kernel_initializer=initializer,
		use_bias=True,
		strides=(1,1))

	batch_norm5 = tf.layers.batch_normalization(conv5,training=is_training)
   
	conv6 = tf.layers.conv2d(
		inputs=batch_norm5,
		filters=32,               # change number of filters!!
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu,
		kernel_initializer=initializer,
		use_bias=True,
		strides=(1,1))

	batch_norm6 = tf.layers.batch_normalization(conv6,training=is_training)
	pool2 = tf.layers.max_pooling2d(inputs=batch_norm6, pool_size=[2, 2], strides=2)

	# Dense Layer
	pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 32])    # put 12 X 12 X {the value of number of filters of last layer}
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu,use_bias=True,kernel_initializer=initializer)
	batch_norm3 = tf.layers.batch_normalization(dense,training=is_training)
	dropout = tf.layers.dropout(inputs=batch_norm3, rate=0.4, training=is_training)

	  # Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=7,use_bias=True,kernel_initializer=initializer)
	return logits




