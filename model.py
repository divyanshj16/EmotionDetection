import tensorflow as tf

def model1(X,y): 							# simple three layer feedforward network!
	W1 = tf.get_variable('W1',shape=[2304,100])
	b1 = tf.get_variable('b1',shape=[100])
	W2 = tf.get_variable('W2',shape=[100,100])
	b2 = tf.get_variable('b2',shape=[100])
	W3 = tf.get_variable('W3',shape=[100,7])
	b3 = tf.get_variable('b3',shape=[7])

	out1 = tf.nn.relu(tf.matmul(X,W1) + b1)
	out2 = tf.nn.relu(tf.matmul(out1,W2) + b2)
	y_out = tf.matmul(out2,W3) + b3

	return y_out


