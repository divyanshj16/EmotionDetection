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

def model2(X,y):
	W1 = tf.get_variable('W1',shape=[2304,5000])
	b1 = tf.get_variable('b1',shape=[5000])
	W2 = tf.get_variable('W2',shape=[5000,1000])
	b2 = tf.get_variable('b2',shape=[1000])
	W3 = tf.get_variable('W3',shape=[1000,100])
	b3 = tf.get_variable('b3',shape=[100])
	W4 = tf.get_variable('W4',shape=[100,100])
	b4 = tf.get_variable('b4',shape=[100])
	W5 = tf.get_variable('W5',shape=[100,7])
	b5 = tf.get_variable('b5',shape=[7])

	out1 = tf.nn.relu(tf.matmul(X,W1) + b1)
	out2 = tf.nn.relu(tf.matmul(out1,W2) + b2)
	out3 = tf.nn.relu(tf.matmul(out2,W3) + b3)
	out4 = tf.nn.relu(tf.matmul(out3,W4) + b4)
	y_out = tf.matmul(out4,W5) + b5

	return y_out

def model3(X,y):
	W1 = tf.get_variable('W1',shape=[2304,5000])
	b1 = tf.get_variable('b1',shape=[5000])
	W2 = tf.get_variable('W2',shape=[5000,300])
	b2 = tf.get_variable('b2',shape=[300])
	W3 = tf.get_variable('W3',shape=[300,100])
	b3 = tf.get_variable('b3',shape=[100])
	W4 = tf.get_variable('W4',shape=[100,7])
	b4 = tf.get_variable('b4',shape=[7])

	out1 = tf.nn.relu(tf.matmul(X,W1) + b1)
	out2 = tf.nn.relu(tf.matmul(out1,W2) + b2)
	out3 = tf.nn.relu(tf.matmul(out2,W3) + b3)
	y_out = tf.matmul(out3,W4) + b4

	return y_out


def model4(X,y):
	W1 = tf.get_variable('W1',shape=[2304,5000])
	b1 = tf.get_variable('b1',shape=[5000])
	gamma1 = tf.get_variable('gamma1',shape=[5000])
	beta1 = tf.get_variable('beta1',shape=[5000])

	W2 = tf.get_variable('W2',shape=[5000,300])
	b2 = tf.get_variable('b2',shape=[300])
	gamma2 = tf.get_variable('gamma2',shape=[300])
	beta2 = tf.get_variable('beta2',shape=[300])

	W3 = tf.get_variable('W3',shape=[300,100])
	b3 = tf.get_variable('b3',shape=[100])
	gamma3 = tf.get_variable('gamma3',shape=[100])
	beta3 = tf.get_variable('beta3',shape=[100])

	W4 = tf.get_variable('W4',shape=[100,7])
	b4 = tf.get_variable('b4',shape=[7])

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

def model5(X,y):

	beta0 = tf.get_variable('beta0',shape=[2304])
	gamma0 = tf.get_variable('gamma0',shape=[2304])

	W1 = tf.get_variable('W1',shape=[2304,5000])
	b1 = tf.get_variable('b1',shape=[5000])
	gamma1 = tf.get_variable('gamma1',shape=[5000])
	beta1 = tf.get_variable('beta1',shape=[5000])

	W2 = tf.get_variable('W2',shape=[5000,300])
	b2 = tf.get_variable('b2',shape=[300])
	gamma2 = tf.get_variable('gamma2',shape=[300])
	beta2 = tf.get_variable('beta2',shape=[300])

	W3 = tf.get_variable('W3',shape=[300,100])
	b3 = tf.get_variable('b3',shape=[100])
	gamma3 = tf.get_variable('gamma3',shape=[100])
	beta3 = tf.get_variable('beta3',shape=[100])

	W4 = tf.get_variable('W4',shape=[100,7])
	b4 = tf.get_variable('b4',shape=[7])

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



