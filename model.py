import tensorflow as tf
import time


class Timer():
	def __init__(self):
		self.total_time = 0
		self.start_time = 0
		self.calls = 0
		self.diff = 0
		self.average_time = 0

	def tic(self):
		self.start_time = time.time()

	def toc(self, average=True):
		self.diff = time.time() - self.start_time
		self.total_time += self.diff
		self.calls += 1
		self.average_time = self.total_time / self.calls
		if average:
			return self.average_time
		return self.diff


class PSP_net():
	def __init__(self, num_classes):
		self._num_classes = num_classes

	def conv2d(self, images, num_output, scope, is_training, kernel_size=3, stride=1, padding="SAME", bias=False, bn=True, relu=True):
		with tf.compat.v1.variable_scope(scope):
			num_input = images.get_shape()[-1]
			weights = tf.compat.v1.get_variable("weights", [kernel_size, kernel_size, num_input, num_output], 
				initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=is_training)
			conv = tf.nn.conv2d(images, weights, strides=[1, stride, stride, 1], padding=padding, name="conv2d")
			if bias:
				biases = tf.compat.v1.get_variable("biases", [num_output], initializer=tf.constant_initializer(0.0), trainable=is_training)
				conv = tf.nn.bias_add(conv, biases, name="bias_add")
			if bn:
				conv = tf.layers.batch_normalization(conv, training=is_training, name="bn_op")
			if relu:
				conv = tf.nn.relu(conv, name="relu_op")
		return conv

	def atrous_conv2d(self, images, num_output, atrous, scope, is_training, kernel_size=3, padding="SAME", bn=True, relu=True):
		with tf.compat.v1.variable_scope(scope):
			num_input = images.get_shape()[-1]
			weights = tf.compat.v1.get_variable("weights", [kernel_size, kernel_size, num_input, num_output], 
				initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=is_training)
			conv = tf.nn.atrous_conv2d(images, weights, rate=atrous, padding=padding, name="atrous_conv2d")
			if bn:
				conv = tf.layers.batch_normalization(conv, training=is_training, name="bn_op")
			if relu:
				conv = tf.nn.relu(conv, name="relu_op")
		return conv

	def unit_v1(self, images, num_output, scope, is_training, stride):
		with tf.compat.v1.variable_scope(scope):
			conv0 = self.conv2d(images, num_output//4, scope="conv0_0", is_training=is_training, kernel_size=1, stride=stride)
			conv0 = self.conv2d(conv0, num_output//4, scope="conv0_1", is_training=is_training, kernel_size=3)
			conv0 = self.conv2d(conv0, num_output, scope="conv0_2", is_training=is_training, kernel_size=1, relu=False)
			conv1 = self.conv2d(images, num_output, scope="conv1", is_training=is_training, kernel_size=1, stride=stride, relu=False)
			conv = tf.math.add(conv0, conv1, name="add_op")
			conv = tf.nn.relu(conv0, name="relu_op")
		return conv

	def unit_v1_seq(self, images, num_output, scope, is_training):
		with tf.compat.v1.variable_scope(scope):
			conv0 = self.conv2d(images, num_output//4, scope="conv0_0", is_training=is_training, kernel_size=1)
			conv0 = self.conv2d(conv0, num_output//4, scope="conv0_1", is_training=is_training, kernel_size=3)
			conv0 = self.conv2d(conv0, num_output, scope="conv0_2", is_training=is_training, kernel_size=1, relu=False)
			conv = tf.math.add(conv0, images, name="add_op")
			conv = tf.nn.relu(conv, name="relu_op")
		return conv

	def unit_v2(self, images, num_output, scope, is_training, atrous):
		with tf.compat.v1.variable_scope(scope):
			conv0 = self.conv2d(images, num_output//4, scope="conv0_0", is_training=is_training, kernel_size=1)
			conv0 = self.atrous_conv2d(conv0, num_output//4, atrous=atrous, scope="conv0_1", is_training=is_training)
			conv0 = self.conv2d(conv0, num_output, scope="conv0_2", is_training=is_training, kernel_size=1, relu=False)
			conv1 = self.conv2d(images, num_output, scope="conv1", is_training=is_training, kernel_size=1, relu=False)
			conv = tf.math.add(conv0, conv1, name="add_op")
			conv = tf.nn.relu(conv, name="relu_op")
		return conv

	def unit_v2_seq(self, images, num_output, scope, is_training, atrous):
		with tf.compat.v1.variable_scope(scope):
			conv0 = self.conv2d(images, num_output//4, scope="conv0_0", is_training=is_training, kernel_size=1)
			conv0 = self.atrous_conv2d(conv0, num_output//4, atrous=atrous, scope="conv0_1", is_training=is_training)
			conv0 = self.conv2d(conv0, num_output, scope="conv0_2", is_training=is_training, kernel_size=1, relu=False)
			conv = tf.math.add(conv0, images, name="add_op")
			conv = tf.nn.relu(conv, name="relu_op")
		return conv

	def build_network(self, images, is_training):
		with tf.compat.v1.variable_scope("inference"):
			conv0_0 = self.conv2d(images, num_output=32, scope="unit0_0", is_training=is_training, stride=2)
			conv0_1 = self.conv2d(conv0_0, num_output=32, scope="unit0_1", is_training=is_training)
			conv0_2 = self.conv2d(conv0_1, num_output=64, scope="unit0_2", is_training=is_training)
			conv0_3 = tf.nn.max_pool2d(conv0_2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name="unit0_3")

			conv1_0 = self.unit_v1(conv0_3, num_output=128, scope="unit1_0", is_training=is_training, stride=1)
			conv1_1 = self.unit_v1_seq(conv1_0, num_output=128, scope="unit1_1", is_training=is_training)
			conv1_2 = self.unit_v1_seq(conv1_1, num_output=128, scope="unit1_2", is_training=is_training)

			conv2_0 = self.unit_v1(conv1_2, num_output=256, scope="unit2_0", is_training=is_training, stride=2)
			conv2_1 = self.unit_v1_seq(conv2_0, num_output=256, scope="unit2_1", is_training=is_training)
			conv2_2 = self.unit_v1_seq(conv2_1, num_output=256, scope="unit2_2", is_training=is_training)
			conv2_3 = self.unit_v1_seq(conv2_2, num_output=256, scope="unit2_3", is_training=is_training)

			conv3_0 = self.unit_v2(conv2_3, num_output=512, scope="unit3_0", is_training=is_training, atrous=2)
			conv3_1 = self.unit_v2_seq(conv3_0, num_output=512, scope="unit3_1", is_training=is_training, atrous=2)
			conv3_2 = self.unit_v2_seq(conv3_1, num_output=512, scope="unit3_2", is_training=is_training, atrous=2)
			conv3_3 = self.unit_v2_seq(conv3_2, num_output=512, scope="unit3_3", is_training=is_training, atrous=2)
			conv3_4 = self.unit_v2_seq(conv3_3, num_output=512, scope="unit3_4", is_training=is_training, atrous=2)
			conv3_5 = self.unit_v2_seq(conv3_4, num_output=512, scope="unit3_5", is_training=is_training, atrous=2)

			conv4_0 = self.unit_v2(conv3_5, num_output=1024, scope="unit4_0", is_training=is_training, atrous=4)
			conv4_1 = self.unit_v2_seq(conv4_0, num_output=1024, scope="unit4_1", is_training=is_training, atrous=4)
			conv4_2 = self.unit_v2_seq(conv4_1, num_output=1024, scope="unit4_2", is_training=is_training, atrous=4)

		with tf.compat.v1.variable_scope("pyramid"):
			the_shape = tf.shape(conv4_2)[1:3]

			conv5_0 = tf.nn.avg_pool2d(conv4_2, ksize=[1,90,90,1], strides=[1,90,90,1], padding="VALID", name="unit5_0")
			conv5_1 = self.conv2d(conv5_0, num_output=256, scope="unit5_1", is_training=is_training, kernel_size=1)
			conv5_2 = tf.compat.v1.image.resize_bilinear(conv5_1, the_shape, align_corners=True, name="unit5_2")

			conv6_0 = tf.nn.avg_pool2d(conv4_2, ksize=[1,45,45,1], strides=[1,45,45,1], padding="VALID", name="unit6_0")
			conv6_1 = self.conv2d(conv6_0, num_output=256, scope="unit6_1", is_training=is_training, kernel_size=1)
			conv6_2 = tf.compat.v1.image.resize_bilinear(conv6_1, the_shape, align_corners=True, name="unit6_2")

			conv7_0 = tf.nn.avg_pool2d(conv4_2, ksize=[1,30,30,1], strides=[1,30,30,1], padding="VALID", name="unit7_0")
			conv7_1 = self.conv2d(conv7_0, num_output=256, scope="unit7_1", is_training=is_training, kernel_size=1)
			conv7_2 = tf.compat.v1.image.resize_bilinear(conv7_1, the_shape, align_corners=True, name="unit7_2")

			conv8_0 = tf.nn.avg_pool2d(conv4_2, ksize=[1,15,15,1], strides=[1,15,15,1], padding="VALID", name="unit8_0")
			conv8_1 = self.conv2d(conv8_0, num_output=256, scope="unit8_1", is_training=is_training, kernel_size=1)
			conv8_2 = tf.compat.v1.image.resize_bilinear(conv8_1, the_shape, align_corners=True, name="unit8_2")

			conv9_0 = tf.concat([conv4_2, conv5_2, conv6_2, conv7_2, conv8_2], -1, name="unit9_0")

		with tf.compat.v1.variable_scope("segmentation"):
			conv10_0 = self.conv2d(conv9_0, num_output=256, scope="unit10_0", is_training=is_training)
			logits = self.conv2d(conv10_0, num_output=self._num_classes, scope="logits", is_training=is_training, kernel_size=1, 
				bias=True, bn=False, relu=False)
			logits_softmax = tf.nn.softmax(logits, axis=-1, name="logits_softmax")
			logits_argmax = tf.argmax(logits, dimension=-1, name="logits_argmax")

			self._logits = logits
			self._logits_softmax = logits_softmax
			self._logits_argmax = logits_argmax

	def add_loss(self, annotations):
		logits, labels = tf.reshape(self._logits, [-1, self._num_classes]), tf.reshape(annotations, [-1])
		effective_indics = tf.squeeze(tf.compat.v2.where(tf.math.less_equal(labels, self._num_classes-1)), 1)
		prediction, ground_truth = tf.gather(logits, effective_indics), tf.gather(labels, effective_indics)
		cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=ground_truth), 
			name="cross_entropy")
		self._cross_entropy = cross_entropy
		return cross_entropy

	def train_step(self, sess, train_op, global_step, merged):
		_, _cross_entropy, step, summary = sess.run([train_op, self._cross_entropy, global_step, merged])
		return _cross_entropy, step, summary
