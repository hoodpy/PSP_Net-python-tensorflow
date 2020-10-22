import tensorflow as tf
import numpy as np
import os
from model import Timer, PSP_net


class Trainer():
	def __init__(self):
		self._data_path = "D:/program/psp_net/data/data.tfrecords"
		self._log_path = "D:/program/psp_net/log/"
		self._model_path = "D:/program/psp_net/model/"
		self._image_size = [713, 713]
		self._batch_size = 5
		self._num_classes = 21
		self._shuffle_size = 10
		self._epochs = 100
		self._learning_rate = 1e-3
		self.network = PSP_net(num_classes=self._num_classes)
		self.timer = Timer()

	def parse(self, record):
		features = tf.io.parse_single_example(record, features={
			"high": tf.io.FixedLenFeature([], tf.int64), 
			"width": tf.io.FixedLenFeature([], tf.int64),
			"depth": tf.io.FixedLenFeature([], tf.int64),
			"image": tf.io.FixedLenFeature([], tf.string),
			"label": tf.io.FixedLenFeature([], tf.string)
			})
		high, width = tf.cast(features["high"], tf.int32), tf.cast(features["width"], tf.int32)
		depth = tf.cast(features["depth"], tf.int32)
		image, label = tf.decode_raw(features["image"], tf.uint8), tf.decode_raw(features["label"], tf.uint8)
		image, label = tf.reshape(image, [high, width, depth]), tf.reshape(label, [high, width, 1])
		return image, label

	def preprocess_for_train(self, image, label):
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		image_shape = tf.shape(image)
		label = tf.cast(label, tf.int32)
		scale = tf.random.uniform([1], minval=0.5, maxval=2.0)
		new_h = tf.cast(tf.multiply(tf.cast(image_shape[0], tf.float32), scale), tf.int32)
		new_w = tf.cast(tf.multiply(tf.cast(image_shape[1], tf.float32), scale), tf.int32)
		new_shape = tf.squeeze(tf.stack([new_h, new_w]), axis=1)
		image = tf.image.resize_images(image, new_shape, method=0)
		label = tf.image.resize_images(label, new_shape, method=1)
		label = tf.cast(label - 255, tf.float32)
		total = tf.concat([image, label], -1)
		total_pad = tf.image.pad_to_bounding_box(total, 0, 0, tf.math.maximum(new_shape[0], self._image_size[0]), 
			tf.math.maximum(new_shape[1], self._image_size[1]))
		total_crop = tf.random_crop(total_pad, [self._image_size[0], self._image_size[1], 4])
		image, label = total_crop[:, :, :3], total_crop[:, :, 3:]
		label = tf.cast(label + 255., tf.int32)
		image.set_shape([self._image_size[0], self._image_size[1], 3])
		label.set_shape([self._image_size[0], self._image_size[1], 1])
		return image, label

	def get_dataset(self):
		dataset = tf.data.TFRecordDataset(self._data_path)
		dataset = dataset.map(self.parse)
		dataset = dataset.map(lambda image, label: self.preprocess_for_train(image, label))
		dataset = dataset.shuffle(self._shuffle_size).repeat(self._epochs).batch(self._batch_size)
		self.iterator = dataset.make_initializable_iterator()
		image_batch, label_batch = self.iterator.get_next()
		return image_batch, label_batch

	def prepare_label(self, annotations):
		with tf.compat.v1.name_scope("label_encode"):
			labels = tf.compat.v1.image.resize_nearest_neighbor(annotations, tf.stack(self.network._logits.get_shape()[1:3]))
			labels = tf.squeeze(labels, axis=[-1])
		return labels

	def train(self):
		config = tf.compat.v1.ConfigProto()
		config.allow_soft_placement = True
		config.gpu_options.allow_growth = True
		with tf.compat.v1.Session(config=config) as sess:
			global_step = tf.Variable(0, trainable=False)
			learning_base = tf.Variable(self._learning_rate, trainable=False)
			learning_rate = tf.compat.v1.train.exponential_decay(learning_base, global_step, 12031, 0.9, staircase=True)
			tf.compat.v1.summary.scalar("learning_rate", learning_rate)

			image_batch, label_batch = self.get_dataset()
			image_batch = tf.reshape(image_batch, [self._batch_size, self._image_size[0], self._image_size[1], 3])
			label_batch = tf.reshape(label_batch, [self._batch_size, self._image_size[0], self._image_size[1], 1])
			self.network.build_network(image_batch, is_training=True)
			label_batch = self.prepare_label(label_batch)
			cross_entropy = self.network.add_loss(label_batch)
			tf.compat.v1.summary.scalar("cross_entropy", cross_entropy)

			with tf.device("/cpu:0"):
				update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
				with tf.control_dependencies(update_ops):
					train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

			self.saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=5)
			merged = tf.compat.v1.summary.merge_all()
			summary_writer = tf.compat.v1.summary.FileWriter(self._log_path, sess.graph)

			sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
			sess.run(self.iterator.initializer)

			while True:
				try:
					self.timer.tic()
					_cross_entropy, step, summary = self.network.train_step(sess, train_op, global_step, merged)
					summary_writer.add_summary(summary, step)
					self.timer.toc()
					if (step + 1) % 1203 == 0:
						print(">>>Step: %d\n>>>Cross_entropy: %.6f\n>>>Speed: %.6fs\n" % (step + 1, _cross_entropy, 
							self.timer.average_time))
					if (step + 1) % 24062 == 0:
						self.snap_shot(sess, step + 1)
				except tf.errors.OutOfRangeError:
					break

			summary_writer.close()

	def snap_shot(self, sess, step):
		network = self.network
		file_name = os.path.join(self._model_path, "model%d.ckpt" % (step))
		self.saver.save(sess, file_name)
		print("Wrote snapshot to: %s\n" % (file_name))


if __name__ == "__main__":
	trainer = Trainer()
	trainer.train()