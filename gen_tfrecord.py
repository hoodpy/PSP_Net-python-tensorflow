import tensorflow as tf
import numpy as np
import cv2
import os


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

images_path = "E:/VOCdevkit/VOC2012/JPEGImages"
labels_path = "E:/VOCdevkit/VOC2012/SegmentationClassAug"
files_name = [name.rstrip(name.split(".")[-1]) for name in os.listdir(labels_path)]
writer = tf.io.TFRecordWriter("D:/program/psp_net/data/data.tfrecords")

for name in files_name:
	image = cv2.imread(os.path.join(images_path, name + "jpg"))[:, :, (2,1,0)]
	label = cv2.imread(os.path.join(labels_path, name + "png"), 0)
	high, width, depth = np.array(np.shape(image)).astype(np.int64)
	image, label = image.tostring(), label.tostring()
	example = tf.train.Example(features=tf.train.Features(feature={
		"high": _int64_feature(value=high),
		"width": _int64_feature(value=width),
		"depth": _int64_feature(value=depth),
		"image": _bytes_feature(value=image),
		"label": _bytes_feature(value=label)
		}))
	writer.write(example.SerializeToString())

writer.close()