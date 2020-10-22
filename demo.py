import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from model import PSP_net


def vis_detection(image, label):
	#0=background 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow
	#11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=monitor
	label_colors = [(0,0,0), (128,0,0), (0,128,0), (128,128,0), (0,0,128), (128,0,128), (0,128,128),
	(128,128,128), (64,0,0), (192,0,0), (64,128,0), (192,128,0), (64,0,128), (192,0,128), (64,128,128),
	(192,128,128), (0,64,0), (128,64,0), (0,192,0), (128,192,0), (0,64,128)]
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6))
	annotation = np.zeros_like(image)
	for i in range(1, 21):
		h_list, w_list = np.where(label==i)
		annotation[h_list, w_list, :] = label_colors[i]
	ax1.imshow(image)
	ax2.imshow(annotation)


image_size = [720, 720]
num_classes = 21
files_path = "D:/program/psp_net/demo"
model_path = "D:/program/psp_net/model/model120310.ckpt"
images_path = [os.path.join(files_path, name) for name in os.listdir(files_path)]
network = PSP_net(num_classes=num_classes)
image_input = tf.placeholder(tf.uint8, shape=[None, None, 3])
image_shape = tf.shape(image_input)
h, w = tf.math.maximum(image_shape[0], image_size[0]), tf.math.maximum(image_shape[1], image_size[1])
image_prepare = tf.image.convert_image_dtype(image_input, dtype=tf.float32)
image_prepare = tf.image.pad_to_bounding_box(image_prepare, 0, 0, h, w)
image_prepare = tf.expand_dims(image_prepare, axis=0)
network.build_network(image_prepare, is_training=False)
logits = tf.image.resize_bilinear(network._logits, size=[h, w], align_corners=True)
logits = tf.image.crop_to_bounding_box(logits, 0, 0, image_shape[0], image_shape[1])
label = tf.argmax(logits, dimension=-1)
saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())


if __name__ == "__main__":
	config = tf.compat.v1.ConfigProto()
	config.allow_soft_placement = True
	config.gpu_options.allow_growth = True
	sess = tf.compat.v1.Session(config=config)
	sess.run(tf.compat.v1.global_variables_initializer())
	saver.restore(sess, model_path)
	for image_path in images_path:
		image = cv2.imread(image_path)[:, :, (2,1,0)]
		result = sess.run(label, feed_dict={image_input: image})
		vis_detection(image, result[0])
	plt.show()