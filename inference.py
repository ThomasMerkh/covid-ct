"""
Thomas Merkh, tmerkh@g.ucla.edu, April 2020

This script can be used for inference using the pre-trained model.

To run this, please include the imagepath input parameter.  For example:

python inference.py --imagepath /home/username/newdata/image1.png
"""
import numpy as np
import tensorflow as tf
import os, argparse
import cv2

parser = argparse.ArgumentParser(description='COVID-Net-CT Inference')
parser.add_argument('--weightspath', default='./PretrainedModel', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
parser.add_argument('--imagepath', default=None, type=str, help='Full path to image to be inferenced')

args = parser.parse_args()

sess = tf.Session()
tf.get_default_graph()
saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
saver.restore(sess, os.path.join(args.weightspath, args.ckptname))
graph = tf.get_default_graph()
mapping = {'Non-COVID': 0, 'COVID': 1}
inv_mapping = {0: 'Non-COVID', 1: 'COVID'}
image_tensor = graph.get_tensor_by_name("input_1:0")
pred_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")
x = cv2.imread(args.imagepath)
x = cv2.resize(x, (224, 224))
x = x.astype('float32') / 255.0
pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})

print('\n\n***Prediction***')
if(pred[0][0] >= pred[0][1]):
    print('Non-COVID')
else:
    print('COVID')
print('\nConfidence')
print('Normal: {:.3f}, COVID: {:.3f}'.format(pred[0][0], pred[0][1]))
print('**DISCLAIMER**')
print('Do not use this prediction for self-diagnosis.')
