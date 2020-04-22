"""
Thomas Merkh, tmerkh@g.ucla.edu, April 2020

This script evaluates the performance of a trained model on the training, validation, and test sets.

"""
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import os, argparse
import cv2
from sklearn.metrics import roc_auc_score
mapping = {'0': 0, '1': 1}

def eval(sess, graph, testfile, testfolder):
    image_tensor = graph.get_tensor_by_name("input_1:0")
    pred_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")

    y_test = []
    pred = []
    for i in range(len(testfile)):
        line = testfile[i].split()
        x = cv2.imread(os.path.join('./Data/', testfolder, line[0]))
        x = cv2.resize(x, (224, 224))
        x = x.astype('float32') / 255.0
        y_test.append(mapping[line[1]])
        pred.append(np.array(sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})).argmax(axis=1))
    y_test = np.array(y_test)
    pred = np.array(pred)

    # y-axis == ground truth, x-axis == prediction
    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    print("The confusion matrix:")
    print(matrix) 
    
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print("The Accuracy:")
    print('Non-COVID: {0:.3f}, COVID: {1:.3f}'.format(class_acc[0],class_acc[1]))

    # precision = true_pos/(true_pos + false_pos)
    # recall = true_pos/(true_pos + false_neg)
    # F-1 Score: The harmonic mean of precision and recall
    precision = matrix[0,0]/(matrix[0,0] + matrix[0,1])
    recall = matrix[0,0]/(matrix[1,0] + matrix[0,0])
    f1score = 2.0*(precision*recall)/(precision+recall)

    print("The F1-Score:")
    print(f1score)

    print("The AUC Score:")
    aucc = roc_auc_score(y_test, pred[:,0])
    print(aucc)


parser = argparse.ArgumentParser(description='COVID-Net Test Performance')
parser.add_argument('--weightspath', default='./PretrainedModel', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpts')
parser.add_argument('--testfile', default='test_CTx.txt', type=str, help='Name of test file')
parser.add_argument('--testfolder', default='./Data/test', type=str, help='Folder where test data is located')
parser.add_argument('--trainfile', default='train_CTx.txt', type=str, help='Name of train file')
parser.add_argument('--trainfolder', default='./Data/train', type=str, help='Folder where train data is located')
parser.add_argument('--valfile', default='val_CTx.txt', type=str, help='Name of validation file')
parser.add_argument('--valfolder', default='./Data/val', type=str, help='Folder where validation data is located')

args = parser.parse_args()

sess = tf.Session()
tf.get_default_graph()
saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
saver.restore(sess, os.path.join(args.weightspath, args.ckptname))
graph = tf.get_default_graph()


print('Quantifying Performance on Training Set:')
with open(args.trainfile) as f:
    trainfiles = f.readlines()
eval(sess, graph, trainfiles, 'train')

print('Quantifying Performance on Validation Set:')
with open(args.valfile) as f:
    valfiles = f.readlines()
eval(sess, graph, valfiles, 'val')

print('Quantifying Performance on Test Set:')
with open(args.testfile) as f:
    testfiles = f.readlines()

eval(sess, graph, testfiles, 'test')