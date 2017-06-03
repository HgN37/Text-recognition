from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import random

import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
BOTTLENECK_TENSOR_SIZE = 2048


class textClassfier():
    def __init__(self, bottleneck_folder, train_per=0.8, val_per=0.1):
        self.sess = tf.Session()
        self.dataset = self.data_init(bottleneck_folder, train_per, val_per)
        class_count = len(self.dataset.keys())
        self.Input_ts = tf.placeholder(
            tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='Input_ts')
        self.Truth_ts = tf.placeholder(
            tf.float32, [None, class_count], name='Truth_ts')
        W = tf.Variable(tf.truncated_normal(
            [BOTTLENECK_TENSOR_SIZE, class_count]), name='weights')
        b = tf.Variable(tf.zeros([class_count]), name='bias')
        self.logits = tf.matmul(self.Input_ts, W) + b
        self.Output_ts = tf.nn.softmax(self.logits, name='final_result')
        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Truth_ts, 1)), tf.float32))
        self.sess.run(tf.global_variables_initializer())

    def data_init(self, bottleneck_dir, train_per, val_per):
        dataset = {}
        sub_dirs = [x[0] for x in gfile.Walk(bottleneck_dir)]
        is_root_dir = True
        for sub_dir in sub_dirs:
            if is_root_dir is True:
                is_root_dir = False
                continue
            class_name = os.path.basename(sub_dir)
            file_list = []
            train_data = []
            val_data = []
            test_data = []
            file_glob = os.path.join(bottleneck_dir, class_name, '*.txt')
            file_list.extend(gfile.Glob(file_glob))
            train_thres = train_per * len(file_list)
            val_thres = (train_per + val_per) * len(file_list)
            count = 0
            for file_name in file_list:
                base_name = os.path.basename(file_name)
                count += 1
                if (count < train_thres):
                    train_data.append(base_name)
                elif (train_thres <= count) & (count < val_thres):
                    val_data.append(base_name)
                else:
                    test_data.append(base_name)
            dataset[class_name] = {
                'dir': sub_dir,
                'training': train_data,
                'validation': val_data,
                'testing': test_data,
            }
        return dataset

    def get_random_data(self, how_many, category):
        class_count = len(self.dataset.keys())
        bottlenecks = []
        truths = []
        if how_many >= 0:
            for i in range(how_many):
                label_index = random.randrange(class_count)
                label_name = list(self.dataset.keys())[label_index]
                bottleneck_max_index = len(self.dataset[label_name][category])
                bottleneck_index = random.randrange(bottleneck_max_index)
                bottleneck_name = self.dataset[label_name][category][bottleneck_index]
                bottleneck_path = os.path.join(
                    self.dataset[label_name]['dir'], bottleneck_name)
                bottleneck_value = bottleneck_read(bottleneck_path)
                truth = np.zeros(class_count, dtype=np.float32)
                truth[label_index] = 1.0
                bottlenecks.append(bottleneck_value)
                truths.append(truth)
        return bottlenecks, truths

    def start_training(self, step, sample_per_step=50, rate=0.03):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Truth_ts)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        self.training_step = tf.train.GradientDescentOptimizer(
            rate).minimize(cross_entropy_mean)
        for i in range(step):
            X_batch, y_batch = self.get_random_data(sample_per_step, 'training')
            self.sess.run(self.training_step, feed_dict={
                          self.Input_ts: X_batch, self.Truth_ts: y_batch})
            if i % 100 == 0:
                X_batch, y_batch = self.get_random_data(100, 'validation')
                acc = self.sess.run(self.accuracy, feed_dict={
                    self.Input_ts: X_batch, self.Truth_ts: y_batch})
                print('Step %i: ' % i, acc)
                print('-------------')
        print('__Hoan tat training__')

    def testAccuracy(self):
        X_batch, y_batch = self.get_random_data(500, 'testing')
        acc = self.sess.run(self.accuracy, feed_dict={
            self.Input_ts: X_batch, self.Truth_ts: y_batch})
        print('Do chinh xac: ', acc)

    def save_graph(self):
        output_graph_def = graph_util.convert_variables_to_constants(
            self.sess, self.sess.graph.as_graph_def(), ['final_result'])
        with gfile.FastGFile('./my_graph.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('Export graph thanh cong')
        with gfile.FastGFile('./my_label.txt', 'w') as f:
            f.write('\n'.join(self.dataset.keys()) + '\n')
        print('Export label thanh cong')


def bottleneck_read(bottleneck_path):
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values
