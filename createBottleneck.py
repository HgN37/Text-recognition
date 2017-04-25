from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import struct

import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
BOTTLENECK_TENSOR_SIZE = 2048


def import_inception(inception_graph_path):
    with tf.Session() as sess:
        with gfile.FastGFile(inception_graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, img_tensor, resized_img_tensor = (
                tf.import_graph_def(graph_def,
                                    return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                                                     RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, img_tensor, resized_img_tensor


def create_img_bottleneck(sess, img, img_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {img_tensor: img})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def write_floats_to_file(floats_list, path):
    s = struct.pack('d' * BOTTLENECK_TENSOR_SIZE, *floats_list)
    with open(path, 'wb') as f:
        f.write(s)


def read_floats_from_file(path):
    with open(path, 'rb') as f:
        s = struct.unpack('d' * BOTTLENECK_TENSOR_SIZE, f.read())
    return list(s)


def create_bottleneck_file(sess, img_path, bottleneck_path, img_tensor, bottleneck_tensor):
    if not gfile.Exists(img_path):
        print('Khong tim thay file anh: %s' % img_path)
    img_data = gfile.FastGFile(img_path, 'rb').read()
    bottleneck_values = create_img_bottleneck(
        sess, img_data, img_tensor, bottleneck_tensor)
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def create_bottleneck_folder(img_dir, bottleneck_dir, inception_graph_path):
    print('__Bat dau kiem tra/tao bottleneck')
    sess = tf.Session()
    graph, bottleneck_tensor, img_tensor, _ = import_inception(
        inception_graph_path)
    sub_dirs = [x[0] for x in gfile.Walk(img_dir)]
    is_root_dir = True
    i = 0
    for sub_dir in sub_dirs:
        if is_root_dir is True:
            is_root_dir = False
            continue
        dir_name = os.path.basename(sub_dir)
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        print('Dang tim hinh trong thu muc: ', dir_name)
        if not os.path.exists(os.path.join(bottleneck_dir, dir_name)):
            os.makedirs(os.path.join(bottleneck_dir, dir_name))
        # Kiem tra dinh dang file
        for extension in extensions:
            glob = os.path.join(img_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(glob))
        for file_name in file_list:
            bottleneck_path = os.path.join(
                bottleneck_dir, dir_name, os.path.basename(file_name) + '.txt')
            img_path = file_name
            i = i + 1
            if not os.path.exists(bottleneck_path):
                create_bottleneck_file(
                    sess, img_path, bottleneck_path, img_tensor, bottleneck_tensor)
                print('Da tao %i: %s' % (i, bottleneck_path))
            else:
                print('___Tim thay: %s' % bottleneck_path)
    print('__Hoan tat kiem tra/tao %i bottleneck__' % i)
