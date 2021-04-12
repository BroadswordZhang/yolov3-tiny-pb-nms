# -*- coding: utf-8 -*-

import tensorflow as tf

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, load_weights

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'class_names', './sur/sur0316/sur.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', './sur/sur0316/yolov3_tiny_sur_best.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_bool(
    'tiny', True, 'Use tiny version of YOLOv3')
tf.app.flags.DEFINE_bool(
    'spp', False, 'Use SPP version of YOLOv3')
tf.app.flags.DEFINE_string(
    'ckpt_file', './sur/sur0316/yolov3_tiny_sur.ckpt', 'Chceckpoint file')


def main(argv=None):
    if FLAGS.tiny:
        model = yolo_v3_tiny.yolo_v3_tiny
    elif FLAGS.spp:
        model = yolo_v3.yolo_v3_spp
    else:
        model = yolo_v3.yolo_v3

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    # any size > 320 will work here
    inputs = tf.placeholder(tf.float32, [None, 608, 608, 3])

    with tf.variable_scope('detector'):
        detections = model(inputs, len(classes),
                           data_format=FLAGS.data_format)
        load_ops = load_weights(tf.global_variables(
            scope='detector'), FLAGS.weights_file)
        for op in load_ops:
            print(op)

    saver = tf.train.Saver(tf.global_variables(scope='detector'))

    with tf.Session() as sess:
        sess.run(load_ops)

        save_path = saver.save(sess, save_path=FLAGS.ckpt_file)
        print('Model saved in path: {}'.format(save_path))


if __name__ == '__main__':
    tf.app.run()
