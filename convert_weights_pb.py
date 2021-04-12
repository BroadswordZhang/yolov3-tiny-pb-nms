# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import yolo_v3
import yolo_v3_tiny
# from PIL import Image, ImageDraw
#import yolov3_tiny_tflite
import yolo_v3_tiny
from utils import load_weights, load_coco_names, detections_boxes, freeze_graph,detections_boxes

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'class_names', "./sur/sur.names", 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', './sur/yolov3_tiny_sur_last.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NHWC', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'output_graph', 'yolov3_tiny_sur.pb', 'Frozen tensorflow protobuf model output path')

tf.app.flags.DEFINE_bool(
    'tiny', True, 'Use tiny version of YOLOv3')
tf.app.flags.DEFINE_bool(
    'spp', False, 'Use SPP version of YOLOv3')
tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')



def main(argv=None):
    if FLAGS.tiny:
        model = yolo_v3_tiny.yolo_v3_tiny
        # model = yolov3_tiny_tflite.yolo_v3_tiny
    elif FLAGS.spp:
        model = yolo_v3.yolo_v3_spp
    else:
        model = yolo_v3.yolo_v3

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [1, FLAGS.size, FLAGS.size, 3], "inputs")

    with tf.variable_scope('detector'):
        # detect_1,detect_2 = model(inputs, len(classes), data_format=FLAGS.data_format)
        detection = model(inputs, len(classes), data_format=FLAGS.data_format)
        load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

    # Sets the output nodes in the current session
    # detect_1,detect_2 = detections_boxes(detect_1,detect_2)
    detection = detections_boxes(detection)

    with tf.Session() as sess:
        sess.run(load_ops)
        freeze_graph(sess, FLAGS.output_graph)

if __name__ == '__main__':
    tf.app.run()
