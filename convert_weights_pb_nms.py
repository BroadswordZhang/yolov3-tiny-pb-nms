# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import yolo_v3
import yolo_v3_tiny
# from PIL import Image, ImageDraw
#import yolov3_tiny_tflite
import os
import yolo_v3_tiny
from utils import load_weights, load_coco_names, detections_boxes, detections_boxes, gpu_nms
from tensorflow.python.tools import freeze_graph

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'class_names', "./sur/sur0316/sur.names", 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', './sur/sur0316/yolov3_tiny_sur_best.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'ckpt_file', './sur/sur0316/yolov3_tiny_sur.ckpt', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'output_graph', './sur/sur0316/yolov3_tiny_sur.pb', 'Frozen tensorflow protobuf model output path')
tf.app.flags.DEFINE_string(
    'output_dir', './sur/sur0316/', 'Frozen tensorflow protobuf model output path')
tf.app.flags.DEFINE_string(
    'output_file', 'yolov3_tiny_sur.pb', 'Frozen tensorflow protobuf model output path')

tf.app.flags.DEFINE_bool(
    'tiny', True, 'Use tiny version of YOLOv3')
tf.app.flags.DEFINE_bool(
    'spp', False, 'Use SPP version of YOLOv3')
tf.app.flags.DEFINE_integer(
    'size', 608, 'Image size')

def build_detection_graph():
    input_data = tf.placeholder(dtype=tf.uint8, shape=[FLAGS.size, FLAGS.size, 3], name='input_data')
    input_data = tf.expand_dims(input_data, 0)
    input_data = tf.cast(input_data, tf.float32)
    input_data = input_data / 255.
    if FLAGS.tiny:
        model = yolo_v3_tiny.yolo_v3_tiny
    elif FLAGS.spp:
        model = yolo_v3.yolo_v3_spp
    else:
        model = yolo_v3.yolo_v3
    classes = load_coco_names(FLAGS.class_names)

   # yolo_model = model(cfgs.class_num, cfgs.anchors)
    with tf.variable_scope('detector'):
        detections = model(input_data, len(classes), data_format=FLAGS.data_format)
        print(detections.get_shape().as_list())
    boxes, pred_confs, pred_probs = tf.split(detections, [4, 1, len(classes)], axis=-1)
    center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2
    pred_boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

    pred_scores = pred_confs * pred_probs
    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, len(classes), max_boxes=20, score_thresh=0.3,
                                    nms_thresh=0.4)

    boxes = tf.identity(boxes, name='boxes')
    scores = tf.identity(scores, name='scores')
    labels = tf.identity(labels, name='labels')

    return boxes, scores, labels


def export_frozenPB():

    tf.reset_default_graph()
    dets = build_detection_graph()

    saver = tf.train.Saver()
    output_node_names = 'boxes,scores,labels'

    with tf.Session() as sess:
        print("we have restred the weights from =====>>\n", FLAGS.ckpt_file)
        saver.restore(sess, FLAGS.ckpt_file)

        tf.train.write_graph(sess.graph_def, FLAGS.output_dir, FLAGS.output_file)
        freeze_graph.freeze_graph(input_graph=FLAGS.output_graph,
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint=FLAGS.ckpt_file,
                                  output_node_names=output_node_names,
                                  restore_op_name="save/restore_all",
                                  filename_tensor_name='save/Const:0',
                                  output_graph=FLAGS.output_graph,
                                  clear_devices=False,
                                  initializer_nodes='')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    export_frozenPB()
