# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os
import time
import random

#from utils.misc_utils import parse_anchors, read_class_names
#from utils.nms_utils import cpu_nms, gpu_nms
#from utils.plot_utils import get_color_table, plot_one_box
#from utils.data_aug import letterbox_resize

#from model import yolov3

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
classes = ["face", "mask"]
parser = argparse.ArgumentParser(description="YOLO-V3 video test procedure.")
parser.add_argument("--input_video", type=str, default="./data/xxx.txt", help="The path of the input video.")
parser.add_argument("--anchor_path", type=str, default="./data/face_mask_anchors.txt", help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[608, 608], help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to use the letterbox resize.")
parser.add_argument("--use_gpu", type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to use gpu nms.")
parser.add_argument("--class_name_path", type=str, default="./data/face_mask.names", help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./checkpoint/model_1.ckpt", help="The path of the weights to restore.")
parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to save the video detection results.")
args = parser.parse_args()

#args.anchors = parse_anchors(args.anchor_path)
#args.classes = read_class_names(args.class_name_path)
num_class = len(classes)

random.seed(2)
color_table = {}
for i in range(num_class):
    color_table[i] = [random.randint(0, 255) for _ in range(3)]

def letterbox_resize(img, new_width, new_height, interp=0):
    '''
    Letterbox resize. keep the original aspect ratio in the resized image.
    '''
    ori_height, ori_width = img.shape[:2]

    resize_ratio = min(new_width / ori_width, new_height / ori_height)

    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

    return image_padded, resize_ratio, dw, dh

vid = cv2.VideoCapture(0)
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
video_fps = int(vid.get(5))

if args.save_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    with tf.gfile.FastGFile('./sur/sur0316/yolov3_tiny_sur.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  #

    sess.run(tf.global_variables_initializer())

    input_data = sess.graph.get_tensor_by_name('input_data:0')
    labels = sess.graph.get_tensor_by_name('labels:0')
    scores = sess.graph.get_tensor_by_name('scores:0')
    boxes = sess.graph.get_tensor_by_name('boxes:0')

    while True:
        ret, img_ori = vid.read()
        if ret == False:
            raise ValueError("No image!")

        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        start_time = time.time()
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})


        end_time = time.time()

        # rescale the coordinates to the original image
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio



        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
        cv2.putText(img_ori, '{:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0, fontScale=1, color=(0, 255, 0), thickness=2)
        #cv2.imshow('image', img_ori)
        if args.save_video:
            videoWriter.write(img_ori)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    if args.save_video:
        videoWriter.release()
