# coding: utf-8

from __future__ import division, print_function

import os, sys
import tensorflow as tf
import time
import cv2
import numpy as np

from utils import plot_one_box

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

color_table = [[0,255,0], [255,0,0],[0,0,255]]

classes = ['face', 'mask', 'glasses']

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

def get_img_list(img_path, exts=['jpg', 'png', 'jpeg', 'JPG']):
        
    img_list = os.listdir(img_path)
    new_list = []
    for img_name in img_list:
        for ext in exts:
            if img_name.endswith(ext):
                new_list.append(img_name)
                break
    return new_list
def iou_label(boxes_glasses,scores_glasses, label_glasses, boxes_face, scores_face, label_face, area_thresh):
    "caculate every glasses areas"

    x1 = boxes_glasses[:, 0]
    y1 = boxes_glasses[:, 1]
    x2 = boxes_glasses[:, 2]
    y2 = boxes_glasses[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    print(20 * "--**", tf.Session().run(areas))
    "for each glasses, caculate nms with face or face_mask, and label on face"
    for i in range(len(scores_glasses.shape)):
        xx1 = tf.maximum(x1[i], boxes_face[:, 0])
        yy1 = tf.maximum(y1[i], boxes_face[:, 1])
        xx2 = tf.minimum(x2[i], boxes_face[:, 2])
        yy2 = tf.minimum(y2[i], boxes_face[:, 3])
        w = tf.maximum(0.0, xx2 - xx1 + 1)
        h = tf.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        print(20 * "--", tf.Session().run(inter))
        ovr = inter / areas[i]
        print(20 * "-*-", tf.Session().run(ovr))
        inds = tf.where(ovr >= area_thresh)
        print(20 * "--", inds, tf.Session().run(inds))


        label_face[inds[0][0]][1] = 1

    return  boxes_face, scores_face, label_face

def label_combine(boxes_, scores_, labels_, area_thresh):
    "concate three matrix , boxes, scores, labels on axis 1 "
    print(20 * "-----")
    # labels_ = tf.expand_dims(labels_, 0)
    # labels_ = tf.transpose(tf.cast(labels_, tf.float32))
    labels_ = tf.reshape(tf.cast(labels_, tf.float32), [-1,1])
    scores_ = tf.reshape(scores_, [-1, 1])
    result = tf.concat([boxes_, scores_, labels_], axis=-1)
    print(20 * "--", tf.Session().run(result))

    "split result to two matrix, glasses and face "
    mask_glasses = tf.equal(result[:, 5], tf.constant(2, dtype=tf.float32))
    mask_face = tf.not_equal(result[:, 5], tf.constant(2, dtype=tf.float32))
    result_glasses = tf.boolean_mask(result, mask_glasses)
    result_face = tf.boolean_mask(result, mask_face)
    boxes_glasses, scores_glasses, label_glasses = tf.split(result_glasses, [4, 1, 1], 1)
    boxes_face, scores_face, label_face = tf.split(result_face, [4, 1, 1], 1)
    print(20 * "--", tf.Session().run(label_face))
    #  print(20 * "--", label_face.get_shape().as_list())

   # new_label = tf.placeholder(tf.float32, shape=[1, 1])
    #new_label = tf.constant([[0.0]])
    label_face = tf.concat([label_face, label_face], axis=-1)
    print(20 * "--", tf.Session().run(label_face))

    print(20 * "--", tf.Session().run(boxes_face))
    print(20 * "--", tf.Session().run(scores_face))
    not_empty = tf.equal(tf.size(label_glasses), 0)
    print(20 * "--*", tf.Session().run(not_empty))

    boxes_face, scores_face, label_face = tf.cond(tf.equal(tf.size(label_glasses), 0),
                                                (boxes_face, scores_face, label_face),
                                                  iou_label(boxes_glasses, scores_glasses, label_glasses, boxes_face, scores_face, label_face, area_thresh))

    print(20 * "--", tf.Session().run(boxes_face, scores_face, label_face))
    return label_face, boxes_face, scores_face

def inference(img_dir, out_dir):
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        with tf.gfile.FastGFile('./sur/sur1030/yolov3_tiny_sur.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='') #

        sess.run(tf.global_variables_initializer())

        input_data = sess.graph.get_tensor_by_name('input_data:0')
        labels = sess.graph.get_tensor_by_name('labels:0')
        scores = sess.graph.get_tensor_by_name('scores:0')
        boxes = sess.graph.get_tensor_by_name('boxes:0')

        img_names = os.listdir(img_dir)
        count = 0
        times = []
        for img_name in img_names:
            img_ori = cv2.imread(os.path.join(img_dir, img_name))
            #img_ori = cv2.imread(img_name)
            start = time.time()
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, 416, 416)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            "img = img[np.newaxis, :]"
            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            print(boxes_, scores_, labels_)
            boxes, scores, labels = label_combine(boxes_, scores_, labels_, area_thresh=0.6)

            end = time.time()

            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]]+',{:.2f}'.format(scores_[i]), color=color_table[labels_[i]])

            img_name = os.path.basename(img_name)
            cv2.imwrite(os.path.join(out_dir, img_name), img_ori)
            count += 1
            print('No.{}, img:{}, time:{:.4f}'.format(count, img_name, end-start))

            if count > 1:
                times.append(end-start)

        print('Total:{}, avg time:{:.4f}'.format(count, np.mean(times)))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    img_dir = './data/test'
    out_dir = './data/test_out'

    inference(img_dir, out_dir)
