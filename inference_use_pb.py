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

classes = ['face', 'mask']

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

def inference(img_dir, out_dir):
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        with tf.gfile.FastGFile('./sur/sur0316/yolov3_tiny_sur.pb', 'rb') as f:
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
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, 608, 608)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            "img = img[np.newaxis, :]"
            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio

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
