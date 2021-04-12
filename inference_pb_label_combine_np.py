# coding: utf-8

from __future__ import division, print_function

import os, sys
import tensorflow as tf
import time
import cv2
import numpy as np

from utils import plot_one_box

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

color_table = [[0, 255, 0], [255, 0, 0], [0, 0, 255]]

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


def inference(img_dir, out_dir):
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        with tf.gfile.FastGFile('./sur/sur0228/yolov3_tiny_sur.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')#

        sess.run(tf.global_variables_initializer())

        input_data = sess.graph.get_tensor_by_name('input_data:0')
        labels = sess.graph.get_tensor_by_name('labels:0')
        scores = sess.graph.get_tensor_by_name('scores:0')
        boxes = sess.graph.get_tensor_by_name('boxes:0')

        img_names = os.listdir(img_dir)
        count = 0
        times = []
        area_thresh = 0.6
        for img_name in img_names:

            img_ori = cv2.imread(os.path.join(img_dir, img_name))
            #img_ori = cv2.imread(img_name)
            start = time.time()
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, 416, 416)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            "img = img[np.newaxis, :]"
            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
            print(20 * "--", boxes_, labels_ )

            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            "----------------------------------------------------------"
            labels_ = np.reshape(labels_, [-1, 1])
            scores_ = np.reshape(scores_, [-1, 1])
            result = np.concatenate([boxes_, scores_, labels_], axis=1)
            #result = tf.concat([boxes_, scores_, labels_], axis=-1)
            print(20 * "--", result)
            "split result to two matrix, glasses and face "
            mask_glasses = np.equal(result[:, 5], 2)
            mask_face = np.not_equal(result[:, 5], 2)
            result_glasses = result[mask_glasses]
            result_face = result[mask_face]
            boxes_glasses, scores_glasses, label_glasses = result_glasses[:, 0:4], result_glasses[:, 4:5], result_glasses[:, 5:6]
            boxes_face, scores_face, label_face = result_face[:, 0:4], result_face[:, 4:5], result_face[:, 5:6]
            #  print(20 * "--", label_face.get_shape().as_list())
            new_label = np.zeros((len(label_face), 1))
            label_face = np.c_[label_face, new_label]
            print(20 * "-*-", label_face)
            if label_glasses.shape[0] != 0:
                x1, y1, x2, y2 = boxes_glasses[:, 0], boxes_glasses[:, 1], boxes_glasses[:, 2], boxes_glasses[:, 3]
                areas = (x2 - x1 + 1) * (y2 - y1 + 1)

                "for each glasses, caculate iou with face or face_mask, and label on face"
                for i in range(scores_glasses.shape[0]):
                    xx1 = np.maximum(x1[i], boxes_face[:, 0])
                    yy1 = np.maximum(y1[i], boxes_face[:, 1])
                    xx2 = np.minimum(x2[i], boxes_face[:, 2])
                    yy2 = np.minimum(y2[i], boxes_face[:, 3])
                    w = np.maximum(0.0, xx2 - xx1 + 1)
                    h = np.maximum(0.0, yy2 - yy1 + 1)
                    inter = w * h
                    ovr = inter / areas[i]
                    inds = np.where(ovr >= area_thresh)
                    print(20 * "--*", inds)
                    label_face[inds, 1] = 1

                    #label_face[inds] = 1

                print(20 * "--*", boxes_face, scores_face, label_face)
                end = time.time()
                print(20 * "--", img_name)

            for i in range(len(boxes_face)):
                x0, y0, x1, y1 = boxes_face[i]
                if label_face[i][1] == 0:
                    print("no glasses")
                else:
                    print("glasses")
         #       plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[label_face[i][0]]+',{:.2f}'.format(scores_face[i]), color=color_table[label_face[i][0]])

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
