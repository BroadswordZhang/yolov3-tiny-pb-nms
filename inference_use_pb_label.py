# coding: utf-8

from __future__ import division, print_function

import os, sys
import tensorflow as tf
import time
import cv2
import numpy as np
import colorsys
import random
from utils import plot_one_box

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

color_table = [[0,255,0], [255,0,0],[0,0,255]]

classes = ['face', 'mask',"glasses"]

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

def save_draw_bbox(image, bboxes_,score_,label_, xml_path, iname, classes=classes, show_label=False):
    """
        bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
        """
    k = 0
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    xml_file = open((xml_path + str(iname[:-4]) + '.xml'), 'w')

    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + str(iname) + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(image_w) + '</width>\n')
    xml_file.write('        <height>' + str(image_h) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    for i, bbox in enumerate(bboxes_):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = score_[i]
        class_ind = int(label_[i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        # if class_ind == 0:

        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
        if class_ind == 0:
                xml_file.write('    <object>\n')
                xml_file.write('        <name>' + "face" + '</name>\n')
                xml_file.write('        <pose>Unspecified</pose>\n')
                xml_file.write('        <truncated>0</truncated>\n')
                xml_file.write('        <difficult>0</difficult>\n')
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(c1[0]) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(c1[1]) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(c2[0]) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(c2[1]) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')
        if class_ind == 1:

                xml_file.write('    <object>\n')
                xml_file.write('        <name>' + "face_mask" + '</name>\n')
                xml_file.write('        <pose>Unspecified</pose>\n')
                xml_file.write('        <truncated>0</truncated>\n')
                xml_file.write('        <difficult>0</difficult>\n')
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(c1[0]) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(c1[1]) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(c2[0]) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(c2[1]) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')
        if class_ind == 2:

                xml_file.write('    <object>\n')
                xml_file.write('        <name>' + "glasses" + '</name>\n')
                xml_file.write('        <pose>Unspecified</pose>\n')
                xml_file.write('        <truncated>0</truncated>\n')
                xml_file.write('        <difficult>0</difficult>\n')
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(c1[0]) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(c1[1]) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(c2[0]) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(c2[1]) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')
    xml_file.write('</annotation>\n')
    k += 1

    return image


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
            #img = img[np.newaxis, :]
            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio

            end = time.time()
            image = save_draw_bbox(img_ori, boxes_,scores_,labels_,  label_dir, img_name)

            #for i in range(len(boxes_)):
            #    x0, y0, x1, y1 = boxes_[i]
            #    plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]]+',{:.2f}'.format(scores_[i]), color=color_table[labels_[i]])

            img_name = os.path.basename(img_name)
            cv2.imwrite(os.path.join(out_dir, img_name), img_ori)
            count += 1
            print('No.{}, img:{}, time:{:.4f}'.format(count, img_name, end-start))

            if count > 1:
                times.append(end-start)

        print('Total:{}, avg time:{:.4f}'.format(count, np.mean(times)))


if __name__ == '__main__':

    img_dir = './data/test'
    out_dir = './data/test_out'
    label_dir = "./data/label/"

    inference(img_dir, out_dir)