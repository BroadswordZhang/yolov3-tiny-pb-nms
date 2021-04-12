# -*- coding:utf-8 -*-
# -*- author:ZuoJianHao -*-
import tensorflow as tf

path = "yolov3_tiny_widerface_boxes.pb"

inputs = ["inputs"]
# outputs = ['detect_1','detect_2']
outputs = ["output_boxes"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(path, inputs, outputs)
# converter.allow_custom_ops=True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
converter.target_spec.supported_types = [tf.lite.constants.QUANTIZED_UINT8]

tflite_model = converter.convert()
open("yolov3_tiny_widerface_boxes_int8.tflite", "wb").write(tflite_model)

