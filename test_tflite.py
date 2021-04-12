# -*- coding:utf-8 -*-
# -*- author:ZuoJianHao -*-
# -*- coding:utf-8 -*-
# -*- author:ZuoJianHao -*-
import tensorflow as tf
import cv2
import numpy as np
weights = "./yolov3_tiny_widerface_boxes_int8.tflite"
image_path = "./416_416.jpg"

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

image_data = original_image[np.newaxis, ...].astype(np.float32)

interpreter = tf.lite.Interpreter(model_path=weights)
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)
interpreter.set_tensor(input_details[0]['index'], image_data)
interpreter.invoke()
pred_bbox = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
print(pred_bbox)