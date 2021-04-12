#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : videoRead.py.py
# @Author: Qinwei
# @Date  : 2021/3/19
# @Desc  :

import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()