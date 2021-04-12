#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : exam.py.py
# @Author: Qinwei
# @Date  : 2021/1/5
# @Desc  :
import tensorflow as tf
# a = tf.constant([[1,2,3,4],[5,6,7,8]])
# b = tf.constant([[1],[2]])
# c = tf.concat([a, b], axis=1)
# print(tf.Session().run(c))
import tensorflow as tf
import numpy as np
a_np = np.ones([0,3])
with tf.Graph().as_default(), tf.Session() as sess:
    a = tf.placeholder(shape=[None,3],dtype=tf.int8)
    size_a = tf.size(a)
    d = tf.constant([2,2])
    e = tf.constant([1,1])
    c = tf.equal(0,size_a)
    result = tf.cond(tf.equal(tf.size(a),0)|tf.not_equal(tf.size(a),0),
            lambda: tf.square(d),
            lambda: tf.square(e))
    print(sess.run(result, feed_dict={a: a_np}))
    print(sess.run(size_a, feed_dict={a: a_np}))

