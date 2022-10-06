# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 20:55:30 2022

@author: weiwei
@function:
"""

from __future__ import division
import os
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import network as net
import test_functions


class MALAR(object):
    def __init__(self, sess, args):
        self.sess = sess

        # save directory
        self.checkpoint_dir = args.checkpoint_dir

        # network
        self.generator = net.generator

        # network options
        OPTIONS = namedtuple('OPTIONS', 'gf_dim glf_dim df_dim')
        self.options = OPTIONS._make((args.ngf, args.nglf, args.ndf))

        # build model
        self.test_X = tf.placeholder(tf.float32, [None, args.img_size, args.img_size, 1], name='X')
        self.test_Y = tf.placeholder(tf.float32, [None, args.img_size, args.img_size, 1], name='Y')
        with tf.variable_scope(tf.get_variable_scope()):
            self.test_G_X = self.generator(self.test_X, self.options, reuse=False, name="generatorX2Y")

        # model saver
        self.saver = tf.train.Saver(max_to_keep=None)

    # load trained model
    def load(self):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_step = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        init_1 = tf.global_variables_initializer()
        init_2 = tf.local_variables_initializer()
        self.sess.run(init_1)
        self.sess.run(init_2)

        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # mk save dir (image & numpy file)
        png_save_dir = args.test_save_dir
        if not os.path.exists(png_save_dir):
            os.makedirs(png_save_dir)

        # dicom path
        UDCT_path = [d for d in glob(args.UDCT_path + '/*')]
        UDCT_path.sort()
        # truncated pixels info path
        UDCT_pixel_info = [d for d in glob(args.pi_path + '/*')]
        UDCT_pixel_info.sort()

        for i in range(len(UDCT_path)):
            """
            testing procedure:
            1.read .dicom
            2.customized windowing & normalization & crop ROI
            3.reconstruction based on generatorX2Y
            4.reverse customized windowing
            5.save to .dicom
            """

            # 1.read .dicom
            _dicom, _array = test_functions.read_dicom(UDCT_path[i])
            _name = UDCT_path[i].replace('\\', '/').split('/')[-1]

            # 2.for img array, customized windowing & normalization & crop ROI
            _array = np.clip(_array, -300, 700)
            _array = 2 * (_array + 300) / float(1000) - 1
            _array = _array[79:463, 66:450]

            # 3.reconstruction
            _array = _array.reshape([1] + self.test_X.get_shape().as_list()[1:])
            prediction = self.sess.run(self.test_G_X,
                                       feed_dict={self.test_X: _array})
            prediction = np.array(prediction[0, :, :, 0]).astype(np.float32)
            prediction = (prediction + 1) / 2 * 1000 - 300 + 1024  # reverse normalization

            # 4.reverse customized windowing
            _pi_path = UDCT_pixel_info[i]
            _pi = np.load(_pi_path).astype(np.int16)
            prediction = test_functions.load_pixel_info(prediction, _pi).astype(np.uint16)

            # 5.save to .dicom
            _dicom.Rows, _dicom.Columns = prediction.shape
            _dicom.PixelData = prediction.tobytes()
            _dicom.save_as(os.path.join(png_save_dir, _name))

            print("----- Testing %s, Saving to dicom -----," % _name)

        print("Test complete ---!!!")
