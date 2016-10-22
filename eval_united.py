#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the TensorDetect model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import scipy as scp
import scipy.misc
import numpy as np
import tensorflow as tf

import time

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, os.path.realpath('incl'))


import train as united_train

import tensorvision.train as train
import tensorvision.utils as utils
import tensorvision.core as core

my_logdir = ('/u/marvin/no_backup/RUNS/UnitedVision2/final_test/'
             'deep_road_united_with_class_2016_09_22_11.29')

data_file = "/u/marvin/no_backup/DATA/data_road/testing.txt"

output_folder = "/u/marvin/no_backup/results"
run_folder = 'tracking'

output_folder = os.path.join(output_folder, run_folder)


def _output_generator(sess, tensor_list, image_pl, data_file,
                      process_image=lambda x: x):
    image_dir = os.path.dirname(data_file)
    with open(data_file) as file:
        for datum in file:
            datum = datum.rstrip()
            image_file = datum.split(" ")[0]
            image_file = os.path.join(image_dir, image_file)

            image = scp.misc.imread(image_file)
            image = process_image(image)

            feed_dict = {image_pl: image}
            start_time = time.time()
            output = sess.run(tensor_list, feed_dict=feed_dict)
            yield image_file, output


def eval_runtime(sess, subhypes, image_pl, eval_list, data_file):
    image_dir = os.path.dirname(data_file)
    with open(data_file) as file:
        for datum in file:
            datum = datum.rstrip()
    image_file = datum.split(" ")[0]
    image_file = os.path.join(image_dir, image_file)
    image = scp.misc.imread(image_file)
    image = process_image(subhypes, image)
    feed = {image_pl: image}
    for i in xrange(100):
        _ = sess.run(eval_list, feed_dict=feed)
    start_time = time.time()
    for i in xrange(100):
        _ = sess.run(eval_list, feed_dict=feed)
    dt = (time.time() - start_time)/100
    logging.info('Speed (msec) %f: ', 1000*dt)
    logging.info('Speed (fps) %f: ', 1/dt)
    return dt


def run_eval(load_out):
    meta_hypes, subhypes, submodules, decoded_logits, sess, image_pl = load_out
    assert(len(meta_hypes['model_list']) == 3)
    # inf_out['pred_boxes_new'], inf_out['pred_confidences']
    seg_softmax = decoded_logits['segmentation']['softmax']
    pred_boxes_new = decoded_logits['detection']['pred_boxes_new']
    pred_confidences = decoded_logits['detection']['pred_confidences']
    road_softmax = decoded_logits['road']['softmax'][0]
    eval_list = [seg_softmax, pred_boxes_new, pred_confidences, road_softmax]

    eval_runtime(sess, subhypes, image_pl, eval_list, data_file)
    exit(0)

    def my_process(image):
        process_image(subhypes, image)

    gen = _output_generator(sess, eval_list, image_pl, data_file, my_process)
    for image_file, output in gen:
        image = scp.misc.imread(image_file)
        shape = image.shape
        output = output[:, 1].reshape(shape[0], shape[1])
        hard = output > 0.5
        overlay_image = utils.fast_overlay(image, hard)
        im_name = os.path.basename(image_file)
        new_im_file = os.path.join(output_folder, im_name)
        scp.misc.imsave(new_im_file, overlay_image)
        print(image_file)


def process_image(subhypes, image):
    hypes = subhypes['road']
    shape = image.shape
    image_height = hypes['jitter']['image_height']
    image_width = hypes['jitter']['image_width']
    assert(image_height >= shape[0])
    assert(image_width >= shape[1])

    offset_x = (image_height - shape[0])//2
    offset_y = (image_width - shape[1])//2
    new_image = np.zeros([image_height, image_width, 3])
    new_image[offset_x:offset_x+shape[0],
              offset_y:offset_y+shape[1]] = image
    input_image = new_image
    return image


def load_united_model(logdir):
    subhypes = {}
    subgraph = {}
    submodules = {}
    subqueues = {}

    first_iter = True

    meta_hypes = utils.load_hypes_from_logdir(logdir, subdir="")
    for model in meta_hypes['models']:
        subhypes[model] = utils.load_hypes_from_logdir(logdir, subdir=model)
        hypes = subhypes[model]
        hypes['dirs']['output_dir'] = meta_hypes['dirs']['output_dir']
        hypes['dirs']['image_dir'] = meta_hypes['dirs']['image_dir']
        submodules[model] = utils.load_modules_from_logdir(logdir,
                                                           dirname=model,
                                                           postfix=model)

        modules = submodules[model]

    image_pl = tf.placeholder(tf.float32)
    image = tf.expand_dims(image_pl, 0)
    decoded_logits = {}
    for model in meta_hypes['models']:
        hypes = subhypes[model]
        modules = submodules[model]
        optimizer = modules['solver']

        with tf.name_scope('Validation_%s' % model):
            reuse = {True: False, False: True}[first_iter]

            with tf.variable_scope("", reuse=reuse):
                logits = modules['arch'].inference(hypes, image, train=False)

            decoded_logits[model] = modules['objective'].decoder(hypes, logits,
                                                                 train=False)

        first_iter = False
    sess = tf.Session()
    saver = tf.train.Saver()
    cur_step = core.load_weights(logdir, sess, saver)

    return meta_hypes, subhypes, submodules, decoded_logits, sess, image_pl


def main(_):
    utils.set_gpus_to_use()

    logdir = my_logdir
    utils.load_plugins()

    if 'TV_DIR_RUNS' in os.environ:
        os.environ['TV_DIR_RUNS'] = os.path.join(os.environ['TV_DIR_RUNS'],
                                                 'UnitedVision2')
    logging_file = os.path.join(logdir, "analysis.log")
    utils.create_filewrite_handler(logging_file, mode='a')
    load_out = load_united_model(logdir)

    run_eval(load_out)

    # stopping input Threads


if __name__ == '__main__':
    tf.app.run()
