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

import scipy as scp

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
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, os.path.realpath('incl'))

import tensorvision.train as train
import tensorvision.utils as utils
import tensorvision.core as core

from PIL import Image, ImageDraw, ImageFont

import time

import random

seg_folder = ("/u/marvin/no_backup/RUNS/KittiSeg/"
              "final_test/kitti_fcn_2016_09_19_07.47")

dec_folder = ("/u/marvin/no_backup/RUNS/TensorDetect2/"
              "final_test/kitti_2016_09_19_07.14")

road_folder = ("/u/marvin/no_backup/RUNS/road_class/"
               "final_test/two_classes_road_2016_09_20_13.57")

data_file = "/u/marvin/no_backup/DATA/data_road/testing.txt"

output_folder = "/u/marvin/no_backup/results"
run_folder = 'tracking'

output_folder = os.path.join(output_folder, run_folder)


tf.reset_default_graph()


def _load_graph_from_logdir(logdir):
    hypes = utils.load_hypes_from_logdir(logdir)
    modules = utils.load_modules_from_logdir(logdir)

    image_pl = tf.placeholder(tf.float32)
    image = tf.expand_dims(image_pl, 0)
    inf_out = core.build_inference_graph(hypes, modules,
                                         image=image)

    sess = tf.Session()
    saver = tf.train.Saver()

    core.load_weights(logdir, sess, saver)

    return inf_out, image_pl, sess, hypes


def _output_generator(sess, tensor_list, image_pl, data_file=data_file,
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


def do_seg_eval(data_file=data_file, output_folder=output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    inf_out, image_pl, sess, hypes = _load_graph_from_logdir(seg_folder)
    gen = _output_generator(sess, inf_out['softmax'], image_pl,
                            data_file=data_file)

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
    return


def road_draw(image, highway):
    im = Image.fromarray(image.astype('uint8'))
    draw = ImageDraw.Draw(im)

    fnt = ImageFont.truetype('FreeMono/FreeMonoBold.ttf', 40)

    shape = image.shape

    if highway:
        draw.text((65, 10), "Highway",
                  font=fnt, fill=(255, 255, 0, 255))

        draw.ellipse([10, 10, 55, 55], fill=(255, 255, 0, 255),
                     outline=(255, 255, 0, 255))
    else:
        draw.text((65, 10), "minor road",
                  font=fnt, fill=(255, 0, 0, 255))

        draw.ellipse([10, 10, 55, 55], fill=(255, 0, 0, 255),
                     outline=(255, 0, 0, 255))

    return np.array(im).astype('float32')


def do_dec_eval(data_file=data_file, output_folder=output_folder, reuse=False):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    inf_out, image_pl, sess, hypes = _load_graph_from_logdir(dec_folder)
    eval_list = [inf_out['pred_boxes_new'], inf_out['pred_confidences']]

    def dec_process(image):
        image = scp.misc.imresize(image, (hypes["image_height"],
                                  hypes["image_width"]),
                                  interp='cubic')
        return image

    gen = _output_generator(sess, eval_list, image_pl,
                            data_file=data_file,
                            process_image=dec_process)

    import utils.train_utils as dec_utils

    for image_file, output in gen:
        image = scp.misc.imread(image_file)
        im_name = os.path.basename(image_file)
        new_im_file = os.path.join(output_folder, im_name)

        if reuse:
            image_file = new_im_file
        image = scp.misc.imread(image_file)

        img = scp.misc.imresize(image, (hypes["image_height"],
                                        hypes["image_width"]),
                                interp='cubic')
        pred_boxes, pred_confidences = output
        new_img, rects = dec_utils.add_rectangles(
            hypes, [img], pred_confidences,
            pred_boxes, show_removed=False,
            use_stitching=True, rnn_len=hypes['rnn_len'],
            min_conf=0.25, tau=hypes['tau'])
        shape = image.shape
        scp.misc.imsave(new_im_file, new_img)
        print(image_file)
    return


def do_road_eval(data_file=data_file, output_folder=output_folder,
                 reuse=False):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    inf_out, image_pl, sess, hypes = _load_graph_from_logdir(road_folder)
    softmax_road, softmax_cross = inf_out['softmax']
    eval_list = [softmax_road]

    def dec_process(image):
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

    gen = _output_generator(sess, eval_list, image_pl,
                            data_file=data_file,
                            process_image=dec_process)

    for image_file, output in gen:
        image = scp.misc.imread(image_file)
        im_name = os.path.basename(image_file)
        new_im_file = os.path.join(output_folder, im_name)

        if reuse:
            image_file = new_im_file
        image = scp.misc.imread(image_file)

        highway = (np.argmax(output[0][0]) == 0)
        image = road_draw(image, highway)
        im_name = os.path.basename(image_file)
        new_im_file = os.path.join(output_folder, im_name)
        scp.misc.imsave(new_im_file, image)
        print(image_file)

    return


def do_combined_eval(data_file=data_file, output_folder=output_folder):
    do_seg_eval(data_file, output_folder)
    tf.reset_default_graph()
    do_dec_eval(data_file, output_folder, reuse=True)
    tf.reset_default_graph()
    do_road_eval(data_file, output_folder, reuse=True)


def create_movies_from_rootdir(root_dir, output_folder=output_folder,
                               postfix="", func=do_combined_eval):
    for subdir, dirs, files in os.walk(root_dir):
        if subdir == root_dir:
            continue
    file_name = "tmp.txt"
    with open(file_name, "w") as f:
        for file in files:
            image_name = os.path.join(subdir, file)
            print(image_name, file=f)
    dirname = os.path.basename(subdir)
    new_folder = os.path.join(output_folder, dirname)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    func(data_file=file_name, output_folder=new_folder)


def create_movies(data_file=data_file, output_folder=output_folder):
    movie_list = ["segmentation", "detection", "classification", "combined"]
    function_list = [do_seg_eval, do_dec_eval, do_road_eval, do_combined_eval]
    for model, func in zip(movie_list, function_list):
        new_folder = os.path.join(output_folder, model)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        tf.reset_default_graph()
        func(output_folder=new_folder)


def main(_):
    utils.set_gpus_to_use()
    utils.load_plugins()
    root_dir = "/u/marvin/no_backup/DATA/training/image_02"
    create_movies_from_rootdir(root_dir, postfix="united",
                               func=do_combined_eval)


if __name__ == '__main__':
    tf.app.run()
