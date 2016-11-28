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
from PIL import Image, ImageDraw, ImageFont

flags.DEFINE_string('data',
                    "data_road/testing.txt",
                    'Text file containing images.')

res_folder = 'results'


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
    logging.info(' ')
    logging.info('Evaluation complete. Measuring runtime.')
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
    logging.info('Joined inference can be conducted at the following rates on'
                 ' your machine:')
    logging.info('Speed (msec): %f ', 1000*dt)
    logging.info('Speed (fps): %f ', 1/dt)
    return dt


def test_constant_input(subhypes):
    road_input_conf = subhypes['road']['jitter']
    seg_input_conf = subhypes['segmentation']['jitter']
    car_input_conf = subhypes['detection']

    gesund = True \
        and road_input_conf['image_width'] == seg_input_conf['image_width'] \
        and road_input_conf['image_height'] == seg_input_conf['image_height'] \
        and car_input_conf['image_width'] == seg_input_conf['image_width'] \
        and car_input_conf['image_height'] == seg_input_conf['image_height'] \

    if not gesund:
        logging.error("The different tasks are training"
                      "using different resolutions. Please retrain all tasks,"
                      "using the same resolution.")
        exit(1)
    return


def test_segmentation_input(subhypes):

    if not subhypes['segmentation']['jitter']['reseize_image']:
        logging.error('')
        logging.error("Issue with Segmentation input handling.")
        logging.error("Segmentation input will be resized during this"
                      "evaluation, but was not resized during training.")
        logging.error("This will lead to bad results.")
        logging.error("To use this script please train segmentation using"
                      "the configuration:.")
        logging.error("""
{
    "jitter": {
    "reseize_image": true,
    "image_height" : 384,
    "image_width" : 1248,
    },
}""")
        logging.error("Alternatively implement evaluation using non-resized"
                      " input.")
        exit(1)
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


def run_eval(load_out, output_folder, data_file):
    meta_hypes, subhypes, submodules, decoded_logits, sess, image_pl = load_out
    assert(len(meta_hypes['model_list']) == 3)
    # inf_out['pred_boxes_new'], inf_out['pred_confidences']
    seg_softmax = decoded_logits['segmentation']['softmax']
    pred_boxes_new = decoded_logits['detection']['pred_boxes_new']
    pred_confidences = decoded_logits['detection']['pred_confidences']
    road_softmax = decoded_logits['road']['softmax'][0]
    eval_list = [seg_softmax, pred_boxes_new, pred_confidences, road_softmax]

    def my_process(image):
        return process_image(subhypes, image)

    eval_runtime(sess, subhypes, image_pl, eval_list, data_file)
    exit(0)

    test_constant_input(subhypes)
    test_segmentation_input(subhypes)

    import utils.train_utils as dec_utils

    gen = _output_generator(sess, eval_list, image_pl, data_file, my_process)
    for image_file, output in gen:
        image = scp.misc.imread(image_file)
        image = process_image(subhypes, image)
        shape = image.shape
        seg_softmax, pred_boxes_new, pred_confidences, road_softmax = output

        # Create Segmentation Overlay
        shape = image.shape
        seg_softmax = seg_softmax[:, 1].reshape(shape[0], shape[1])
        hard = seg_softmax > 0.5
        overlay_image = utils.fast_overlay(image, hard)

        # Draw Detection Boxes
        new_img, rects = dec_utils.add_rectangles(
            subhypes['detection'], [overlay_image], pred_confidences,
            pred_boxes_new, show_removed=False,
            use_stitching=True, rnn_len=subhypes['detection']['rnn_len'],
            min_conf=0.5, tau=0.4)

        # Draw road classification
        highway = (np.argmax(output[0][0]) == 0)
        new_img = road_draw(new_img, highway)

        # Save image file
        im_name = os.path.basename(image_file)
        new_im_file = os.path.join(output_folder, im_name)
        im_name = os.path.basename(image_file)
        new_im_file = os.path.join(output_folder, im_name)
        scp.misc.imsave(new_im_file, new_img)

        logging.info("Plotting file: {}".format(new_im_file))

    eval_runtime(sess, subhypes, image_pl, eval_list, data_file)
    exit(0)


def process_image(subhypes, image):
    hypes = subhypes['road']
    shape = image.shape
    image_height = hypes['jitter']['image_height']
    image_width = hypes['jitter']['image_width']
    assert(image_height >= shape[0])
    assert(image_width >= shape[1])

    image = scp.misc.imresize(image, (image_height,
                                      image_width, 3),
                              interp='cubic')
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

            scope = tf.get_variable_scope()

            with tf.variable_scope(scope, reuse=reuse):
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

    logdir = FLAGS.logdir
    data_file = FLAGS.data

    if logdir is None:
        logging.error('Usage python predict_joint --logdir /path/to/logdir'
                      '--data /path/to/data/txt')
        exit(1)

    output_folder = os.path.join(logdir, res_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    logdir = logdir
    utils.load_plugins()

    if 'TV_DIR_DATA' in os.environ:
        data_file = os.path.join(os.environ['TV_DIR_DATA'], data_file)
    else:
        data_file = os.path.join('DATA', data_file)

    if not os.path.exists(data_file):
        logging.error('Please provide a valid data_file.')
        logging.error('Use --data_file')
        exit(1)

    if 'TV_DIR_RUNS' in os.environ:
        os.environ['TV_DIR_RUNS'] = os.path.join(os.environ['TV_DIR_RUNS'],
                                                 'UnitedVision2')
    logging_file = os.path.join(output_folder, "analysis.log")
    utils.create_filewrite_handler(logging_file, mode='a')
    load_out = load_united_model(logdir)

    run_eval(load_out, output_folder, data_file)

    # stopping input Threads


if __name__ == '__main__':
    tf.app.run()
