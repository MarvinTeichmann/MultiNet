#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Marvin Teichmann


"""
Detects Cars in an image using KittiSeg.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiSeg weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf

import time

from PIL import Image, ImageDraw, ImageFont


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

try:
    # Check whether setup was done correctly

    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)


flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input', None,
                    'Image to apply KittiSeg.')
flags.DEFINE_string('output', None,
                    'Image to apply KittiSeg.')


default_run = 'MultiNet_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/MultiNet_pretrained.zip")


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
        # weights are downloaded. Nothing to do

    import zipfile
    download_name = tv_utils.download(weights_url, runs_dir)

    logging.info("Extracting MultiNet_pretrained.zip")

    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


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

    meta_hypes = tv_utils.load_hypes_from_logdir(logdir, subdir="")
    for model in meta_hypes['models']:
        subhypes[model] = tv_utils.load_hypes_from_logdir(logdir, subdir=model)
        hypes = subhypes[model]
        hypes['dirs']['output_dir'] = meta_hypes['dirs']['output_dir']
        hypes['dirs']['image_dir'] = meta_hypes['dirs']['image_dir']
        submodules[model] = tv_utils.load_modules_from_logdir(logdir,
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
    tv_utils.set_gpus_to_use()

    if FLAGS.input is None:
        logging.error("No input was given.")
        logging.info(
            "Usage: python demo.py --input data/test.png "
            "[--output_image output_image] [--logdir /path/to/weights] "
            "[--gpus GPUs_to_use] ")
        exit(1)

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'MultiNet')
        else:
            runs_dir = 'RUNS'
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loads the model from rundir
    load_out = load_united_model(logdir)

    # Create list of relevant tensors to evaluate
    meta_hypes, subhypes, submodules, decoded_logits, sess, image_pl = load_out

    seg_softmax = decoded_logits['segmentation']['softmax']
    pred_boxes_new = decoded_logits['detection']['pred_boxes_new']
    pred_confidences = decoded_logits['detection']['pred_confidences']
    if len(meta_hypes['model_list']) == 3:
        road_softmax = decoded_logits['road']['softmax'][0]
    else:
        road_softmax = None

    eval_list = [seg_softmax, pred_boxes_new, pred_confidences, road_softmax]

    # Run some tests on the hypes
    test_constant_input(subhypes)
    test_segmentation_input(subhypes)

    # Load and reseize Image
    image_file = FLAGS.input
    image = scp.misc.imread(image_file)

    hypes_road = subhypes['road']
    shape = image.shape
    image_height = hypes_road['jitter']['image_height']
    image_width = hypes_road['jitter']['image_width']
    assert(image_height >= shape[0])
    assert(image_width >= shape[1])

    image = scp.misc.imresize(image, (image_height,
                                      image_width, 3),
                              interp='cubic')

    import utils.train_utils as dec_utils

    # Run KittiSeg model on image
    feed_dict = {image_pl: image}
    output = sess.run(eval_list, feed_dict=feed_dict)

    seg_softmax, pred_boxes_new, pred_confidences, road_softmax = output

    # Create Segmentation Overlay
    shape = image.shape
    seg_softmax = seg_softmax[:, 1].reshape(shape[0], shape[1])
    hard = seg_softmax > 0.5
    overlay_image = tv_utils.fast_overlay(image, hard)

    # Draw Detection Boxes
    new_img, rects = dec_utils.add_rectangles(
        subhypes['detection'], [overlay_image], pred_confidences,
        pred_boxes_new, show_removed=False,
        use_stitching=True, rnn_len=subhypes['detection']['rnn_len'],
        min_conf=0.50, tau=subhypes['detection']['tau'])

    # Draw road classification
    highway = (np.argmax(output[0][0]) == 0)
    new_img = road_draw(new_img, highway)

    logging.info("")

    # Printing some more output information
    threshold = 0.5
    accepted_predictions = []
    # removing predictions <= threshold
    for rect in rects:
        if rect.score >= threshold:
            accepted_predictions.append(rect)

    print('')
    logging.info("{} Cars detected".format(len(accepted_predictions)))

    # Printing coordinates of predicted rects.
    for i, rect in enumerate(accepted_predictions):
        logging.info("")
        logging.info("Coordinates of Box {}".format(i))
        logging.info("    x1: {}".format(rect.x1))
        logging.info("    x2: {}".format(rect.x2))
        logging.info("    y1: {}".format(rect.y1))
        logging.info("    y2: {}".format(rect.y2))
        logging.info("    Confidence: {}".format(rect.score))

    if len(meta_hypes['model_list']) == 3:
        logging.info("Raw Classification Softmax outputs are: {}"
                     .format(output[0][0]))

    # Save output image file
    if FLAGS.output is None:
        output_base_name = input
        out_image_name = output_base_name.split('.')[0] + '_out.png'
    else:
        out_image_name = FLAGS.output

    scp.misc.imsave(out_image_name, new_img)

    logging.info("")
    logging.info("Output image has been saved to: {}".format(
        os.path.realpath(out_image_name)))

    exit(0)

if __name__ == '__main__':
    tf.app.run()
