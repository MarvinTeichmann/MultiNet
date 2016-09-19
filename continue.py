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

import time

import random

flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('hypes', 'hypes/united.json',
                    'File storing model parameters.')

tf.app.flags.DEFINE_boolean(
    'save', True, ('Whether to save the run. In case --nosave (default) '
                   'output will be saved to the folder TV_DIR_RUNS/debug, '
                   'hence it will get overwritten by further runs.'))


def build_united_model(meta_hypes):

    logging.info("Initialize training folder")

    subhypes = {}
    subgraph = {}
    submodules = {}
    subqueues = {}

    base_path = meta_hypes['dirs']['base_path']
    first_iter = True

    for model in meta_hypes['models']:
        subhypes_file = os.path.join(base_path, meta_hypes['models'][model])
        with open(subhypes_file, 'r') as f:
            logging.info("f: %s", f)
            subhypes[model] = json.load(f)
        hypes = subhypes[model]
        utils.set_dirs(hypes, subhypes_file)
        hypes['dirs']['output_dir'] = meta_hypes['dirs']['output_dir']
        train.initialize_training_folder(hypes, files_dir=model,
                                         logging=first_iter)
        meta_hypes['dirs']['image_dir'] = hypes['dirs']['image_dir']
        train.maybe_download_and_extract(hypes)
        submodules[model] = utils.load_modules_from_hypes(
            hypes, postfix="_%s" % model)
        modules = submodules[model]

        logging.info("Build %s computation Graph.", model)
        with tf.name_scope("Queues_%s" % model):
            subqueues[model] = modules['input'].create_queues(hypes, 'train')

        logging.info('Building Model: %s' % model)

        reuse = {True: False, False: True}[first_iter]
        with tf.variable_scope("", reuse=reuse):
            subgraph[model] = core.build_training_graph(hypes,
                                                        subqueues[model],
                                                        modules)

        first_iter = False

    if meta_hypes['loss_build']['recombine']:
        weight_loss = subgraph['segmentation']['losses']['weight_loss']
        segmentation_loss = subgraph['segmentation']['losses']['xentropy']
        detection_loss = subgraph['detection']['losses']['loss']
        if meta_hypes['loss_build']['weighted']:
            w = meta_hypes['loss_build']['weights']
            total_loss = segmentation_loss*w[0] + \
                detection_loss*w[1] + weight_loss
            subgraph['segmentation']['losses']['total_loss'] = total_loss
            subgraph['detection']['losses']['total_loss'] = total_loss
        elif meta_hypes['loss_build']['rwdecay']:
            total_loss = segmentation_loss + detection_loss + weight_loss
            subgraph['segmentation']['losses']['total_loss'] = total_loss
            if meta_hypes['loss_build']['reweight']:
                w = meta_hypes['loss_build']["rdweight"]
                detection_loss = w[0]*weight_loss + detection_loss
            subgraph['detection']['losses']['total_loss'] = detection_loss
        else:
            total_loss = segmentation_loss + detection_loss
            subgraph['segmentation']['losses']['total_loss'] = total_loss

        for model in meta_hypes['models']:
            hypes = subhypes[model]
            modules = submodules[model]
            optimizer = modules['solver']
            gs = subgraph[model]['global_step']
            losses = subgraph[model]['losses']
            lr = subgraph[model]['learning_rate']
            subgraph[model]['train_op'] = optimizer.training(hypes, losses,
                                                             gs, lr)

    tv_sess = core.start_tv_session(hypes)
    sess = tv_sess['sess']
    for model in meta_hypes['models']:
        hypes = subhypes[model]
        modules = submodules[model]
        optimizer = modules['solver']

        with tf.name_scope('Validation_%s' % model):
            tf.get_variable_scope().reuse_variables()
            image_pl = tf.placeholder(tf.float32)
            image = tf.expand_dims(image_pl, 0)
            inf_out = core.build_inference_graph(hypes, modules,
                                                 image=image)
            subgraph[model]['image_pl'] = image_pl
            subgraph[model]['inf_out'] = inf_out

        # Start the data load
        modules['input'].start_enqueuing_threads(hypes, subqueues[model],
                                                 'train', sess)

    target_file = os.path.join(meta_hypes['dirs']['output_dir'], 'hypes.json')
    with open(target_file, 'w') as outfile:
        json.dump(meta_hypes, outfile, indent=2, sort_keys=True)

    return subhypes, submodules, subgraph, tv_sess


def main(_):
    utils.set_gpus_to_use()

    with open(tf.app.flags.FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)
    utils.load_plugins()

    if 'TV_DIR_RUNS' in os.environ:
        os.environ['TV_DIR_RUNS'] = os.path.join(os.environ['TV_DIR_RUNS'],
                                                 'UnitedVision')
    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)
    utils._add_paths_to_sys(hypes)

    # Build united Model
    subhypes, submodules, subgraph, tv_sess = build_united_model(hypes)

    # Run united training
    run_united_training(hypes, subhypes, submodules, subgraph, tv_sess)

    # stopping input Threads
    tv_sess['coord'].request_stop()
    tv_sess['coord'].join(tv_sess['threads'])


if __name__ == '__main__':
    tf.app.run()
