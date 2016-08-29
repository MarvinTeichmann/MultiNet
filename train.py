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
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, os.path.realpath('incl'))

import tensorvision.train as train
import tensorvision.utils as utils
import tensorvision.core as core

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


def create_united_model(hypes):

    logging.info("Initialize training folder")

    subhypes = {}
    subgraph = {}
    submodules = {}

    base_path = hypes['dirs']['base_path']
    first_iter = True

    for model in hypes['models']:
        subhypes_file = os.path.join(base_path, hypes['models'][model])
        with open(subhypes_file, 'r') as f:
            logging.info("f: %s", f)
            subhypes[model] = json.load(f)
        subh = subhypes[model]
        utils.set_dirs(subh, subhypes_file)
        subh['dirs']['output_dir'] = hypes['dirs']['output_dir']
        train.initialize_training_folder(subh, files_dir=model,
                                         logging=first_iter)
        train.maybe_download_and_extract(subh)
        submodules[model] = utils.load_modules_from_hypes(subh)
        modules = submodules[model]

        logging.info("Build %s computation Graph.")
        with tf.name_scope("Queues"):
            queue = modules['input'].create_queues(subh, 'train')

        reuse = {True: False, False: None}[first_iter]
        with tf.variable_scope(model, reuse=reuse):
            subgraph[model] = core.build_training_graph(subh, queue, modules)

        first_iter = False
        logging.info("Start training")
        logging.info("Finished training")


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
    create_united_model(hypes)


if __name__ == '__main__':
    tf.app.run()
