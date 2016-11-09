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

import tensorflow_fcn

import time

import random

flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('logdir', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('hypes', 'hypes/multinet2.json',
                    'File storing model parameters.')

tf.app.flags.DEFINE_boolean(
    'save', True, ('Whether to save the run. In case --nosave (default) '
                   'output will be saved to the folder TV_DIR_RUNS/debug, '
                   'hence it will get overwritten by further runs.'))


def _print_training_status(hypes, step, loss_values, start_time, lr):

    # Prepare printing
    duration = (time.time() - start_time) / int(utils.cfg.step_show)
    examples_per_sec = hypes['solver']['batch_size'] / duration
    sec_per_batch = float(duration)

    if len(loss_values.keys()) >= 2:
        info_str = ('Step {step}/{total_steps}: losses = ({loss_value1:.2f}, '
                    '{loss_value2:.2f});'
                    ' lr = ({lr_value1:.2e}, {lr_value2:.2e}); '
                    '({sec_per_batch:.3f} sec)')
        losses = loss_values.values()
        lrs = lr.values()
        logging.info(info_str.format(step=step,
                                     total_steps=hypes['solver']['max_steps'],
                                     loss_value1=losses[0],
                                     loss_value2=losses[1],
                                     lr_value1=lrs[0],
                                     lr_value2=lrs[1],
                                     sec_per_batch=sec_per_batch)
                     )
    else:
        assert(False)


def build_training_graph(hypes, queue, modules, first_iter):
    """
    Build the tensorflow graph out of the model files.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    queue: tf.queue
        Data Queue
    modules : tuple
        The modules load in utils.

    Returns
    -------
    tuple
        (q, train_op, loss, eval_lists) where
        q is a dict with keys 'train' and 'val' which includes queues,
        train_op is a tensorflow op,
        loss is a float,
        eval_lists is a dict with keys 'train' and 'val'
    """

    data_input = modules['input']
    encoder = modules['arch']
    objective = modules['objective']
    optimizer = modules['solver']

    reuse = {True: False, False: True}[first_iter]

    scope = tf.get_variable_scope()

    with tf.variable_scope(scope, reuse=reuse):

        learning_rate = tf.placeholder(tf.float32)

        # Add Input Producers to the Graph
        with tf.name_scope("Inputs"):
            image, labels = data_input.inputs(hypes, queue, phase='train')

        # Run inference on the encoder network
        logits = encoder.inference(hypes, image, train=True)

    # Build decoder on top of the logits
    decoded_logits = objective.decoder(hypes, logits, train=True)

    # Add to the Graph the Ops for loss calculation.
    with tf.name_scope("Loss"):
        losses = objective.loss(hypes, decoded_logits,
                                labels)

    # Add to the Graph the Ops that calculate and apply gradients.
    with tf.name_scope("Optimizer"):
        global_step = tf.Variable(0, trainable=False)
        # Build training operation
        train_op = optimizer.training(hypes, losses,
                                      global_step, learning_rate)

    with tf.name_scope("Evaluation"):
        # Add the Op to compare the logits to the labels during evaluation.
        eval_list = objective.evaluation(
            hypes, image, labels, decoded_logits, losses, global_step)

        summary_op = tf.merge_all_summaries()

    graph = {}
    graph['losses'] = losses
    graph['eval_list'] = eval_list
    graph['summary_op'] = summary_op
    graph['train_op'] = train_op
    graph['global_step'] = global_step
    graph['learning_rate'] = learning_rate

    return graph


def run_united_training(meta_hypes, subhypes, submodules, subgraph, tv_sess,
                        start_step=0):

    """Run one iteration of training."""
    # Unpack operations for later use
    summary = tf.Summary()
    sess = tv_sess['sess']
    summary_writer = tv_sess['writer']

    solvers = {}
    for model in meta_hypes['models']:
        solvers[model] = submodules[model]['solver']

    display_iter = meta_hypes['logging']['display_iter']
    write_iter = meta_hypes['logging'].get('write_iter', 5*display_iter)
    eval_iter = meta_hypes['logging']['eval_iter']
    save_iter = meta_hypes['logging']['save_iter']
    image_iter = meta_hypes['logging'].get('image_iter', 5*save_iter)

    models = meta_hypes['model_list']
    num_models = len(models)

    py_smoothers = {}
    dict_smoothers = {}
    for model in models:
        py_smoothers[model] = train.ExpoSmoother(0.95)
        dict_smoothers[model] = train.MedianSmoother(0.50)

    n = 0

    eval_names = {}
    eval_ops = {}
    for model in models:
        names, ops = zip(*subgraph[model]['eval_list'])
        eval_names[model] = names
        eval_ops[model] = ops

    weights = meta_hypes['selection']['weights']
    aweights = np.array([sum(weights[:i+1]) for i in range(len(weights))])
    # eval_names, eval_ops = zip(*tv_graph['eval_list'])
    # Run the training Step
    start_time = time.time()
    for step in xrange(start_step, meta_hypes['solver']['max_steps']):

        # select on which model to run the training step
        # select model randomly?
        if not meta_hypes['selection']['random']:
            if not meta_hypes['selection']['use_weights']:
                # non-random selection
                model = models[step % num_models]
            else:
                # non-random, some models are selected multiple times
                select = np.argmax((aweights > step % aweights[-1]))
                model = models[select]
        else:
            # random selection. Use weights
            # to increase chance
            r = random.random()
            select = np.argmax((aweights > r))
            model = models[select]

        lr = solvers[model].get_learning_rate(subhypes[model], step)
        feed_dict = {subgraph[model]['learning_rate']: lr}

        sess.run([subgraph[model]['train_op']], feed_dict=feed_dict)

        # Write the summaries and print an overview fairly often.
        if step % display_iter == 0:
            # Print status to stdout.
            loss_values = {}
            eval_results = {}
            lrs = {}
            if select == 1:
                logging.info("Detection Loss was used.")
            else:
                logging.info("Segmentation Loss was used.")
            for model in models:
                loss_values[model] = sess.run(subgraph[model]['losses']
                                              ['total_loss'])

                eval_results[model] = sess.run(eval_ops[model])
                dict_smoothers[model].update_weights(eval_results[model])
                lrs[model] = solvers[model].get_learning_rate(subhypes[model],
                                                              step)

            _print_training_status(meta_hypes, step,
                                   loss_values,
                                   start_time, lrs)

            for model in models:
                train._print_eval_dict(eval_names[model], eval_results[model],
                                       prefix='   (raw)')

                smoothed_results = dict_smoothers[model].get_weights()

                train._print_eval_dict(eval_names[model], smoothed_results,
                                       prefix='(smooth)')

            if step % write_iter == 0:
                # write values to summary
                summary_str = sess.run(tv_sess['summary_op'],
                                       feed_dict=feed_dict)
                summary_writer.add_summary(summary_str,
                                           global_step=step)
                for model in models:
                    summary.value.add(tag='training/%s/total_loss' % model,
                                      simple_value=float(loss_values[model]))
                    summary.value.add(tag='training/%s/learning_rate' % model,
                                      simple_value=lrs[model])
                summary_writer.add_summary(summary, step)
                # Convert numpy types to simple types.
                if False:
                    eval_results = np.array(eval_results)
                    eval_results = eval_results.tolist()
                    eval_dict = zip(eval_names[model], eval_results)
                    train._write_eval_dict_to_summary(eval_dict,
                                                      'Eval/%s/raw' % model,
                                                      summary_writer, step)
                    eval_dict = zip(eval_names[model], smoothed_results)
                    train._write_eval_dict_to_summary(eval_dict,
                                                      'Eval/%s/smooth' % model,
                                                      summary_writer, step)

            # Reset timer
            start_time = time.time()

        # Do a evaluation and print the current state
        if (step) % eval_iter == 0 and step > 0 or \
           (step + 1) == meta_hypes['solver']['max_steps']:
            # write checkpoint to disk

            logging.info('Running Evaluation Scripts.')
            for model in models:
                eval_dict, images = submodules[model]['eval'].evaluate(
                    subhypes[model], sess,
                    subgraph[model]['image_pl'],
                    subgraph[model]['inf_out'])

                train._write_images_to_summary(images, summary_writer, step)

                if images is not None and len(images) > 0:

                    name = str(n % 10) + '_' + images[0][0]
                    image_dir = subhypes[model]['dirs']['image_dir']
                    image_file = os.path.join(image_dir, name)
                    scp.misc.imsave(image_file, images[0][1])
                    n = n + 1

                logging.info("%s Evaluation Finished. Results" % model)

                logging.info('Raw Results:')
                utils.print_eval_dict(eval_dict, prefix='(raw)   ')
                train._write_eval_dict_to_summary(
                    eval_dict, 'Evaluation/%s/raw' % model,
                    summary_writer, step)

                logging.info('Smooth Results:')
                names, res = zip(*eval_dict)
                smoothed = py_smoothers[model].update_weights(res)
                eval_dict = zip(names, smoothed)
                utils.print_eval_dict(eval_dict, prefix='(smooth)')
                train._write_eval_dict_to_summary(
                    eval_dict, 'Evaluation/%s/smoothed' % model,
                    summary_writer, step)

                if step % image_iter == 0 and step > 0 or \
                        (step + 1) == meta_hypes['solver']['max_steps']:
                    train._write_images_to_disk(meta_hypes, images, step)
            logging.info("Evaluation Finished. All results will be saved to:")
            logging.info(subhypes[model]['dirs']['output_dir'])

            # Reset timer
            start_time = time.time()

        # Save a checkpoint periodically.
        if (step) % save_iter == 0 and step > 0 or \
           (step + 1) == meta_hypes['solver']['max_steps']:
            # write checkpoint to disk
            checkpoint_path = os.path.join(meta_hypes['dirs']['output_dir'],
                                           'model.ckpt')
            tv_sess['saver'].save(sess, checkpoint_path, global_step=step)
            # Reset timer
            start_time = time.time()
    return


def _recombine_2_losses(meta_hypes, subgraph, subhypes, submodules):
    if meta_hypes['loss_build']['recombine']:
        # Computing weight loss
        enc_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        dec_loss = tf.add_n(tf.get_collection('dec_losses'), name='total_loss')
        fc_loss = tf.add_n(tf.get_collection('fc_wlosses'), name='total_loss')

        if meta_hypes['loss_build']['fc_loss']:
            weight_loss = enc_loss + dec_loss + fc_loss
        else:
            weight_loss = enc_loss + dec_loss

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
            total_loss = segmentation_loss + detection_loss + weight_loss
            subgraph['segmentation']['losses']['total_loss'] = total_loss
            detection_loss = detection_loss + weight_loss
            subgraph['detection']['losses']['total_loss'] = detection_loss

        for model in meta_hypes['model_list']:
            hypes = subhypes[model]
            modules = submodules[model]
            optimizer = modules['solver']
            gs = subgraph[model]['global_step']
            losses = subgraph[model]['losses']
            lr = subgraph[model]['learning_rate']
            subgraph[model]['train_op'] = optimizer.training(hypes, losses,
                                                             gs, lr)


def _recombine_3_losses(meta_hypes, subgraph, subhypes, submodules):
    if meta_hypes['loss_build']['recombine']:
        enc_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        dec_loss = tf.add_n(tf.get_collection('dec_losses'), name='total_loss')
        fc_loss = tf.add_n(tf.get_collection('fc_wlosses'), name='total_loss')
        segmentation_loss = subgraph['segmentation']['losses']['xentropy']
        detection_loss = subgraph['detection']['losses']['loss']

        if meta_hypes['loss_build']['fc_loss']:
            weight_loss = enc_loss + dec_loss + fc_loss
        else:
            weight_loss = enc_loss + dec_loss
        road_loss = subgraph['road']['losses']['loss']

        subgraph['segmentation']['losses']['total_loss'] = \
            segmentation_loss + detection_loss + road_loss + weight_loss

        if not meta_hypes['loss_build']['rwdecay']:
            subgraph['detection']['losses']['total_loss'] = \
                detection_loss + weight_loss

        for model in meta_hypes['models']:
            hypes = subhypes[model]
            modules = submodules[model]
            optimizer = modules['solver']
            gs = subgraph[model]['global_step']
            losses = subgraph[model]['losses']
            lr = subgraph[model]['learning_rate']
            subgraph[model]['train_op'] = optimizer.training(hypes, losses,
                                                             gs, lr)
        if meta_hypes['loss_build']['reAdam']:
            subgraph['road']['losses']['total_loss'] = \
                segmentation_loss + weight_loss
            lr = subgraph['segmentation']['learning_rate']
            subgraph['road']['learning_rate'] = lr
            gs = subgraph['segmentation']['global_step']
            optimizer = submodules['segmentation']['solver']
            opt = subhypes['segmentation']['opt']
            subgraph['road']['train_op'] = optimizer.training(hypes, losses,
                                                              gs, lr, opt=opt)


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

        logging.info("Build %s computation Graph.", model)
        with tf.name_scope("Queues_%s" % model):
            subqueues[model] = modules['input'].create_queues(hypes, 'train')

        logging.info('Building Model: %s' % model)

        subgraph[model] = build_training_graph(hypes,
                                               subqueues[model],
                                               modules,
                                               first_iter)

        first_iter = False

    if len(meta_hypes['models']) == 2:
        _recombine_2_losses(meta_hypes, subgraph, subhypes, submodules)
    else:
        _recombine_3_losses(meta_hypes, subgraph, subhypes, submodules)

    tv_sess = core.start_tv_session(hypes)
    sess = tv_sess['sess']
    saver = tv_sess['saver']

    cur_step = core.load_weights(logdir, sess, saver)
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

    return meta_hypes, subhypes, submodules, subgraph, tv_sess, cur_step


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

        subgraph[model] = build_training_graph(hypes,
                                               subqueues[model],
                                               modules,
                                               first_iter)

        first_iter = False

    if len(meta_hypes['models']) == 2:
        _recombine_2_losses(meta_hypes, subgraph, subhypes, submodules)
    else:
        _recombine_3_losses(meta_hypes, subgraph, subhypes, submodules)

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

    load_weights = tf.app.flags.FLAGS.logdir is not None

    if not load_weights:
        with open(tf.app.flags.FLAGS.hypes, 'r') as f:
            logging.info("f: %s", f)
            hypes = json.load(f)
    utils.load_plugins()

    if 'TV_DIR_RUNS' in os.environ:
        os.environ['TV_DIR_RUNS'] = os.path.join(os.environ['TV_DIR_RUNS'],
                                                 'UnitedVision2')

    if not load_weights:
        utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)
        utils._add_paths_to_sys(hypes)

        # Build united Model
        subhypes, submodules, subgraph, tv_sess = build_united_model(hypes)
        start_step = 0
    else:
        logdir = tf.app.flags.FLAGS.logdir
        logging_file = os.path.join(logdir, "output.log")
        utils.create_filewrite_handler(logging_file, mode='a')
        hypes, subhypes, submodules, subgraph, tv_sess, start_step = \
            load_united_model(logdir)
        if start_step is None:
            start_step = 0

    # Run united training
    run_united_training(hypes, subhypes, submodules, subgraph,
                        tv_sess, start_step=start_step)

    # stopping input Threads
    tv_sess['coord'].request_stop()
    tv_sess['coord'].join(tv_sess['threads'])


if __name__ == '__main__':
    tf.app.run()
