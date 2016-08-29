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


def _print_training_status(hypes, step, loss_values, start_time, lr):

    # Prepare printing
    duration = (time.time() - start_time) / int(utils.cfg.step_show)
    examples_per_sec = hypes['solver']['batch_size'] / duration
    sec_per_batch = float(duration)

    if len(loss_values.keys()) == 2:
        info_str = ('Step {step}/{total_steps}: losses = ( {loss_value1:.2f}, '
                    '{loss_value2:.2f} );'
                    ' lr = ( {lr_value1:.2e}, {lr_value2:.2e} ); '
                    '( {sec_per_batch:.3f} sec )')
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
        dict_smoothers[model] = train.ExpoSmoother(0.95)

    n = 0

    eval_names = {}
    eval_ops = {}
    for model in models:
        names, ops = zip(*subgraph[model]['eval_list'])
        eval_names[model] = names
        eval_ops[model] = ops

    # eval_names, eval_ops = zip(*tv_graph['eval_list'])
    # Run the training Step
    start_time = time.time()
    for step in xrange(start_step, meta_hypes['solver']['max_steps']):

        model = models[step % num_models]
        lr = solvers[model].get_learning_rate(subhypes[model], step)
        feed_dict = {subgraph[model]['learning_rate']: lr}

        sess.run([subgraph[model]['train_op']], feed_dict=feed_dict)

        # Write the summaries and print an overview fairly often.
        if step % display_iter == 0:
            # Print status to stdout.
            loss_values = {}
            eval_results = {}
            lrs = {}
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
        meta_hypes['dirs']['image_dir'] = hypes['dirs']['image_dir']
        train.initialize_training_folder(hypes, files_dir=model,
                                         logging=first_iter)
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

    tv_sess = core.start_tv_session(hypes)
    sess = tv_sess['sess']
    for model in meta_hypes['models']:
        hypes = subhypes[model]
        modules = submodules[model]

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

    my_loss = subgraph['segmentation']['losses']['total_loss']
    my_loss2 = subgraph['detection']['losses']['total_loss']
    seg_train = subgraph['segmentation']['train_op']
    dec_train = subgraph['detection']['train_op']

    logging.info("Start training")

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
