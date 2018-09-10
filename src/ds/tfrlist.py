###########################################
# tfrlist.py
#
# Estimate positive/negative label distribution in tensorflow dataset.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import cv2

import numpy as np

import tensorflow as tf

from utils import reset_tensorflow, create_logger, resizemax

from ds.reader import make_tfr_dataset, make_tfr_input_fn

logger = create_logger(__name__)


def run(sources, threads, prefetch, batch, steps, epochs, verbose, seed):
    """
    sources: list of tf-record files
    threads: number of concurrent read pipelines
    prefetch: reader prefetch count
    batch: batch size
    steps: number of steps per epoch
    epochs: number of epochs
    verbose: verbose output
    """

    reset_tensorflow(seed)

    datasets = [ make_tfr_dataset(source, threads=threads, read_rgb=False, read_mask=False, read_weight=False) for source in sources ]

    global_stats = dict()
    global_batch_counter = 0
    global_instance_counter = 0
    with tf.device('/cpu:0'):
        for epoch in range(epochs):
            for dataset in datasets:
                input_fn = make_tfr_input_fn(
                    dataset,
                    threads=threads,
                    offset=max(0, epoch*steps*batch),
                    limit=max(0, steps*batch),
                    shuffle=max(0, steps*batch),
                    prefetch=prefetch,
                    batch=batch,
                    repeat=-1
                )
                stats = dict()
                batch_counter = 0
                instance_counter = 0
                with tf.Session() as session:
                    try:
                        it = input_fn()
                        while batch_counter < steps:
                            features, label = session.run(it)
                            names = features['name']
                            variants = features['variant']
                            for i in range(names.shape[0]):
                                key = '%s-%s' % (str(names[i], 'utf-8'), str(variants[i], 'utf-8'))
                                if key not in stats:
                                    stats[key] = list()
                                stats[key].append(instance_counter)
                                instance_counter += 1
                            batch_counter += 1
                    except tf.errors.DataLossError as e:
                        logger.warn(e)
                        pass
                    except tf.errors.OutOfRangeError:
                        pass
                logger.info(
                    "EPOCH[%d]: %d / %d batches, %d / %d instances",
                    epoch,
                    batch_counter,
                    steps,
                    instance_counter,
                    steps * batch
                )
                for key in sorted(stats.keys()):
                    if verbose:
                        logger.debug("  %20s: %4d %s", key, len(stats[key]), stats[key])
                    if key not in global_stats:
                        global_stats[key] = 0
                    global_stats[key] += len(stats[key])
                global_batch_counter += batch_counter
                global_instance_counter += instance_counter

    logger.info(
        "FINAL: %d / %d batches, %d / %d instances",
        global_batch_counter,
        epochs * len(datasets) * steps,
        global_instance_counter,
        epochs * len(datasets) * steps * batch
    )
    min_p = 1.0
    max_p = 0.0
    sum_p = 0.0
    for key in sorted(global_stats.keys()):
        p = global_stats[key] / global_instance_counter
        min_p = min(min_p, p)
        max_p = max(max_p, p)
        sum_p += p
        if verbose:
            logger.info("  %20s: %4d (%02.01f %%)", key, global_stats[key], 100 * p)
    logger.info(
        "  %d instance, %d variants seen (C=[%d ; %d], P=[%.03e ; %.03e], E=%.03e, dE=%.03e)",
        global_instance_counter,
        len(global_stats),
        min_p * global_instance_counter,
        max_p * global_instance_counter,
        min_p,
        max_p,
        1.0 / len(global_stats),
        1.0 / len(global_stats) - sum_p / len(global_stats)
    )
