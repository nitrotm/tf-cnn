###########################################
# tfrorder.py
#
# Check ordering in tensorflow dataset.
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

    items = list()
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
                with tf.Session() as session:
                    try:
                        it = input_fn()
                        batch_counter = 0
                        while batch_counter < steps:
                            features, label = session.run(it)
                            names = features['name']
                            variants = features['variant']
                            for i in range(names.shape[0]):
                                key = '%s-%s' % (str(names[i], 'utf-8'), str(variants[i], 'utf-8'))
                                items.append(key)
                            batch_counter += 1
                    except tf.errors.DataLossError as e:
                        logger.warn(e)
                        pass
                    except tf.errors.OutOfRangeError:
                        pass

    logger.info("ORDER: %d items", len(items))
    for i in range(len(items)):
        print("%05d. %s" % (i, items[i]))
