###########################################
# tfrdist.py
#
# List items in tensorflow dataset.
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


def run(sources, threads, verbose, seed):
    """
    sources: list of tf-record files
    threads: number of concurrent read pipelines
    verbose: verbose output
    """

    reset_tensorflow(seed)

    global_instances = 0
    global_pixels = 0
    global_positives = 0
    global_weighted_positives = 0
    global_negatives = 0
    global_weighted_negatives = 0
    with tf.device('/cpu:0'):
        for source in sources:
            input_fn = make_tfr_input_fn(
                make_tfr_dataset(source, threads=threads, read_rgb=False, read_mask=True, read_weight=True),
                threads=threads,
                batch=10
            )
            instances = 0
            pixels = 0
            positives = 0
            weighted_positives = 0
            negatives = 0
            weighted_negatives = 0
            with tf.Session() as session:
                try:
                    it = input_fn()
                    while True:
                        features, label = session.run(it)
                        z = np.sum(1.0 - label)
                        wz = np.sum((1.0 - label) * features['weight'])
                        nz = np.sum(label)
                        wnz = np.sum(label * features['weight'])
                        instances += label.shape[0]
                        pixels += label.size
                        positives += nz
                        weighted_positives += wnz
                        negatives += z
                        weighted_negatives += wz
                except tf.errors.DataLossError as e:
                    logger.warn(e)
                    pass
                except tf.errors.OutOfRangeError:
                    pass
            global_instances += instances
            global_pixels += pixels
            global_positives += positives
            global_weighted_positives += weighted_positives
            global_negatives += negatives
            global_weighted_negatives += weighted_negatives
            logger.info(
                "%s: %d instances, %d pixels, %d:%d (%.02f:%.02f%%) std:weighted positives, %d:%d (%.02f:%.02f%%) std:weighted negatives",
                source,
                instances,
                pixels,
                positives,
                weighted_positives,
                100 * positives / max(1, positives + negatives),
                100 * weighted_positives / max(1, weighted_positives + weighted_negatives),
                negatives,
                weighted_negatives,
                100 * negatives / max(1, positives + negatives),
                100 * weighted_negatives / max(1, weighted_positives + weighted_negatives)
            )
    logger.info(
        "TOTAL: %d instances, %d pixels, %d:%d (%.02f:%.02f%%) std:weighted positives, %d:%d (%.02f:%.02f%%) std:weighted negatives",
        global_instances,
        global_pixels,
        global_positives,
        global_weighted_positives,
        100 * global_positives / max(1, global_positives + global_negatives),
        100 * global_weighted_positives / max(1, global_weighted_positives + global_weighted_negatives),
        global_negatives,
        global_weighted_negatives,
        100 * global_negatives / max(1, global_positives + global_negatives),
        100 * global_weighted_negatives / max(1, global_weighted_positives + global_weighted_negatives)
    )
