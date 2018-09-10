###########################################
# preview.py
#
# Preview tensorflow dataset content.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import cv2

import numpy as np

import tensorflow as tf

from utils import reset_tensorflow, create_logger, resizemax, tobytes, tofloats, mix

from ds.reader import make_tfr_dataset

logger = create_logger(__name__)


def run(sources):
    """
    sources: list of tf-record files
    """

    key = None
    for source in sources:
        logger.info(source)
        dataset = make_tfr_dataset(source)

        reset_tensorflow()
        step = 0
        with tf.Session() as session:
            with tf.device('/cpu:0'):
                it = dataset().make_one_shot_iterator().get_next()
                while key != ord('q'):
                    item = session.run(it)
                    step += 1

                    name = str(item['name'], 'utf-8')
                    rgb = cv2.cvtColor(item['rgb'], cv2.COLOR_RGB2BGR)
                    mask = item['mask']
                    weight = item['weight']
                    result = resizemax(
                        np.concatenate(
                            [
                                rgb,
                                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                                tobytes(cv2.cvtColor(weight, cv2.COLOR_GRAY2BGR) / 255)
                                # tobytes(mix(tofloats(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), 0.5 + 0.5 * tofloats(rgb))),
                                # tobytes(mix(cv2.cvtColor(weight, cv2.COLOR_GRAY2BGR).astype(np.float32) / 65535.0, 0.5 + 0.5 * tofloats(rgb)))
                            ],
                            axis=0
                        ),
                        1024
                    )
                    cv2.putText(result, '%d' % step, (result.shape[1]-50, result.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                    cv2.putText(result, name, (5, result.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                    cv2.imshow('cv2: preview', result)

                    key = cv2.waitKey()
                    while key != ord('q') and key != ord('c'):
                        key = cv2.waitKey()
