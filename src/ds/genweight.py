###########################################
# genweight.py
#
# Generate 2d-tagged training weights.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import cv2, json, io, os

from pathlib import Path

import numpy as np

from utils import create_logger, normalize, resizemax, tobytes, towords, invert

logger = create_logger(__name__)


def run(sources, expand_size=0, pn_ratio=1.0, border="0:1.0", preview=False):
    """
    sources: list of image folders
    """

    border = border.split(':')
    if len(border) > 0:
        border_size = int(border[0])
    else:
        border_size = 0
    if len(border) > 1:
        border_weight = float(border[1])
    else:
        border_weight = 0.5

    for source in sources:
        path = Path(source)
        if not (path / "items.json").exists():
            continue
        with io.open(path / "items.json") as f:
            pn_sum = 0
            pn_count = 0
            for (filename, meta) in json.load(f).items():
                if not meta['active']:
                    continue

                srcimg = path / filename
                srcmask = srcimg.parent / (srcimg.stem + '.mask')
                srcweight = srcimg.parent / (srcimg.stem + '.weight')
                if not srcmask.exists():
                    continue

                mask = normalize(cv2.imread(str(srcmask), cv2.IMREAD_GRAYSCALE))
                pixels = mask.shape[0] * mask.shape[1]

                binarymask = 255 * np.ones(mask.shape, dtype=np.uint8) * (mask > 0)
                if expand_size > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_size, expand_size))
                    expanded_mask = cv2.dilate(binarymask, kernel)
                else:
                    expanded_mask = binarymask
                expanded_mask = normalize(expanded_mask)
                if border_size > 0 and np.sum(binarymask) > 0:
                    _, contours, hierarchy = cv2.findContours(binarymask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    contours_outline = np.zeros(binarymask.shape, np.float32)
                    cv2.drawContours(contours_outline, contours, -1, 1.0, border_size)
                    expanded_mask = normalize(expanded_mask * invert(contours_outline) + contours_outline * border_weight)
                if pn_ratio > 1.0:
                    weight = expanded_mask / pn_ratio + invert(expanded_mask)
                else:
                    weight = expanded_mask + invert(expanded_mask) * pn_ratio

                pn_sum += np.sum((mask > 0) * weight) / max(1e-6, np.sum((mask == 0) * weight))
                pn_count += 1
                logger.info('%s (p/n: %.02f)', srcweight, pn_sum / pn_count)

                if preview:
                    logger.info("range: [%f ; %f], total weight: %.01f", np.amin(weight), np.amax(weight), np.sum(weight))

                    cv2.imshow('cv2: weight', np.concatenate([resizemax(mask, 1000), resizemax(weight, 1000)], axis=1))

                    key = cv2.waitKey()
                    while key != ord('q') and key != ord('c'):
                        key = cv2.waitKey()
                    if key == ord('q'):
                        break
                else:
                    cv2.imwrite(str(srcweight) + '.png', towords(weight))
                    (srcimg.parent / (srcimg.stem + '.weight.png')).replace(srcweight)
