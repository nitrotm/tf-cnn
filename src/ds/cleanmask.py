###########################################
# cleanmask.py
#
# Clean 2d-tagged training/validation masks.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import cv2, json, io, os

from pathlib import Path

import numpy as np

from utils import create_logger, normalize, resizemax, tobytes, invert

logger = create_logger(__name__)


def run(sources, threshold=1e-6, open_size=0, close_size=0, min_area=0, expand_size=0, preview=False):
    """
    sources: list of image folders
    """

    for source in sources:
        path = Path(source)
        if not (path / "items.json").exists():
            continue
        with io.open(path / "items.json") as f:
            for (filename, meta) in json.load(f).items():
                if not meta['active']:
                    continue

                srcimg = path / filename
                srcmask = srcimg.parent / (srcimg.stem + '.mask')
                if not srcmask.exists():
                    continue

                logger.info(srcmask)

                mask = normalize(cv2.imread(str(srcmask), cv2.IMREAD_GRAYSCALE))

                binarymask = 255 * np.ones(mask.shape, dtype=np.uint8) * (mask >= threshold)

                if close_size > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
                    binarymask = cv2.dilate(binarymask, kernel)
                    binarymask = cv2.erode(binarymask, kernel)

                if open_size > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
                    binarymask = cv2.erode(binarymask, kernel)
                    binarymask = cv2.dilate(binarymask, kernel)

                contours_outline = np.zeros(binarymask.shape, np.uint8)
                if min_area > 0:
                    _, contours, hierarchy = cv2.findContours(binarymask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    valid = list()
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area >= min_area:
                            valid.append(contour)
                    binarymask = np.zeros(binarymask.shape, np.uint8)
                    cv2.drawContours(binarymask, valid, -1, 255, cv2.FILLED)
                    cv2.drawContours(contours_outline, valid, -1, 255, 2)

                if expand_size > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_size, expand_size))
                    binarymask = cv2.dilate(binarymask, kernel)

                binarymask = normalize(binarymask)
                contours_outline = normalize(contours_outline)

                if preview:
                    preview_before = cv2.cvtColor(resizemax(mask, 1600), cv2.COLOR_GRAY2BGR)
                    preview_after = cv2.cvtColor(resizemax(binarymask, 1600), cv2.COLOR_GRAY2BGR)
                    preview_contours = resizemax(contours_outline, 1600)
                    preview_after[:,:,0] = preview_contours
                    preview_after[:,:,1] *= invert(preview_contours)
                    preview_after[:,:,2] = preview_before[:,:,0] * invert(preview_contours)

                    zeros1 = np.zeros([preview_before.shape[0], 10, 3], np.float32)
                    ones1 = np.ones([preview_before.shape[0], 10, 3], np.float32)
                    padded1 = np.concatenate([zeros1, preview_before, zeros1, ones1, zeros1, preview_after, zeros1 ], axis=1)
                    zeros2 = np.zeros([10, padded1.shape[1], 3], np.float32)
                    padded2 = np.concatenate([zeros2, padded1, zeros2], axis=0)
                    cv2.imshow('cv2: mask', padded2)

                    key = cv2.waitKey()
                    while key != ord('q') and key != ord('c'):
                        key = cv2.waitKey()
                    if key == ord('q'):
                        break
                else:
                    cv2.imwrite(str(srcmask) + '.png', tobytes(binarymask))
                    (srcimg.parent / (srcimg.stem + '.mask.png')).replace(srcmask)
