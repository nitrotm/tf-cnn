###########################################
# importflower.py
#
# Import flower segmentation dataset as list of image/mask.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import cv2, io, json, os

from pathlib import Path
from configparser import ConfigParser

import numpy as np

from utils import create_logger, normalize, invert, tobytes, binarymask, image_size

logger = create_logger(__name__)


def run(source, destination):
    """
    source: flower image folder
    destination: target folder
    """

    dst = Path(destination)
    dst.mkdir(exist_ok=True)

    items = dict()
    for child in sorted(Path(source).glob('image_*.jpg')):
        colorfile = child
        maskfile = child.parent / child.name.replace('image_', 'segmim_')

        dstinput = dst / child.name
        dstmask = dst / (child.stem + '.mask')

        if colorfile.exists() and not dstinput.exists():
            logger.info(dstinput)
            os.symlink(colorfile.resolve(), dstinput)
        if not dstmask.exists() and maskfile.exists():
            logger.info(dstmask)
            threshold = 0.1
            mask = normalize(cv2.imread(str(maskfile), cv2.IMREAD_COLOR))
            mask = 255 * np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8) * ((mask[:,:,0] <= (1.0 - threshold)) | (mask[:,:,1] >= threshold) | (mask[:,:,2] >= threshold))
            mask = binarymask(mask, 0.1, 5, 5, 10)
            cv2.imwrite(str(dstmask) + '.png', tobytes(mask))
            (dst / (child.stem + '.mask.png')).replace(dstmask)

        items[child.name] = dict({
            "active": True
        })

    with io.open(dst / "items.json", 'w') as f:
        json.dump(items, f, indent=' ')
