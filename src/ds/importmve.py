###########################################
# importmve.py
#
# Import mve scene as list of image/mask/depth.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import cv2, os

from pathlib import Path
from configparser import ConfigParser

import numpy as np

from utils import create_logger, normalize, invert, tobytes, binarymask, image_size

logger = create_logger(__name__)


def run(sources, destination):
    """
    sources: list of mve scene folders (name:path)
    destination: target folder
    """

    Path(destination).mkdir(exist_ok=True)

    for source in sources:
        i = source.index(':')
        setname = source[0:i]
        path = source[i+1:]
        dst = Path(destination) / setname
        dst.mkdir(exist_ok=True)
        children = sorted((Path(path) / "output" / "views").iterdir())
        for child in children:
            metafile = child / "meta.ini"
            meta = ConfigParser()
            meta.read(metafile)
            if 'view' not in meta or 'camera' not in meta or float(meta['camera']['focal_length']) <= 0:
                continue
            name = meta['view']['name']

            orgfile = child / "original.jpg"
            colorfile = child / "undistorted.png"
            maskfile = child / "mask.png"
            depthfile = child / "mask-depth.png"
            if not metafile.exists() or not (orgfile.exists() or colorfile.exists()):
                continue

            dstinput = dst / (name + '.png')
            dstmask = dst / (name + '.mask')
            dstweight = dst / (name + '.weight')

            if colorfile.exists() and not dstinput.exists():
                logger.info(dstinput)
                os.symlink(colorfile.resolve(), dstinput)
            elif orgfile.exists() and not dstinput.exists():
                logger.info(dstinput)
                os.symlink(orgfile.resolve(), dstinput)
            if not dstmask.exists() and maskfile.exists():
                logger.info(dstmask)
                mask = binarymask(normalize(cv2.imread(str(maskfile), cv2.IMREAD_ANYDEPTH)), 0.1, 5, 5, 10)
                cv2.imwrite(str(dstmask) + '.png', tobytes(mask))
                (dst / (name + '.mask.png')).replace(dstmask)
            if not dstweight.exists() and depthfile.exists():
                logger.info(dstweight)
                weight = invert(normalize(cv2.imread(str(depthfile), cv2.IMREAD_ANYDEPTH)))
                cv2.imwrite(str(dstweight) + '.png', tobytes(weight))
                (dst / (name + '.weight.png')).replace(dstweight)
