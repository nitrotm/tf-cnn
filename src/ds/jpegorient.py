###########################################
# jpegorient.py
#
# Automatically rotate jpeg image based on exif orientation.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import cv2, json, io, os

from pathlib import Path

import numpy as np

from utils import create_logger, find_images, image_orientation, image_fix_orientation

logger = create_logger(__name__)


def run(sources):
    """
    sources: list of image folders
    """

    files = set()
    for source in sources:
        path = Path(source)
        for image in find_images(source, ['.jpg', '.jpeg']):
            filename = path / image
            if image_orientation(filename) == 1:
                continue
            files.add(filename.resolve())

    for filename in files:
        logger.info(filename.name)
        image_fix_orientation(filename)
