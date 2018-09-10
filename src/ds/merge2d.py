###########################################
# merge2d.py
#
# Merge selected 2d-tagged images into a single folder.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import cv2, json, io, os

from pathlib import Path

import numpy as np

from utils import create_logger, image_size

logger = create_logger(__name__)


def run(sources, destination):
    """
    sources: list of image folders
    destination: target folder
    """

    dst = Path(destination)
    dst.mkdir(parents=True, exist_ok=True)

    items = dict()
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
                srcweight = srcimg.parent / (srcimg.stem + '.weight')

                dstimg = dst / srcimg.name
                dstmask = dst / (srcimg.stem + '.mask')
                dstweight = dst / (srcimg.stem + '.weight')

                if not dstimg.exists():
                    logger.info(dstimg)
                    os.symlink(srcimg.resolve(), dstimg)
                if not dstmask.exists():
                    logger.info(dstmask)
                    if srcmask.exists():
                        dstmask.write_bytes(srcmask.read_bytes())
                    else:
                        width, height = image_size(srcimg)
                        mask = np.zeros([height, width], dtype=np.uint8)
                        cv2.imwrite(str(dstmask) + '.png', mask)
                        (dst / (srcimg.stem + '.mask.png')).replace(dstmask)
                if not dstweight.exists():
                    logger.info(dstweight)
                    if srcweight.exists():
                        dstweight.write_bytes(srcweight.read_bytes())
                    else:
                        width, height = image_size(srcimg)
                        weight = 255 * np.ones([height, width], dtype=np.uint8)
                        cv2.imwrite(str(dstweight) + '.png', weight)
                        (dst / (srcimg.stem + '.weight.png')).replace(dstweight)

                items[srcimg.name] = dict({
                    "active": True
                })

    with io.open(dst / "items.json", 'w') as f:
        json.dump(items, f, indent=' ')
