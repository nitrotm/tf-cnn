###########################################
# utils.py
#
# Various image processing utilities
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import cv2, logging, math, piexif, sys, random

import numpy as np

import tensorflow as tf

from pathlib import Path

from PIL import Image


def reset_tensorflow(seed=75437):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    random.seed(seed + 99991)


FILE_HANDLERS = list()
ACTIVE_LOGGERS = list()


def create_logger(name, level=logging.DEBUG):
    global FILE_HANDLERS, ACTIVE_LOGGERS

    if sys.flags.interactive:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter('{asctime:s} {levelname[0]:s} {name:s}: {message:s}', datefmt='%Y-%m-%d %H:%M:%S', style='{'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    for handler in FILE_HANDLERS:
        logger.addHandler(handler)
    ACTIVE_LOGGERS.append(logger)
    return logger


def set_logger_file(filename, level=logging.DEBUG):
    global FILE_HANDLERS, ACTIVE_LOGGERS

    handler = logging.FileHandler(filename)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter('{asctime:s} {levelname[0]:s} {name:s}: {message:s}', datefmt='%Y-%m-%d %H:%M:%S', style='{'))
    FILE_HANDLERS.append(handler)
    for logger in ACTIVE_LOGGERS:
        logger.addHandler(handler)


def find_images(rootpath, exts=['.png', '.jpg', '.jpeg'], path=None):
    if not path:
        path = Path(rootpath)
    result = list()
    for child in path.iterdir():
        if child.is_dir():
            result = result + find_images(rootpath, exts, child)
        if child.is_file():
            filename = child.name.lower()
            for ext in exts:
                if not filename.endswith(ext):
                    continue
                result.append(str(child.relative_to(rootpath)))
                break
    return sorted(result)


def image_channels(src):
    image = Image.open(src)
    if image.mode == '1' or image.mode == 'L' or image.mode == 'P' or image.mode == 'I' or image.mode == 'F':
        return 1
    if image.mode == 'LA':
        return 2
    if image.mode == 'RGB' or image.mode == 'YCbCr' or image.mode == 'LAB' or image.mode == 'HSV':
        return 3
    if image.mode == 'RGBA' or image.mode == 'CMYK':
        return 4
    return 0


def image_orientation(src):
    image = Image.open(src)
    if 'exif' in image.info:
        exif = piexif.load(image.info["exif"])
        if '0th' in exif and piexif.ImageIFD.Orientation in exif['0th']:
            return exif['0th'][piexif.ImageIFD.Orientation]
    return 1


def image_size(src):
    image = Image.open(src)
    orientation = image_orientation(src)
    if orientation >= 5 and orientation <= 8:
        return (image.height, image.width)
    return (image.width, image.height)


def image_fix_orientation(src):
    image = Image.open(src)
    if 'exif' in image.info:
        exif = piexif.load(image.info["exif"])
        if '0th' in exif and piexif.ImageIFD.Orientation in exif['0th']:
            orientation = exif['0th'][piexif.ImageIFD.Orientation]
            if orientation == 2:
                image = image.tranpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                image = image.rotate(180)
            elif orientation == 4:
                image = image.rotate(180).tranpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 5:
                image = image.rotate(-90, expand=True).tranpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                image = image.rotate(-90, expand=True)
            elif orientation == 7:
                image = image.rotate(90, expand=True).tranpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
            else:
                return
            exif['0th'].pop(piexif.ImageIFD.Orientation)
            image.save(src, exif=piexif.dump(exif))


def tf_normalize(src, eps=1e-6):
    a = tf.reduce_min(src)
    b = tf.reduce_max(src)
    return (src - a) / tf.maximum(b - a, eps)


def normalize_ab(img, a, b):
    """
    Normalize image range from [a;b] to [0;1].
    """

    img = img.astype(np.float32)
    if a == b:
        return 0 * img
    return np.clip((img - float(a)) / float(b - a), 0.0, 1.0)


def normalize(img):
    """
    Normalize image in range [0;1].
    """

    img = img.astype(np.float32)
    a = np.amin(img)
    b = np.amax(img)
    return normalize_ab(img, a, b)


def invert(img):
    """
    Invert normalized image (black <-> white).
    """

    return np.ones(img.shape, dtype=np.float32) - img


def mix(a, b):
    """
    Multiply two images.
    """
    return np.multiply(a, b)


def tobytes(img):
    """
    Convert normalized image to uint8.
    """

    return (255.0 * img).astype(np.uint8)


def towords(img):
    """
    Convert normalized image to uint16.
    """

    return (65535.0 * img).astype(np.uint16)


def tofloats(img):
    """
    Convert uint8 image to float.
    """

    return img.astype(np.float32) / 255.0


def normalize8(img, invertChannels=False):
    """
    Perform normalization with inversion (opt).
    """

    img = normalize(img)
    if invertChannels:
        img = invert(img)
    return tobytes(img)


def binarymask(img, t=0.1, c=5, b=3, it=1):
    """
    Clean binary mask from point-cloud projection.
    """

    mask = 255 * np.ones(img.shape, dtype=np.uint8) * (img >= t)
    for i in range(it):
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)))
    mask = cv2.blur(mask, (b, b))
    return mask.astype(np.float32) / 255.0


def resizemax(img, maxsize):
    """
    Resize image to fit maxsize constraint.
    """

    scale = maxsize / max(img.shape)
    return cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def flipxy(img, x=False, y=False):
    """
    Flip image.
    """

    if x and y:
        return cv2.flip(img, -1)
    elif x and not y:
        return cv2.flip(img, 1)
    elif not x and y:
        return cv2.flip(img, 0)
    return img


def rotatedRectWithMaxArea(w, h, angle):
    """
    Find largest rectangle that fit after image rotation.

    Solution from coproc (https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders)
    """

    angle = math.radians(angle)
    side_long, side_short = (w, h) if w >= h else (h, w)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if w >= h else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    return math.floor(wr), math.floor(hr)


def rotatecrop(img, angle):
    """
    Rotate and crop to usable area.
    """

    width = img.shape[1]
    height = img.shape[0]
    m = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    img = cv2.warpAffine(img, M=m, dsize=(img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    width2, height2 = rotatedRectWithMaxArea(width, height, angle)
    return cv2.getRectSubPix(img, (width2, height2), (width / 2.0, height / 2.0))


def rotatepreview(img, angle):
    """
    Rotate and show crop area.
    """

    width = img.shape[1]
    height = img.shape[0]
    m = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    img = cv2.warpAffine(img, M=m, dsize=(img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    width2, height2 = rotatedRectWithMaxArea(width, height, angle)
    x = round((width - width2) / 2.0)
    y = round((height - height2) / 2.0)
    return cv2.rectangle(img, (x, y), (x+width2, y+height2), color=(0, 255, 255), thickness=10)


def cropsquare(img):
    """
    Crop image to square ratio.
    """

    width = img.shape[1]
    height = img.shape[0]
    size = min(width, height)
    return cv2.getRectSubPix(img, (size, size), (width/2.0, height/2.0))


def gaussian_kernel(radius):
    def gaussian2d(x, y, var=1):
        return math.exp(-(x ** 2 + y ** 2) / (2.0 * var))

    kernel = np.array(
        [ [ gaussian2d(x, y) for x in range(-radius, radius+1) ] for y in range(-radius, radius+1) ],
        np.float32
    )
    kernel /= np.sum(kernel)
    return np.repeat(np.reshape(kernel, [radius*2+1, radius*2+1, 1, 1]), 1, axis=2)


def f1_score(p, r):
  if p == 0 and r == 0:
    return 0
  return 2 * (p * r) / (p + r)
