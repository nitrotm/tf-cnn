###########################################
# reader.py
#
# Utilities to decode tf-record datasets.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import time

import tensorflow as tf

from pathlib import Path

from utils import create_logger, find_images

logger = create_logger(__name__)


def cached_dataset(dataset, filename):
    """
    Cache dataset in a memory buffer.

    dataset: input dataset

    return dataset in memory
    """

    logger.info("Caching dataset %s...", filename)
    records = list()
    last_time = time.time()
    with tf.Session() as session:
        it = dataset.make_one_shot_iterator().get_next()
        try:
            while True:
                records.append(session.run(it))
                if time.time() - last_time > 60:
                    last_time = time.time()
                    logger.info("  %d instances loaded...", len(records))
        except tf.errors.DataLossError as e:
            logger.warn(e)
            pass
        except tf.errors.OutOfRangeError:
            pass
    logger.info("Dataset cached (%d instances).", len(records))

    def gen():
        for record in records:
            yield record
    return tf.data.Dataset.from_generator(gen, dataset.output_types, dataset.output_shapes)


def make_tfr_dataset(filename, threads=1, buffer_size=8192, cache=False, read_rgb=True, read_mask=True, read_weight=True):
    """
    Open a tf-record dataset reader.

    filename: tf-record dataset filename
    threads: number of read/decode parallel calls
    buffer_size: read buffer size (bytes)
    cache: cache dataset in ram
    read_rgb: read color images
    read_mask: read mask images
    read_weight: read weight images

    each record is made of:
        name: string
        width: int64
        height: int64
        rgb: string (uint8 image)
        mask: string (uint8 image)
        weight: string (uint8 image)

    return a dataset generator
    """

    def decode_fn(data):
        def decode_rgb_image(data, width, height):
            return tf.reshape(tf.decode_raw(data, tf.uint8), tf.stack([height, width, 3]))

        def decode_gray_image(data, width, height):
            return tf.reshape(tf.decode_raw(data, tf.uint8), tf.stack([height, width, 1]))

        def decode_gray_image16(data, width, height):
            return tf.reshape(tf.decode_raw(data, tf.uint16), tf.stack([height, width, 1]))

        features = dict()
        with tf.name_scope('decode_fn'):
            sample = tf.parse_single_example(
                data,
                features={
                    'name': tf.FixedLenFeature([], tf.string),
                    'variant': tf.FixedLenFeature([], tf.string),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'height': tf.FixedLenFeature([], tf.int64),
                    'rgb': tf.FixedLenFeature([], tf.string),
                    'mask': tf.FixedLenFeature([], tf.string),
                    'weight': tf.FixedLenFeature([], tf.string),
                }
            )
            features['name'] = sample['name']
            features['variant'] = sample['variant']
            if read_rgb:
                features['rgb'] = decode_rgb_image(sample['rgb'], sample['width'], sample['height'])
            else:
                features['rgb'] = tf.zeros(tf.stack([sample['height'], sample['width'], 3]), tf.uint8)
            if read_mask:
                features['mask'] = decode_gray_image(sample['mask'], sample['width'], sample['height'])
            else:
                features['mask'] = tf.zeros(tf.stack([sample['height'], sample['width'], 1]), tf.uint8)
            if read_weight:
                features['weight'] = decode_gray_image16(sample['weight'], sample['width'], sample['height'])
            else:
                features['weight'] = tf.ones(tf.stack([sample['height'], sample['width'], 1]), tf.uint16)
        return features

    if filename.endswith('.zlib'):
        compression = 'ZLIB'
    elif filename.endswith('.gz'):
        compression = 'GZIP'
    else:
        compression = ''
    dataset = tf.data.TFRecordDataset(filename, compression_type=compression, buffer_size=buffer_size, num_parallel_reads=threads)
    dataset = dataset.map(decode_fn, num_parallel_calls=threads)
    if cache:
        dataset = cached_dataset(dataset, filename)

    def gen_fn():
        return dataset
    return gen_fn


def make_tfr_input_fn(dataset_fn, threads=1, offset=0, limit=0, shuffle=0, prefetch=0, batch=10, repeat=0, label='mask', standardize_rgb=True, seed=None):
    """
    A generator to create an input function for tensorflow estimators.

    dataset_fn: dataset generator returned with make_tfr_dataset()
    threads: number of read/decode parallel calls
    offset: read offset
    limit: read limit (0 is all)
    shuffle: shuffle buffer size
    prefetch: prefetch size
    batch: batch size
    repeat: number of time to repeat (-1 is infinite)
    label: feature name used as label
    standardize_rgb: standardize rgb images

    return an input function suitable for estimators
    """

    def standardize_rgb_image(features):
        if 'rgb' in features:
            features['rgb'] = tf.image.per_image_standardization(features['rgb'])
        return features

    def normalize_images(features):
        if 'rgb' in features:
            features['rgb'] = tf.image.convert_image_dtype(features['rgb'], tf.float32)
        if 'mask' in features:
            features['mask'] = tf.image.convert_image_dtype(features['mask'], tf.float32)
        if 'weight' in features:
            features['weight'] = tf.image.convert_image_dtype(features['weight'], tf.float32)
        return features

    def input_fn():
        with tf.name_scope('input_fn'):
            dataset = dataset_fn()
            if repeat != 0:
                dataset = dataset.repeat(repeat)
            if offset > 0:
                dataset = dataset.skip(offset)
            if limit > 0:
                dataset = dataset.take(limit)
            if shuffle > 0:
                dataset = dataset.shuffle(shuffle, seed=seed, reshuffle_each_iteration=True)
            if standardize_rgb:
                dataset = dataset.map(standardize_rgb_image, num_parallel_calls=threads)
            dataset = dataset.map(normalize_images, num_parallel_calls=threads)
            dataset = dataset.batch(batch)
            dataset = dataset.map(lambda features: (features, features[label]), num_parallel_calls=threads)
            if prefetch > 0:
                dataset = dataset.prefetch(buffer_size=prefetch)
            return dataset.make_one_shot_iterator().get_next()
    return input_fn


def make_predict_dataset(path, threads=1, cache=False):
    """
    Create a dataset reader from existing path.

    path: raw image folder (jpegs or pngs)
    threads: number of read/decode parallel calls
    cache: cache dataset in ram

    return a dataset of decoded samples
    """

    def decode_fn(filename):
        with tf.name_scope('decode_fn'):
            return tf.image.convert_image_dtype(
                tf.image.decode_image(tf.read_file(filename), channels=3),
                tf.float32
            )

    def gen():
        for filename in find_images(path):
            yield str(Path(path) / filename)

    dataset = tf.data.Dataset.from_generator(gen, tf.string, tf.TensorShape([]))
    dataset = dataset.map(decode_fn, num_parallel_calls=threads)
    if cache:
        dataset = cached_dataset(dataset, path)

    def gen_fn():
        return dataset
    return gen_fn


def make_predict_input_fn(dataset_fn, size, threads=1, offset=0, limit=0, prefetch=0, batch=10, repeat=0, standardize_rgb=True, seed=None):
    """
    A generator to create an input function for tensorflow estimators (prediction only).

    dataset_fn: dataset generator returned with make_predict_dataset()
    size: target image size (square)
    threads: number of read/decode parallel calls
    offset: read offset
    limit: read limit (0 is all)
    prefetch: prefetch size
    batch: batch size
    repeat: number of time to repeat (-1 is infinite)
    standardize_rgb: standardize rgb images

    return a dataset suitable for estimators prediction
    """

    def random_crop_fn(src):
        width = tf.to_double(tf.shape(src)[1])
        height = tf.to_double(tf.shape(src)[0])
        ratio = width / height

        small = tf.minimum(width, height)
        scale = size * 3 / small
        width = width * scale
        height = height * scale

        width, height = tf.cond(
            width < size,
            lambda: (tf.to_double(size), size / ratio),
            lambda: (width, height)
        )
        width, height = tf.cond(
            height < size,
            lambda: (size * ratio, tf.to_double(size)),
            lambda: (width, height)
        )
        width = tf.to_int32(width)
        height = tf.to_int32(height)

        result = tf.random_crop(tf.image.resize_bilinear([src], [height, width])[0], [size, size, 3])
        if standardize_rgb:
            return tf.image.per_image_standardization(result)
        return result

    def input_fn():
        with tf.name_scope('input_fn'):
            dataset = dataset_fn()
            if repeat != 0:
                dataset = dataset.repeat(repeat)
            if offset > 0:
                dataset = dataset.skip(offset)
            if limit > 0:
                dataset = dataset.take(limit)
            dataset = dataset.map(random_crop_fn, num_parallel_calls=threads)
            dataset = dataset.batch(batch)
            dataset = dataset.map(lambda rgb: { 'rgb': rgb }, num_parallel_calls=threads)
            if prefetch > 0:
                dataset = dataset.prefetch(buffer_size=prefetch)
            return dataset.make_one_shot_iterator().get_next()
    return input_fn
