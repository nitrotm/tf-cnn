###########################################
# bake.py
#
# Generate tensorflow datasets from tagged images.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import cv2, json, io, math, os, random, time

from pathlib import Path

import numpy as np
import tensorflow as tf

from utils import reset_tensorflow, create_logger, image_channels, image_size, image_orientation, gaussian_kernel

logger = create_logger(__name__)


def build_scale_graph(image, scale, size):
    """
    Tensorflow graph to scale image by given factor while
    keeping minimum size constraint.

    image: input image
    scale: scale factor
    size: minimum final width/heigth

    return scaled image
    """

    width = tf.shape(image)[1]
    width2 = tf.to_double(width) * scale

    height = tf.shape(image)[0]
    height2 = tf.to_double(height) * scale

    ratio = tf.to_double(width) / tf.to_double(height)
    width2, height2 = tf.cond(
        width2 < size,
        lambda: (tf.to_double(size), size / ratio),
        lambda: (width2, height2)
    )
    width2, height2 = tf.cond(
        height2 < size,
        lambda: (size * ratio, tf.to_double(size)),
        lambda: (width2, height2)
    )
    width2 = tf.to_int32(width2)
    height2 = tf.to_int32(height2)
    return tf.image.resize_bilinear([image], [height2, width2])[0]

def build_flip_graph(image, flip):
    """
    Tensorflow graph to flip image.

    image: input image
    flip: flip mode (none, x, y, xy)

    return flipped image
    """

    if flip == 'none':
        return image
    if flip == 'x':
        return tf.image.flip_left_right(image)
    if flip == 'y':
        return tf.image.flip_up_down(image)
    if flip == 'xy':
        return tf.image.flip_up_down(tf.image.flip_left_right(image))
    raise Exception("Invalid flip mode (%s)" % flip)

def build_rotate_graph(image, angle):
    """
    Tensorflow graph to rotate image and crop to usable area.

    image: input image
    angle: rotation angle in degree

    return rotated and cropped image
    """

    if angle == 0:
        return image

    angle = math.radians(angle)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    cos_2a = cos_a * cos_a - sin_a * sin_a
    is_degenerate = abs(sin_a - cos_a) < 1e-10

    width = tf.to_double(tf.shape(image)[1])
    height = tf.to_double(tf.shape(image)[0])
    side_long = tf.cond(
        width >= height,
        lambda: width,
        lambda: height
    )
    side_short = tf.cond(
        width >= height,
        lambda: height,
        lambda: width
    )

    width2 = tf.minimum(
        width,
        tf.floor(
            tf.cond(
                tf.logical_or(
                    side_short <= 2.0 * sin_a * cos_a * side_long,
                    is_degenerate
                ),
                lambda: tf.cond(
                    width >= height,
                    lambda: 0.5 * side_short / sin_a,
                    lambda: 0.5 * side_short / cos_a
                ),
                lambda: (width * cos_a - height * sin_a) / cos_2a
            )
        )
    )
    height2 = tf.minimum(
        height,
        tf.floor(
            tf.cond(
                tf.logical_or(
                    side_short <= 2.0 * sin_a * cos_a * side_long,
                    is_degenerate
                ),
                lambda: tf.cond(
                    width >= height,
                    lambda: 0.5 * side_short / cos_a,
                    lambda: 0.5 * side_short / sin_a
                ),
                lambda: (height * cos_a - width * sin_a) / cos_2a
            )
        )
    )
    return tf.image.crop_to_bounding_box(
        tf.contrib.image.rotate(image, angle, "BILINEAR"),
        tf.to_int32(tf.floor((height - height2) / 2)),
        tf.to_int32(tf.floor((width - width2) / 2)),
        tf.to_int32(height2),
        tf.to_int32(width2)
    )

def build_variation_graphs(image, size, count, brightness, contrast, gaussian_noise, uniform_noise):
    """
    Tensorflow graph to create image variations (random crops/brightness/contrast/noise).

    image: input image
    size: target width/height
    count: number of variation to create
    brightness: random brightness delta
    contrast: random contrast delta
    gaussian_noise: gaussian noise standard deviation
    uniform_noise: uniform noise magnitude

    return list of image variations
    """

    items = list()
    for crop in range(count):
        with tf.name_scope("crop%d" % crop):
            image2 = tf.random_crop(image, [size, size, 5])
            rgb = image2[:,:,0:3]
            if brightness > 0:
                rgb = tf.image.random_brightness(rgb, brightness)
            if contrast > 0:
                rgb = tf.image.random_contrast(rgb, 1.0 - contrast, 1.0 + contrast)
            if gaussian_noise > 0:
                rgb = rgb + tf.random_normal([size, size, 3], 0.0, gaussian_noise)
            if uniform_noise > 0:
                rgb = rgb + tf.random_uniform([size, size, 3], -uniform_noise, uniform_noise)
            mask = image2[:,:,3:4]
            weight = image2[:,:,4:5]
            items.append({
                "width": tf.constant(size, tf.int32),
                "height": tf.constant(size, tf.int32),
                "crop": tf.constant(crop, tf.int32),
                "rgb": tf.reshape(tf.image.convert_image_dtype(rgb, tf.uint8, True), [-1]),
                "mask": tf.reshape(tf.image.convert_image_dtype(mask, tf.uint8, True), [-1]),
                "weight": tf.reshape(tf.image.convert_image_dtype(weight, tf.uint16, True), [-1]),
            })
    return items

def build_graphs(batch, size, blur_radius, blur_scale, center_crop, scale, flip, rotation, crops, brightness, contrast, gaussian_noise, uniform_noise):
    """
    Tensorflow graph to load and transform images.

    batch: number of images to process in batch
    size: target width/height
    blur_radius: mask/weight blur radius
    blur_scale: gaussian blur scale factor for mask/weight images
    center_crop: source images center crop percentage
    scale: scale factor
    flip: flip mode
    rotation: rotation in degree
    crops: number of variation to create
    brightness: random brightness delta
    contrast: random contrast delta
    gaussian_noise: gaussian noise standard deviation
    uniform_noise: uniform noise magnitude

    return tensorflow graph pipeline
    """

    items = list()
    for i in range(batch):
        with tf.name_scope("batch%d" % i):
            # inputs
            name = tf.placeholder(tf.string, name="name")
            width = tf.placeholder(tf.int32, name="width")
            height = tf.placeholder(tf.int32, name="height")
            orientation = tf.placeholder(tf.int32, name="orientation")
            rgbpath = tf.placeholder(tf.string, name="rgbpath")
            maskpath = tf.placeholder(tf.string, name="maskpath")
            weightpath = tf.placeholder(tf.string, name="weightpath")

            # read images
            rgb    = tf.image.decode_image(tf.read_file(rgbpath), channels=3, name="decode_rgb")
            mask   = tf.image.decode_image(tf.read_file(maskpath), channels=1, name="decode_mask")
            weight = tf.image.decode_image(tf.read_file(weightpath), channels=1, name="decode_weight")
            rgb    = tf.image.convert_image_dtype(rgb, tf.float32, name="convert_rgb")
            mask   = tf.image.convert_image_dtype(mask, tf.float32, name="convert_mask")
            weight = tf.image.convert_image_dtype(weight, tf.float32, name="convert_weight")
            rgb    = tf.reshape(rgb, [height, width, 3], name="reshape_rgb")
            mask   = tf.reshape(mask, [height, width, 1], name="reshape_mask")
            weight = tf.reshape(weight, [height, width, 1], name="reshape_weight")

            # blur mask/weight
            if blur_radius > 0:
                with tf.name_scope("blur"):
                    kernel = tf.constant(gaussian_kernel(blur_radius), name="kernel")
                    blur_in = tf.stack([mask, weight], name="stack")
                    if blur_scale < 1.0:
                        height2 = tf.to_double(height) * blur_scale
                        width2 = tf.to_double(width) * blur_scale
                        blur_in = tf.image.resize_bilinear(blur_in, tf.to_int32([height2, width2]), name="downscale")
                    blur_out = tf.nn.conv2d(input=blur_in, filter=kernel, strides=[1, 1, 1, 1], padding="SAME", name="conv2d")
                    if blur_scale < 1.0:
                        blur_out = tf.image.resize_bilinear(blur_out, [height, width], name="upscale")
                    mask, weight = tf.unstack(blur_out, name="unstack")

            # normalize mask/weight
            mask /= tf.maximum(1.0, tf.reduce_max(mask))
            weight /= tf.maximum(1.0, tf.reduce_max(weight))

            # preprocess image
            image = tf.concat([rgb, mask, weight], axis=2, name="concat")
            image = tf.image.central_crop(image, center_crop)
            with tf.name_scope("flip"):
                image = build_flip_graph(image, flip)
            with tf.name_scope("rotatation"):
                image = build_rotate_graph(image, rotation)
            with tf.name_scope("scale"):
                image = build_scale_graph(image, scale, size)

            # generate variations
            with tf.name_scope("variations"):
                variations = build_variation_graphs(
                    image,
                    size,
                    crops,
                    brightness,
                    contrast,
                    gaussian_noise,
                    uniform_noise
                )
            items.append({
                "name": name,
                "width": width,
                "height": height,
                "orientation": orientation,
                "transform": tf.constant("%.02f_%s_%.01f" % (scale, flip, rotation), tf.string),
                "rgbpath": rgbpath,
                "maskpath": maskpath,
                "weightpath": weightpath,
                "variations": variations,
            })
    return items

def make_int64_feature(value):
    """
    Build tensorflow record file integer feature.

    value: feature value
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def make_bytes_feature(value):
    """
    Build tensorflow record file byte array feature.

    value: feature value
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_abytes_feature(values):
    """
    Build tensorflow record file array of byte array feature.

    values: feature values
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def write_records(writer, results):
    """
    Write tensorflow sample to tf-record file.

    writer: tensorflow writer
    results: evaluated tensors

    each record is serialized as:
        name: string
        width: int64
        height: int64
        rgb: string (uint8 image)
        mask: string (uint8 image)
        weight: string (uint8 image)
    """
    for result in results:
        name = str(result['name'])
        transform = str(result['transform'], 'utf-8')
        for variation in result['variations']:
            width   = variation['width']
            height  = variation['height']
            crop    = variation['crop']
            rgb     = variation['rgb']
            mask    = variation['mask']
            weight  = variation['weight']
            variant = '%s_%d' % (transform, crop)
            writer.write(
                tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'name':    make_bytes_feature(bytes(name, 'utf-8')),
                            'variant': make_bytes_feature(bytes(variant, 'utf-8')),
                            'width':   make_int64_feature(width),
                            'height':  make_int64_feature(height),
                            'rgb':     make_bytes_feature(rgb.tobytes()),
                            'mask':    make_bytes_feature(mask.tobytes()),
                            'weight':  make_bytes_feature(weight.tobytes()),
                        }
                    )
                ).SerializeToString()
            )


def run(device, name, sources, compression, batch, test_ratio, eval_ratio, size, blur_radius, blur_scale, center_crop, scales, flips, rotations, crops, brightness, contrast, gaussian_noise, uniform_noise, seed):
    """
    device: tensorflow device
    name: tf-record prefix
    sources: list of input image folders
    compression: tf-record compression
    batch: image pipeline size
    test_ratio: percentage of final test instances in the total set
    eval_ratio: percentage of evaluation instances in the training set
    size: target image size (width and height)
    blur_radius: gaussian blur radius for mask/weight images
    blur_scale: gaussian blur scale factor for mask/weight images
    center_crop: source images center crop percentage
    scales: list of scale factors
    flips: list of flip modes
    rotations: list of rotations
    crops: number of crops/variations to create for each configuration
    brightness: random brightness factor
    contrast: random contrast factor
    brightness: random brightness factor
    gaussian_noise: gaussian noise standard deviation
    uniform_noise: uniform noise magnitude
    """

    if compression == 'none':
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        ext = 'tfr'
    elif compression == 'zlib':
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        ext = 'tfr.zlib'
    elif compression == 'gzip':
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        ext = 'tfr.gz'
    else:
        raise Exception("Unsupported compression format (%s)" % compression)
    batch = max(1, batch)
    test_ratio = max(0.0, min(1.0, test_ratio))
    eval_ratio = max(0.0, min(1.0, eval_ratio))
    size = int(max(1, size))
    blur_radius = int(max(0, min(size/2, blur_radius)))
    center_crop = max(0.0, min(1.0, center_crop))
    blur_scale = max(0.0, min(1.0, blur_scale))
    scales = [ max(0.0, min(1.0, scale)) for scale in scales ]
    rotations = [ max(-180, min(180, rotation)) for rotation in rotations ]
    crops = int(max(1, crops))
    brightness = max(0.0, min(1.0, brightness))
    contrast = max(0.0, min(1.0, contrast))
    gaussian_noise = max(0.0, gaussian_noise)
    uniform_noise = max(0.0, min(1.0, uniform_noise))

    # find samples
    items = list()
    logger.info('Finding input files...')
    for source in sources:
        path = Path(source)
        bakemeta = path / "bake.json"
        if not bakemeta.exists():
            meta_items = list()
            listing = path / "items.json"
            for (filename, meta) in json.loads(listing.read_text()).items():
                if not meta['active']:
                    continue

                rgb = path / filename
                mask = rgb.parent / (rgb.stem + '.mask')
                weight = rgb.parent / (rgb.stem + '.weight')

                width, height = image_size(rgb)
                assert width >= size and height >= size
                assert image_channels(rgb) == 3

                if not mask.exists():
                    imask = np.zeros([height, width], dtype=np.uint8)
                    cv2.imwrite(str(mask) + '.png', imask)
                    Path(str(mask) + '.png').replace(mask)
                assert (width, height) == image_size(mask)
                assert image_channels(mask) == 1

                if not weight.exists():
                    iweight = 255 * np.ones([height, width], dtype=np.uint8)
                    cv2.imwrite(str(weight) + '.png', iweight)
                    Path(str(weight) + '.png').replace(weight)
                assert (width, height) == image_size(weight)
                assert image_channels(weight) == 1

                meta_items.append({
                    'name': rgb.stem,
                    'orientation': image_orientation(rgb),
                    'width': width,
                    'height': height,
                    'rgbpath': str(rgb),
                    'maskpath': str(mask),
                    'weightpath': str(weight),
                })
            random.shuffle(meta_items)
            bakemeta.write_text(json.dumps(meta_items, indent=' '))
        items += json.loads(bakemeta.read_text())

    # select test set
    i = int(len(items) * (1.0 - test_ratio))
    remaining_items = items[0:i]
    test_items = items[i:]

    # select train/eval sets
    i = int(len(remaining_items) * (1.0 - eval_ratio))
    train_items = remaining_items[0:i]
    eval_items = remaining_items[i:]

    # save partitioning
    if len(train_items) > 0:
        with io.open('%s-%dx%d-train.json' % (name, size, size), "w") as f:
            json.dump(train_items, f, indent=' ')
    if len(eval_items) > 0:
        with io.open('%s-%dx%d-eval.json' % (name, size, size), "w") as f:
            json.dump(eval_items, f, indent=' ')
    if len(test_items) > 0:
        with io.open('%s-%dx%d-test.json' % (name, size, size), "w") as f:
            json.dump(test_items, f, indent=' ')

    # generate tf-records
    def bake_records(session, writer, inputs, graphs):
        feeds = dict()
        for i in range(len(inputs)):
            if i > 0 and i % batch == 0:
                logger.debug('%.02f %%', 100.0 * i / len(inputs))
                write_records(writer, session.run(graphs, feeds))
            k = i % batch
            for (key, value) in inputs[i].items():
                feeds[graphs[k][key]] = value
        k = len(inputs) % batch
        if k == 0:
            k = batch
        logger.debug('%.02f %%', 100)
        write_records(writer, session.run(graphs[0:k], feeds))

    # show generation statistics
    start_time = time.time()
    n = len(scales) * len(flips) * len(rotations)
    m = crops

    logger.info(
        'Generating %d variants (%d train, %d eval, %d test)...',
        n * m * len(items),
        n * m * len(train_items),
        n * m * len(eval_items),
        n * m * len(test_items)
    )

    train_writer = tf.python_io.TFRecordWriter('%s-%dx%d-train.%s' % (name, size, size, ext), options) if len(train_items) > 0 else None
    eval_writer  = tf.python_io.TFRecordWriter('%s-%dx%d-eval.%s'  % (name, size, size, ext), options) if len(eval_items) > 0  else None
    test_writer  = tf.python_io.TFRecordWriter('%s-%dx%d-test.%s'  % (name, size, size, ext), options) if len(test_items) > 0  else None
    try:
        i = 0
        for scale in scales:
            for flip in flips:
                for rotation in rotations:
                    if i > 0:
                        elapsed_time = (time.time() - start_time) / 60
                        total_time = elapsed_time * (n / i)
                        remaining_time = total_time - elapsed_time
                        logger.info('[%d / %d] scale=%f, flip=%s, rotation=%f (remaining=%d[min], total=%d[min])', i + 1, n, scale, flip, rotation, remaining_time, total_time)
                    else:
                        logger.info('[%d / %d] scale=%f, flip=%s, rotation=%f', i + 1, n, scale, flip, rotation)
                    i += 1

                    reset_tensorflow(seed)
                    with tf.Session() as session:
                        with tf.device(device):
                            # build tf graph
                            logger.info('building tf graphs...')
                            graphs = build_graphs(
                                batch,
                                size,
                                blur_radius,
                                blur_scale,
                                center_crop,
                                scale,
                                flip,
                                rotation,
                                crops,
                                brightness,
                                contrast,
                                gaussian_noise,
                                uniform_noise
                            )
                            # with io.open('%s-%dx%d.graph' % (name, size, size), 'w') as f:
                            #     f.write(str(tf.get_default_graph().as_graph_def()))

                            # run tf graph
                            if len(train_items) > 0:
                                logger.info('processing samples (train)...')
                                bake_records(session, train_writer, train_items, graphs)
                            if len(eval_items) > 0:
                                logger.info('processing samples (eval)...')
                                bake_records(session, eval_writer,  eval_items,  graphs)
                            if len(test_items) > 0:
                                logger.info('processing samples (test)...')
                                bake_records(session, test_writer,  test_items,  graphs)
                    seed = seed * 11113 + seed
    finally:
        if train_writer is not None:
            train_writer.close()
        if eval_writer is not None:
            eval_writer.close()
        if test_writer is not None:
            test_writer.close()
