###########################################
# model.py
#
# U-net inspired model architecture in tensorflow.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import math, time

import tensorflow as tf

from utils import create_logger, gaussian_kernel

from cnn.functions import resolve_loss, resolve_rate_decay, resolve_optimizer
from cnn.topology  import TopologyGenerator as TopologyGenerator1
from cnn.topology2 import TopologyGenerator as TopologyGenerator2

logger = create_logger(__name__)


def resolve_topology(version):
    """
    Resolve topology generator by version.

    version: requested version

    return topology generator
    """

    if version <= 1:
        return TopologyGenerator1
    elif version == 2:
        return TopologyGenerator2
    else:
        raise Exception("Unsupported topology generator version (%d)" % version)


def parse_learning_rate_decay(value):
    params = str(value).split(':')
    if len(params) == 1:
        return (float(params[0]), 0.0)
    if len(params) == 2:
        return (float(params[0]), float(params[1]))
    raise Exception("Invalid learning rate decay (%s)" % value)


def build_mosaic(t, size):
    """
    Build image mosaic from tensor channels

    t: input tensor
    size: target mosaic width and height

    return mosaic image (single channel), or None if the input tensor is not known
    """

    if t.shape.ndims != 4:
        return None
    shape = t.shape[1:].as_list()
    height, width, depth = shape[0], shape[1], shape[2]
    if not height or not width or not depth:
        return None
    if depth <= 3:
        n = 1
    else:
        n = int(math.ceil(math.sqrt(float(depth))))

    longest = max(width, height)
    scale = min(1.0, size / longest / n)
    width2 = int(math.floor(width * scale))
    height2 = int(math.floor(height * scale))

    x = 0
    rows = list()
    columns = list()
    with tf.name_scope('mosaic'):
        for channel in tf.unstack(tf.image.resize_bilinear(t, [height2, width2]), axis=-1):
            if x == n:
                rows.append(tf.concat(columns, axis=2))
                columns = list()
                x = 0
            columns.append(tf.reshape(channel, [-1, height2, width2, 1]))
            x += 1
        if x == n:
            rows.append(tf.concat(columns, axis=2))
        else:
            paddings = [
                [0, 0],
                [0, 0],
                [0, (n-x) * width2],
                [0, 0]
            ]
            rows.append(tf.pad(tf.concat(columns, axis=2), paddings))
        return tf.concat(rows, axis=1)


def build_prediction_image(num_classes, labels_index, predictions):
    """
    Build visual prediction

    num_classes: number of classes
    labels_index: expected labels
    predictions: predicted labels

    return mosaic image for each class > 0
    """
    with tf.name_scope('predictions_color'):
        images = list()
        for i in range(1, num_classes):
            true_positive = tf.logical_and(tf.equal(labels_index, i), tf.equal(predictions, i))
            false_positive = tf.logical_and(tf.not_equal(labels_index, i), tf.equal(predictions, i))
            true_negative = tf.logical_and(tf.not_equal(labels_index, i), tf.not_equal(predictions, i))
            false_negative = tf.logical_and(tf.equal(labels_index, i), tf.not_equal(predictions, i))
            images.append(
                tf.cast(tf.stack([tf.logical_or(false_positive, false_negative), true_positive, false_negative], axis=-1), tf.uint8)
            )
        return 255 * tf.concat(images, axis=1)


def topology_cnn_model(topology, verbose=False):
    """
    CNN/FCN tensorflow estimator model function generator based on a topological definition file.

    topology: configured topology
    verbose: verbose output mode

    return estimator model function
    """

    def model_fn(features, labels, mode, params, config):
        """
        CNN/FCN tensorflow estimator model function based on a topological definition file.

        features: sample features for current batch (dict of tensors)
        labels: sample labels for current batch (tensor)
        mode: train, eval or predict
        params: model hyperparameters
        config: global configuration

        return estimator operation specification
        """

        classes = params.get("classes", [0.0, 1.0])
        label_weighting = params.get("label_weighting", 0.0)
        initial_learning_rate = params.get("learning_rate", 0.001)

        # Generate topology
        (outputs, inputs, layer_names, layers) = topology.create(features["rgb"], mode == tf.estimator.ModeKeys.TRAIN)

        # Normalize predictions
        with tf.name_scope('normalization'):
            logits = tf.layers.conv2d(outputs, len(classes), 1, padding="same")
            predictions = tf.argmax(tf.nn.softmax(logits), axis=-1, output_type=tf.int32)

        # Prepare labels and weights if available
        with tf.name_scope('labelling'):
            if labels is not None:
                labels = tf.reshape(labels, [-1, topology.input_size, topology.input_size])
            elif 'mask' in features:
                labels = tf.reshape(features["mask"], [-1, topology.input_size, topology.input_size])
            if label_weighting > -1 and 'weight' in features:
                weights = tf.reshape(features["weight"], [-1, topology.input_size, topology.input_size])
            else:
                weights = None
            if labels is not None:
                labels_one_hot = tf.stack([tf.cast(tf.equal(labels, value), tf.float32) for value in classes], axis=-1)
                labels_index = tf.argmax(labels_one_hot, axis=-1, output_type=tf.int32)
            else:
                labels_one_hot = None
                labels_index = None

        # Configure the prediction (for PREDICT mode)
        if mode == tf.estimator.ModeKeys.PREDICT:
            # store predictions
            data = dict()
            data["inputs"] = inputs
            data["logits"] = logits
            data["predictions"] = predictions
            if labels is not None:
                data["labels"] = labels
                data["labels_one_hot"] = labels_one_hot
                data["labels_index"] = labels_index
                data["mosaic"] = build_prediction_image(len(classes), labels_index, predictions)
            if weights is not None:
                data["weight"] = weights
            if 'name' in features:
                data['name'] = features['name']
            if 'variant' in features:
                data['variant'] = features['variant']
            if verbose:
                with tf.name_scope('hidden'):
                    names = list()
                    for i in range(len(layer_names)):
                        for layer_name in layer_names[i]:
                            names.append(layer_name)
                            data["layer_" + layer_name] = layers[layer_name]
                            mosaic = build_mosaic(layers[layer_name], 1024)
                            if mosaic is not None:
                                data["mosaic_" + layer_name] = mosaic

            # return prediction
            return tf.estimator.EstimatorSpec(mode=mode, predictions=data)

        # Loss function
        if labels is None:
            raise Exception("Missing labels in loss function definition")
        if weights is None:
            weights = tf.ones_like(labels)
        with tf.name_scope('loss_fn'):
            if label_weighting > -1 and label_weighting != 0:
                weights = tf.maximum(tf.reduce_min(weights), labels * label_weighting)
            loss_func = resolve_loss(params.get("loss", 'absdiff'))
            loss = loss_func(
                len(classes),
                labels_one_hot,
                labels_index,
                logits,
                weights
            ) + tf.losses.get_regularization_loss()

        # Configure the training (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # compute learning rate
            global_step = tf.train.get_global_step()
            decay_fn = resolve_rate_decay(params.get("learning_rate_decay", "0.0:0.0"))
            with tf.name_scope('learning_schedule'):
                learning_rate = decay_fn(initial_learning_rate, global_step)

            # output summaries
            tf.summary.scalar('learning_rate', learning_rate)
            with tf.name_scope('predictions'):
                tf.summary.image('predictions', build_prediction_image(len(classes), labels_index, predictions), max_outputs=1)
            with tf.name_scope('observables'):
                mosaic = build_mosaic(logits, 1024)
                if mosaic is not None:
                    tf.summary.image('logits', mosaic, max_outputs=1)
                if verbose:
                    tf.summary.image('inputs', inputs, max_outputs=1)
                    tf.summary.image('labels', tf.stack([tf.cast(labels * 255.0, tf.uint8)], axis=-1), max_outputs=1)
                    tf.summary.image('weights', tf.stack([tf.cast(weights * 255.0, tf.uint8)], axis=-1), max_outputs=1)
                tf.summary.histogram('dist_outputs', outputs)
                if verbose:
                    tf.summary.histogram('dist_inputs', inputs)
                    tf.summary.histogram('dist_labels', labels)
                    tf.summary.histogram('dist_weights', weights)
            with tf.name_scope('activations'):
                if verbose:
                    for i in range(len(layer_names)):
                        for layer_name in layer_names[i]:
                            mosaic = build_mosaic(layers[layer_name], 1024)
                            if mosaic is not None:
                                tf.summary.image('%02d_img_%s' % (i, layer_name), mosaic, max_outputs=1)
                            tf.summary.histogram('%02d_dist_%s' % (i, layer_name), layers[layer_name])

            # TODO: support for adaptative rate per layer?
            # v1 = tf.get_variables(...)
            # v2 = ...
            # op1 = optimizer1.apply_gradients(zip(tf.gradients(loss, v1), v1))
            # op2 = ...
            # op = tf.group(op1, ...)

            # return configured optimizer
            optimizer_func = resolve_optimizer(params.get("optimizer", 'gd'))
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=optimizer_func(learning_rate).minimize(
                    loss=loss,
                    global_step=global_step,
                    name="optimizer"
                ),
            )

        # Configure the evaluation (for EVAL mode)
        if mode == tf.estimator.ModeKeys.EVAL:
            with tf.name_scope('metrics'):
                # compute jaccard index, precision and recall
                metrics = dict()
                metrics["jaccard"] = tf.metrics.mean_iou(
                    labels=labels_index,
                    predictions=predictions,
                    num_classes=len(classes),
                    name="jaccard"
                )
                for i in range(len(classes)):
                    metrics["precision_c%d" % i] = tf.metrics.precision(
                        labels=tf.equal(labels_index, i),
                        predictions=tf.equal(predictions, i),
                        name="precision_c%d" % i
                    )
                    metrics["recall_c%d" % i] = tf.metrics.recall(
                        labels=tf.equal(labels_index, i),
                        predictions=tf.equal(predictions, i),
                        name="recall_c%d" % i
                    )

            # return evaluation metrics
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

        raise Exception("Unsupported model mode (%s)" % mode)

    return model_fn
