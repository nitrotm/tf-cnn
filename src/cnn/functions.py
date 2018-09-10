###########################################
# functions.py
#
# Model function parser and generator for tensorflow.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import math, yaml

from pathlib import Path

import tensorflow as tf

from utils import create_logger

logger = create_logger(__name__)


def parse_function(name):
    """
    Parse function name.

    name: function name and optional parameters (name:param1:param2:...)

    return function name and optional params
    """

    params = name.split(':')
    return (params[0], params[1:])


def resolve_initializer(name):
    """
    Resolve kernel initializer function name.

    name: function name and optional parameters (name:param1:param2:...)

    return resolved function
    """

    name, params = parse_function(name)
    if name is None or name == 'none':
        return None
    if name == 'random_uniform' or name == 'uniform':
        return tf.keras.initializers.random_uniform(*[ float(value) for value in params ])
    if name == 'glorot_uniform':
        return tf.keras.initializers.glorot_uniform()
    if name == 'he_uniform':
        return tf.keras.initializers.he_uniform()
    if name == 'lecun_uniform':
        return tf.keras.initializers.lecun_uniform()
    if name == 'random_normal' or name == 'normal':
        return tf.keras.initializers.random_normal(*[ float(value) for value in params ])
    if name == 'truncated_normal':
        return tf.keras.initializers.truncated_normal(*[ float(value) for value in params ])
    if name == 'glorot_normal' or name == 'glorot' or name == 'xavier':
        return tf.keras.initializers.glorot_normal()
    if name == 'he_normal' or name == 'he':
        return tf.keras.initializers.he_normal()
    if name == 'lecun_normal' or name == 'lecun':
        return tf.keras.initializers.lecun_normal()
    raise Exception("Unsupported initializer function (%s)" % name)


def resolve_regularizer(name):
    """
    Resolve kernel regularizer function name.

    name: function name and optional parameters (name:param1:param2:...)

    return resolved function
    """

    name, params = parse_function(name)
    if name is None or name == 'none':
        return None
    if name == 'l1':
        return tf.keras.regularizers.l1(*[ float(value) for value in params ])
    if name == 'l2':
        return tf.keras.regularizers.l2(*[ float(value) for value in params ])
    if name == 'l1_l2':
        return tf.keras.regularizers.l1_l2(*[ float(value) for value in params ])
    raise Exception("Unsupported regularizer function (%s)" % name)


def resolve_constraint(name):
    """
    Resolve kernel constraint function name.

    name: function name and optional parameters (name:param1:param2:...)

    return resolved function
    """

    name, params = parse_function(name)
    if name is None or name == 'none':
        return None
    if name == 'nonneg':
        return tf.keras.constraints.NonNeg()
    if name == 'unit':
        return tf.keras.constraints.UnitNorm()
    if name == 'max':
        return tf.keras.constraints.MaxNorm(*[ float(value) for value in params ])
    if name == 'minmax':
        return tf.keras.constraints.MinMaxNorm(*[ float(value) for value in params ])
    raise Exception("Unsupported constraint function (%s)" % name)


def resolve_activation(name):
    """
    Resolve activation function name.

    name: function name and optional parameters (name:param1:param2:...)

    return resolved function
    """

    name, params = parse_function(name)
    if name is None or name == 'none' or name == 'identity':
        return None
    if name == 'sigmoid':
        return tf.sigmoid
    if name == 'tanh':
        return tf.tanh
    if name == 'relu':
        return tf.nn.relu
    if name == 'relu6':
        return tf.nn.relu6
    if name == 'crelu':
        return tf.nn.crelu
    if name == 'lrelu' or name == 'leaky_relu':
        return tf.nn.leaky_relu
    if name == 'elu':
        return tf.nn.elu
    if name == 'selu':
        return tf.nn.selu
    if name == 'softmax':
        return tf.nn.softmax
    if name == 'softplus':
        return tf.nn.softplus
    if name == 'softsign':
        return tf.nn.softsign
    raise Exception("Unsupported activation function (%s)" % name)


def resolve_loss(name):
    """
    Resolve loss function name.

    name: function name and optional parameters (name:param1:param2:...)

    return resolved function generator f(num_classes, labels_one_hot, labels_index, logits, weights)
    """

    name, params = parse_function(name)
    if name == 'abs' or name == 'absdiff' or name == 'absolute_difference':
        return lambda num_classes, labels_one_hot, labels_index, logits, weights: tf.losses.absolute_difference(
            labels_one_hot,
            logits,
            tf.stack([ weights for i in range(num_classes) ], axis=-1)
        )
    if name == 'huber' or name == 'huber_loss':
        delta = float(params[0]) if len(params) > 0 else 1.0
        return lambda num_classes, labels_one_hot, labels_index, logits, weights: tf.losses.huber_loss(
            labels_one_hot,
            logits,
            tf.stack([ weights for i in range(num_classes) ], axis=-1),
            delta=delta
        )
    if name == 'mse' or name == 'mean_squared_error':
        return lambda num_classes, labels_one_hot, labels_index, logits, weights: tf.losses.mean_squared_error(
            labels_one_hot,
            logits,
            tf.stack([ weights for i in range(num_classes) ], axis=-1)
        )
    if name == 'log' or name == 'log_loss':
        return lambda num_classes, labels_one_hot, labels_index, logits, weights: tf.losses.log_loss(
            labels_one_hot,
            tf.nn.softmax(logits),
            tf.stack([ weights for i in range(num_classes) ], axis=-1),
            epsilon=1e-5
        )
    if name == 'hinge' or name == 'hinge_loss':
        return lambda num_classes, labels_one_hot, labels_index, logits, weights: tf.losses.hinge_loss(
            labels_one_hot,
            logits,
            tf.stack([ weights for i in range(num_classes) ], axis=-1)
        )
    if name == 'sigmoid' or name == 'sigmoid_cross_entropy':
        return lambda num_classes, labels_one_hot, labels_index, logits, weights: tf.losses.sigmoid_cross_entropy(
            labels_one_hot,
            logits,
            tf.stack([ weights for i in range(num_classes) ], axis=-1)
        )
    if name == 'softmax' or name == 'softmax_cross_entropy':
        return lambda num_classes, labels_one_hot, labels_index, logits, weights: tf.losses.softmax_cross_entropy(
            labels_one_hot,
            logits,
            weights
        )
    if name == 'sparsesoftmax' or name == 'sparse_softmax_cross_entropy':
        return lambda num_classes, labels_one_hot, labels_index, logits, weights: tf.losses.sparse_softmax_cross_entropy(
            labels_index,
            logits,
            weights
        )
    if name == 'jaccard' or name == 'jaccard_loss':
        def jaccard_loss(labels_one_hot, logits, weights, epsilon=1e-6, loss_collection=tf.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
            predictions = tf.nn.softmax(logits)
            intersection = labels_one_hot * predictions
            intersection_sum = tf.losses.compute_weighted_loss(
                intersection,
                weights,
                loss_collection=loss_collection,
                reduction=reduction
            )
            union = labels_one_hot + predictions - intersection
            union_sum = tf.losses.compute_weighted_loss(
                union,
                weights,
                loss_collection=loss_collection,
                reduction=reduction
            )
            return 1.0 - intersection_sum / tf.maximum(union_sum, epsilon)
        return lambda num_classes, labels_one_hot, labels_index, logits, weights: jaccard_loss(
            labels_one_hot,
            logits,
            tf.stack([ weights for i in range(num_classes) ], axis=-1)
        )
    raise Exception("Unsupported loss function (%s)" % name)


def resolve_optimizer(name):
    """
    Resolve activation function name.

    name: function name and optional parameters (name:param1:param2:...)

    return resolved function generator f(learning_rate)
    """

    name, params = parse_function(name)
    if name == 'gd' or name == 'gradient_descent':
        return lambda learning_rate: tf.train.GradientDescentOptimizer(learning_rate)
    if name == 'momentum':
        momentum = float(params[0]) if len(params) > 0 else 0.5
        return lambda learning_rate: tf.train.MomentumOptimizer(learning_rate, momentum)
    if name == 'nesterov':
        momentum = float(params[0]) if len(params) > 0 else 0.5
        return lambda learning_rate: tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    if name == 'rmsprop':
        decay = float(params[0]) if len(params) > 0 else 0.9
        momentum = float(params[1]) if len(params) > 1 else 0.0
        return lambda learning_rate: tf.train.RMSPropOptimizer(learning_rate)
    if name == 'adam':
        beta1 = float(params[0]) if len(params) > 0 else 0.9
        beta2 = float(params[1]) if len(params) > 1 else 0.999
        return lambda learning_rate: tf.train.AdamOptimizer(learning_rate, beta1, beta2)
    if name == 'nadam':
        beta1 = float(params[0]) if len(params) > 0 else 0.9
        beta2 = float(params[1]) if len(params) > 1 else 0.999
        return lambda learning_rate: tf.contrib.opt.NadamOptimizer(learning_rate, beta1, beta2)
    raise Exception("Unsupported optimizer function (%s)" % name)


def resolve_rate_decay(name):
    """
    Resolve learning rate schedule function name.

    name: function name and optional parameters (name:param1:param2:...)

    return resolved function generator f(learning_rate, global_step)
    """

    name, params = parse_function(name)
    if name == 'none' or name == 'constant' or str(name) == '0' or str(name) == '0.0':
        return lambda learning_rate, global_step: learning_rate
    if name == 'itd' or name == 'inverse_time' or name == 'inverse_time_decay':
        decay_steps = float(params[0]) if len(params) > 0 else 1.0
        decay_rate = float(params[1]) if len(params) > 1 else 0.5
        return lambda learning_rate, global_step: tf.train.inverse_time_decay(
            learning_rate,
            global_step,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
    if name == 'poly' or name == 'polynomial' or name == 'polynomial_decay':
        decay_steps = float(params[0]) if len(params) > 0 else 1.0
        end_learning_rate = float(params[1]) if len(params) > 1 else 1e-6
        power = float(params[2]) if len(params) > 2 else 1.0
        return lambda learning_rate, global_step: tf.train.polynomial_decay(
            learning_rate,
            global_step,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=power
        )
    if name == 'exp' or name == 'exponential' or name == 'exponential_decay':
        decay_steps = float(params[0]) if len(params) > 0 else 1.0
        decay_rate = float(params[1]) if len(params) > 1 else 0.99
        return lambda learning_rate, global_step: tf.train.exponential_decay(
            learning_rate,
            global_step,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
    if name == 'natexp' or name == 'natural_exponential' or name == 'natural_exp_decay':
        decay_steps = float(params[0]) if len(params) > 0 else 1.0
        decay_rate = float(params[1]) if len(params) > 1 else 0.0023
        return lambda learning_rate, global_step: tf.train.natural_exp_decay(
            learning_rate,
            global_step,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
    if name == 'cos' or name == 'cosine' or name == 'cosine_decay':
        decay_steps = float(params[0]) if len(params) > 0 else 1.0
        alpha = float(params[1]) if len(params) > 1 else 0.0
        return lambda learning_rate, global_step: tf.train.cosine_decay(
            learning_rate,
            global_step,
            decay_steps=decay_steps,
            alpha=alpha
        )
    if name == 'lincos' or name == 'linear_cosine' or name == 'linear_cosine_decay':
        decay_steps = float(params[0]) if len(params) > 0 else 1.0
        num_periods = float(params[1]) if len(params) > 1 else 0.5
        alpha = float(params[2]) if len(params) > 2 else 0.0
        beta = float(params[3]) if len(params) > 3 else 0.001
        return lambda learning_rate, global_step: tf.train.linear_cosine_decay(
            learning_rate,
            global_step,
            decay_steps=decay_steps,
            num_periods=num_periods,
            alpha=alpha,
            beta=beta
        )
    if name == 'noisylincos' or name == 'noisy_linear_cosine' or name == 'noisy_linear_cosine_decay':
        decay_steps = float(params[0]) if len(params) > 0 else 1.0
        initial_variance = float(params[1]) if len(params) > 1 else 1.0
        variance_decay = float(params[2]) if len(params) > 2 else 0.55
        num_periods = float(params[3]) if len(params) > 3 else 0.5
        alpha = float(params[4]) if len(params) > 4 else 0.0
        beta = float(params[5]) if len(params) > 5 else 0.001
        return lambda learning_rate, global_step: tf.train.noisy_linear_cosine_decay(
            learning_rate,
            global_step,
            decay_steps=decay_steps,
            initial_variance=initial_variance,
            variance_decay=variance_decay,
            num_periods=num_periods,
            alpha=alpha,
            beta=beta
        )
    raise Exception("Unsupported learning rate schedule function (%s)" % name)
