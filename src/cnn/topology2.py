###########################################
# topology2.py
#
# Model topology file parser and generator V2 for tensorflow.
#
# changelog:
#
# - named scope are used to better structure the graph
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import yaml

from pathlib import Path

import tensorflow as tf

from utils import create_logger

import cnn.topology as v1

logger = create_logger(__name__)


class LayerGenerator(v1.LayerGenerator):
    """
    A CNN layer generator from topology file configuration.
    """

    def __init__(self, topology, name, config):
        """
        Initialize a new cnn layer generator.

        topology: topology configuration
        name: layer name
        config: layer configuration dictionary
        """

        super(LayerGenerator, self).__init__(topology, name, config)


    def create(self, inputs, training=True):
        """
        Generate layer according to configuration.

        inputs: layer inputs

        return layer outputs
        """

        if len(inputs) == 0:
            raise Exception("Layer %s has no input" % self.name)

        with tf.name_scope(self.name):
            # prepare input(s)
            if len(inputs) == 1:
                input_layer = inputs[0]
            else:
                input_layer = tf.concat(inputs, axis=-1, name="input_concat")

            # add optional batch normalization layer
            if self.batch_normalization and not self.topology.batch_normalization < 0:
                input_layer = tf.layers.batch_normalization(
                    input_layer,
                    training=training,
                    name="%s_batch_norm" % self.name
                )

            # add optional dropout layer
            if self.dropout_rate > 0.0 and self.topology.dropout_rate >= 0:
                input_layer = tf.layers.dropout(
                    inputs=input_layer,
                    rate=self.dropout_rate,
                    training=training,
                    name="dropout_%d" % int(self.dropout_rate * 100)
                )

            # create concrete layer
            output_layer, convolution = self.create_impl(input_layer)

            # add optional local response normalization (convolution only)
            if convolution and self.lrn_radius > 0 and self.topology.lrn_radius >= 0:
                output_layer = tf.nn.local_response_normalization(
                    output_layer,
                    depth_radius=self.lrn_radius,
                    name="lrn_%d" % self.lrn_radius
                )

            # logging
            if self.topology.first:
                params = list()
                if self.config.get('activation', self.topology.activation) != 'none':
                    params.append(('act', self.config.get('activation', self.topology.activation)))
                if self.config.get('initializer', self.topology.initializer) != 'none':
                    params.append(('ini', self.config.get('initializer', self.topology.initializer)))
                if self.config.get('regularizer', self.topology.regularizer) != 'none':
                    params.append(('reg', self.config.get('regularizer', self.topology.regularizer)))
                if self.config.get('constraint', self.topology.constraint) != 'none':
                    params.append(('cnt', self.config.get('constraint', self.topology.constraint)))
                if self.batch_normalization and not self.topology.batch_normalization < 0:
                    params.append(('bn', 'y'))
                if self.dropout_rate > 0.0 and self.topology.dropout_rate >= 0:
                    params.append(('dr', self.dropout_rate))
                if convolution and self.lrn_radius > 0 and self.topology.lrn_radius >= 0:
                    params.append(('lrn', self.lrn_radius))
                logger.debug(" %9s: %s -> %s %s", self.name, v1.format_shape(input_layer), v1.format_shape(output_layer), v1.format_params(params))

        return output_layer


    def create_conv2d(self, inputs):
        """
        Generate a 2d convolution layer.

        inputs: layer inputs

        return layer outputs
        """

        return tf.layers.conv2d(
            inputs=inputs,
            filters=self.get_param('filters'),
            kernel_size=self.get_param('kernel_size'),
            strides=self.get_param('strides', False, 1),
            dilation_rate=self.get_param('dilation_rate', False, 1),
            padding=self.get_param('padding', False, 'valid'),
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            kernel_constraint=self.constraint,
            activation=self.activation,
            name="%s_conv2d" % self.name
        )


    def create_conv2d_transpose(self, inputs):
        """
        Generate a transposed 2d convolution layer.

        inputs: layer inputs

        return layer outputs
        """

        return tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=self.get_param('filters'),
            kernel_size=self.get_param('kernel_size'),
            strides=self.get_param('strides', False, 1),
            padding=self.get_param('padding', False, 'valid'),
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            kernel_constraint=self.constraint,
            activation=self.activation,
            name="%s_conv2d_transpose" % self.name
        )


    def create_dense(self, inputs):
        """
        Generate a dense layer with optional flatten before input and reshape after output.

        inputs: layer inputs

        return layer outputs
        """

        if self.get_param("flatten_input", False, False):
            inputs = tf.layers.flatten(inputs, name="input_flatten")
        outputs = tf.layers.dense(
            inputs=inputs,
            units=self.get_param("units"),
            use_bias=True,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            kernel_constraint=self.constraint,
            activation=self.activation,
            name="%s_dense" % self.name
        )
        output_shape = self.get_param("output_shape", False)
        if output_shape:
            outputs = tf.reshape(outputs, output_shape, name="output_reshape")
        return outputs



class TopologyGenerator(v1.TopologyGenerator):
    """
    A CNN model generator from topology file configuration.
    """

    def __init__(self, filename, initializer=None, regularizer=None, constraint=None, activation=None, lrn_radius=0, batch_normalization=False, dropout_rate=0.0):
        """
        Initialize a new cnn model generator.

        filename: topology filename to load (yaml)
        initializer: optional kernel initializer function
        regularizer: optional kernel regularizer function
        constraint: optional kernel constraint function
        activation: optional layer activation function
        lrn_radius: optional local response normalization radius (global default if not specified in layer)
        batch_normalization: enable optional batch normalization (global default if not specified in layer)
        dropout_rate: optional layer dropout rate (global default if not specified in layer)
        """

        super(TopologyGenerator, self).__init__(filename, initializer, regularizer, constraint, activation, lrn_radius, batch_normalization, dropout_rate)


    def new_layer(self, name, config):
        """
        Create a new layer generator.

        name: layer name
        config: layer configuration

        return created layer
        """

        return LayerGenerator(self, name, config)


    def create(self, inputs, training):
        """
        Create model layer(s) based on file configuration.

        inputs: input layer
        training: generate a model for training (if True) or eval/predict (if False)

        return output layer, input layer, layer names and layer dictionary
        """

        with tf.name_scope('topology'):
            return super(TopologyGenerator, self).create(inputs, training)
