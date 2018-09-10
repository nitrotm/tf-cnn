###########################################
# topology.py
#
# Model topology file parser and generator V1 for tensorflow.
#
# changelog:
#
# - initial version
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import math, yaml

from pathlib import Path

import tensorflow as tf

from utils import create_logger

from cnn.functions import resolve_initializer, resolve_regularizer, resolve_constraint, resolve_activation

logger = create_logger(__name__)


def format_shape(src, n=[3, 3, 4]):
    items = list()
    s = src.shape.as_list()
    for i in range(len(s)):
        item = s[i]
        m = n[i] if i < len(n) else n[-1]
        if item is None:
            items.append(('%' + str(m) + 's') % '*')
        else:
            items.append(('%' + str(m) + 'd') % item)
    return '[' + ', '.join(items) + ']'

def format_params(params):
    return '{' + ','.join([ '%s=%s' % (key, value) for (key, value) in params ]) + '}'


class LayerGenerator(object):
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

        self.topology = topology
        self.name = name
        self.config = config
        self.inputs = list()
        if 'input' in config:
            self.inputs.append(config['input'])
        if 'inputs' in config:
            self.inputs = self.inputs + config['inputs']
        self.type = config['type']
        self.initializer = resolve_initializer(config.get('initializer', self.topology.initializer))
        self.regularizer = resolve_regularizer(config.get('regularizer', self.topology.regularizer))
        self.constraint = resolve_constraint(config.get('constraint', self.topology.constraint))
        self.activation = resolve_activation(config.get('activation', self.topology.activation))
        self.lrn_radius = config.get('lrn_radius', self.topology.lrn_radius)
        self.batch_normalization = config.get('batch_normalization', self.topology.batch_normalization)
        self.dropout_rate = config.get('dropout_rate', self.topology.dropout_rate)
        self.params = config.get('params', dict())


    def get_param(self, name, required=True, default_value=None):
        """
        Get configuration parameters.

        name: parameter name
        required: raise exception if parameter is not defined
        default_value: default value if parameter is optional and not defined

        return parameter value
        """

        if required and name not in self.params:
            raise Exception("Missing layer parameter %s in %s" % (name, self.name))
        return self.params.get(name, default_value)


    def create(self, inputs, training=True):
        """
        Generate layer according to configuration.

        inputs: layer inputs
        training: training mode

        return layer outputs
        """

        # prepare input(s)
        if len(inputs) == 0:
            raise Exception("Layer %s has no input" % self.name)
        if len(inputs) == 1:
            input_layer = inputs[0]
        else:
            input_layer = tf.concat(inputs, axis=-1, name="%s.input" % self.name)

        # add optional batch normalization layer
        if self.batch_normalization and not self.topology.batch_normalization < 0:
            input_layer = tf.layers.batch_normalization(
                input_layer,
                training=training,
                name="%s.batch_norm" % self.name
            )

        # add optional dropout layer
        if self.dropout_rate > 0.0 and self.topology.dropout_rate >= 0:
            input_layer = tf.layers.dropout(
                inputs=input_layer,
                rate=self.dropout_rate,
                training=training,
                name="%s.dropout%d" % (self.name, int(self.dropout_rate * 100))
            )

        # create concrete layer
        output_layer, convolution = self.create_impl(input_layer)

        # add optional local response normalization (convolution only)
        if convolution and self.lrn_radius > 0 and self.topology.lrn_radius >= 0:
            output_layer = tf.nn.local_response_normalization(
                output_layer,
                depth_radius=self.lrn_radius,
                name="%s.lrn%d" % (self.name, self.lrn_radius)
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
            logger.debug(" %10s: %s -> %s %s", self.name, format_shape(input_layer), format_shape(output_layer), format_params(params))

        return output_layer


    def create_impl(self, inputs):
        """
        Generate concrete layer according to configuration.

        inputs: layer inputs

        return layer outputs
        """

        if self.type == 'avgpool2d' or self.type == 'average_pooling2d':
            return (self.create_avgpool2d(inputs), True)
        elif self.type == 'maxpool2d' or self.type == 'max_pooling2d':
            return (self.create_maxpool2d(inputs), True)
        elif self.type == 'conv2d':
            return (self.create_conv2d(inputs), True)
        elif self.type == 'conv2d_transpose':
            return (self.create_conv2d_transpose(inputs), True)
        elif self.type == 'dense':
            return (self.create_dense(inputs), False)
        else:
            raise Exception("Unsupported layer type %s in %s" % (self.type, self.name))


    def create_avgpool2d(self, inputs):
        """
        Generate a 2d average-pooling layer.

        inputs: layer inputs

        return layer outputs
        """

        return tf.layers.average_pooling2d(
            inputs=inputs,
            pool_size=self.get_param('pool_size'),
            strides=self.get_param('strides'),
            padding=self.get_param('padding', False, 'valid'),
            name=self.name
        )


    def create_maxpool2d(self, inputs):
        """
        Generate a 2d max-pooling layer.

        inputs: layer inputs

        return layer outputs
        """

        return tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=self.get_param('pool_size'),
            strides=self.get_param('strides'),
            padding=self.get_param('padding', False, 'valid'),
            name=self.name
        )


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
            name=self.name
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
            name=self.name
        )


    def create_dense(self, inputs):
        """
        Generate a dense layer with optional flatten before input and reshape after output.

        inputs: layer inputs

        return layer outputs
        """

        if self.get_param("flatten_input", False, False):
            inputs = tf.layers.flatten(
                inputs,
                name="%s.flatten" % self.name
            )
        outputs = tf.layers.dense(
            inputs=inputs,
            units=self.get_param("units"),
            use_bias=True,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            kernel_constraint=self.constraint,
            activation=self.activation,
            name=self.name
        )
        output_shape = self.get_param("output_shape", False)
        if output_shape:
            outputs = tf.reshape(
                outputs,
                output_shape,
                name="%s.output_shape" % self.name
            )
        return outputs



class TopologyGenerator(object):
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

        self.filename = filename
        self.initializer = initializer
        self.regularizer = regularizer
        self.constraint = constraint
        self.activation = activation
        self.lrn_radius = lrn_radius
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate
        self.first = True

        # read configuration
        self.config = yaml.load(Path(self.filename).read_text())
        if 'input_size' not in self.config:
            raise Exception("Missing 'input_size' parameter in topology (%s)" % self.filename)
        if 'input_channels' not in self.config:
            raise Exception("Missing 'input_channels' parameter in topology (%s)" % self.filename)
        if 'input' not in self.config:
            raise Exception("Missing 'input' parameter in topology (%s)" % self.filename)
        if 'output' not in self.config:
            raise Exception("Missing 'output' parameter in topology (%s)" % self.filename)
        if 'layers' not in self.config:
            raise Exception("Missing 'layers' parameter in topology (%s)" % self.filename)

        # required parameters
        self.input_size = self.config['input_size']
        self.input_channels = self.config['input_channels']
        self.input_name = self.config['input']
        self.output_name = self.config['output']
        self.layer_configs = [ self.new_layer(name, layer_config) for (name, layer_config) in self.config['layers'].items() ]

        # optional parameters
        self.output_bias = self.config.get('output_bias', None)
        self.output_scale = self.config.get('output_scale', None)

        # find dependencies
        self.deps = dict()
        self.deps[self.input_name] = set()
        for layer_config in self.layer_configs:
            if layer_config.name in self.deps:
                raise Exception("Duplicate layer found: %s" % layer_config.name)
            self.deps[layer_config.name] = set(layer_config.inputs)

        # check output is defined
        if self.output_name not in self.deps:
            raise Exception("Missing output layer: %s" % self.output_name)


    def new_layer(self, name, config):
        """
        Create a new layer generator.

        name: layer name
        config: layer configuration

        return created layer
        """

        return LayerGenerator(self, name, config)


    def find_layer_config(self, name):
        """
        Find a layer configuration by name.

        name: layer name

        return layer configuration
        """

        for layer_config in self.layer_configs:
            if layer_config.name == name:
                return layer_config
        return None


    def create(self, inputs, training):
        """
        Create model layer(s) based on file configuration.

        inputs: input layer
        training: generate a model for training (if True) or eval/predict (if False)

        return output layer, input layer, layer names and layer dictionary
        """

        # reshape inputs
        inputs = tf.reshape(inputs, [-1, self.input_size, self.input_size, self.input_channels])

        if self.first:
            logger.debug("topology %s (%d layers)", self.filename, len(self.layer_configs))
            logger.debug(" %10s: %s", "inputs", format_shape(inputs))

        # resolve dependencies and generate layers
        layers = dict()
        layers[self.input_name] = inputs
        layer_names = list()
        layer_names.append(list(self.input_name))
        current_configs = self.layer_configs
        while len(current_configs) > 0:
            next_configs = list()
            current_layers = dict()
            current_names = list()
            for layer_config in current_configs:
                layer_inputs = list()
                for layer_name in layer_config.inputs:
                    if layer_name not in layers:
                        break
                    layer_inputs.append(layers[layer_name])
                if len(layer_inputs) != len(layer_config.inputs):
                    next_configs.append(layer_config)
                    continue

                current_layers[layer_config.name] = layer_config.create(layer_inputs, training)
                current_names.append(layer_config.name)

            if len(next_configs) == len(current_configs):
                remaining = set()
                for layer_config in current_configs:
                    remaining = remaining.union(layer_config.inputs)
                raise Exception("Unresolvable dependency(ies) in topology (cycle detected): %s" % str(sorted(remaining)))

            layers.update(current_layers)
            layer_names.append(current_names)
            current_configs = next_configs

        if self.first:
            self.first = False
            logger.debug(" %10s: %s", "outputs", format_shape(layers[self.output_name]))
            logger.debug("topology has %d levels", len(layer_names) - 1)

        # return generated layers
        return (layers[self.output_name], layers[self.input_name], layer_names, layers)


    def normalize_predictions(self, outputs):
        """
        Normalize output values.

        outputs: output layer

        return normalized prediction values
        """

        if self.output_bias is not None:
            outputs += self.output_bias
        if self.output_scale is not None:
            outputs *= self.output_scale
        return tf.sigmoid(outputs)
