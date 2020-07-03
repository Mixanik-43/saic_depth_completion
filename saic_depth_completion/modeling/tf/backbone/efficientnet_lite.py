# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.
[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""
# modified by Mikhail Artemyev

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math
import re

from absl import logging
import numpy as np
import six
from six.moves import xrange
import tensorflow as tf

from saic_depth_completion.modeling.tf.checkpoint_utils import submodel_state_dict, default_set_torch_weights
from saic_depth_completion.utils import registry

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'survival_prob', 'relu_fn', 'batch_norm', 'use_se',
    'se_coefficient', 'local_pooling', 'condconv_num_experts',
    'clip_projection_output', 'blocks_args', 'fix_head_stem',
])
# Note: the default value of None is not necessarily valid. It is valid to leave
# width_coefficient, depth_coefficient at None, which is treated as 1.0 (and
# which also allows depth_divisor and min_depth to be left at None).
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type', 'fused_conv',
    'super_pixel', 'condconv'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

_DEFAULT_BLOCKS_ARGS = [
    'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
]

def efficientnet_lite(width_coefficient=None,
                      depth_coefficient=None,
                      dropout_rate=0.2,
                      survival_prob=0.8):
    """Creates a efficientnet model."""
    global_params = GlobalParams(
        blocks_args=_DEFAULT_BLOCKS_ARGS,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-5,
        dropout_rate=dropout_rate,
        survival_prob=survival_prob,
        data_format='channels_last',
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        relu_fn=tf.keras.layers.ReLU(6.0),  # Relu6 is for easier quantization.
        # The default is TPU-specific batch norm.
        # The alternative is tf.layers.BatchNormalization.
        batch_norm=tf.keras.layers.BatchNormalization,  # TPU-specific requirement.
        clip_projection_output=False,
        fix_head_stem=True,  # Don't scale stem and head.
        local_pooling=True,  # special cases for tflite issues.
        use_se=False)  # SE is not well supported on many lite devices.
    return global_params


class BlockDecoder(object):
    """Block Decoder for readability."""

    def _decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        if six.PY2:
            assert isinstance(block_string, (str, unicode))
        else:
            assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=[int(options['s'][0]),
                     int(options['s'][1])],
            conv_type=int(options['c']) if 'c' in options else 0,
            fused_conv=int(options['f']) if 'f' in options else 0,
            super_pixel=int(options['p']) if 'p' in options else 0,
            condconv=('cc' in block_string))

    def _encode_block_string(self, block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters,
            'c%d' % block.conv_type,
            'f%d' % block.fused_conv,
            'p%d' % block.super_pixel,
        ]
        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:  # pylint: disable=g-bool-id-comparison
            args.append('noskip')
        if block.condconv:
            args.append('cc')
        return '_'.join(args)

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.
        Args:
          string_list: a list of strings, each string is a notation of block.
        Returns:
          A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.
        Args:
          blocks_args: A list of namedtuples to represent blocks arguments.
        Returns:
          a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings



def efficientnet_lite_params(model_name):
    """Get efficientnet params based on model name."""
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
        'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
        'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
        'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
        'efficientnet-lite4': (1.4, 1.8, 300, 0.3),
    }
    return params_dict[model_name]


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model."""
    if model_name.startswith('efficientnet-lite'):
        width_coefficient, depth_coefficient, _, dropout_rate = (
            efficientnet_lite_params(model_name))
        global_params = efficientnet_lite(
            width_coefficient, depth_coefficient, dropout_rate)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params:
        # ValueError will be raised here if override_params has fields not included
        # in global_params.
        global_params = global_params._replace(**override_params)

    decoder = BlockDecoder()
    blocks_args = decoder.decode(global_params.blocks_args)

    logging.info('global_params= %s', global_params)
    return blocks_args, global_params


conv_kernel_initializer = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

dense_kernel_initializer = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def Conv2dPad(*args, padding=(1, 1), **kwargs):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.ZeroPadding2D(padding))
    result.add(tf.keras.layers.Conv2D(*args, **kwargs))
    return result


def DepthwiseConv2dPad(*args, padding=(1, 1), **kwargs):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.ZeroPadding2D(padding))
    result.add(tf.keras.layers.DepthwiseConv2D(*args, **kwargs))
    return result


def round_filters(filters, global_params, skip=False):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if skip or not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    logging.info('round_filter input=%s output=%s', orig_f, new_filters)
    return int(new_filters)


def round_repeats(repeats, global_params, skip=False):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if skip or not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class MBConvBlock(tf.keras.layers.Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.
    Attributes:
      endpoints: dict. A list of internal tensors.
    """

    def __init__(self, block_args, global_params):
        """Initializes a MBConv block.
        Args:
          block_args: BlockArgs, arguments to create a Block.
          global_params: GlobalParams, a set of global parameters.
        """
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self._local_pooling = global_params.local_pooling
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self._batch_norm = global_params.batch_norm
        self._condconv_num_experts = global_params.condconv_num_experts
        self._data_format = global_params.data_format
        self._se_coefficient = global_params.se_coefficient
        if self._data_format == 'channels_first':
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]

        self._relu_fn = global_params.relu_fn

        self._clip_projection_output = global_params.clip_projection_output

        self.endpoints = None

        self.conv_cls = tf.keras.layers.Conv2D
        self.depthwise_conv_cls = DepthwiseConv2dPad

        # Builds the block accordings to arguments.
        self._build()

    def block_args(self):
        return self._block_args

    def _build(self):
        """Builds block according to the arguments."""
        if self._block_args.super_pixel == 1:
            self._superpixel = tf.keras.layers.Conv2D(
                self._block_args.input_filters,
                kernel_size=[2, 2],
                strides=[2, 2],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                data_format=self._data_format,
                use_bias=False)
            self._bnsp = self._batch_norm(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon)

        if self._block_args.condconv:
            # Add the example-dependent routing function
            self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
                data_format=self._data_format)
            self._routing_fn = tf.keras.layers.Dense(
                self._condconv_num_experts, activation=tf.nn.sigmoid)

        filters = self._block_args.input_filters * self._block_args.expand_ratio
        kernel_size = self._block_args.kernel_size

        # Fused expansion phase. Called if using fused convolutions.
        self._fused_conv = self.conv_cls(
            filters=filters,
            kernel_size=[kernel_size, kernel_size],
            strides=self._block_args.strides,
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False)

        # Expansion phase. Called if not using fused convolutions and expansion
        # phase is necessary.
        self._expand_conv = self.conv_cls(
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            data_format=self._data_format,
            use_bias=False)
        self._bn0 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

        # Depth-wise convolution phase. Called if not using fused convolutions.
        self._depthwise_conv = self.depthwise_conv_cls(
            kernel_size=[kernel_size, kernel_size],
            strides=self._block_args.strides,
            depthwise_initializer=conv_kernel_initializer,
            padding=(kernel_size // 2, kernel_size // 2),
            data_format=self._data_format,
            use_bias=False)

        self._bn1 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)


        # Output phase.
        filters = self._block_args.output_filters
        self._project_conv = self.conv_cls(
            filters=filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same',
            data_format=self._data_format,
            use_bias=False)
        self._bn2 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

    def call(self, inputs, training=True, survival_prob=None):
        """Implementation of call().
        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
          survival_prob: float, between 0 to 1, drop connect rate.
        Returns:
          A output tensor.
        """

        x = inputs

        fused_conv_fn = self._fused_conv
        expand_conv_fn = self._expand_conv
        depthwise_conv_fn = self._depthwise_conv
        project_conv_fn = self._project_conv

        if self._block_args.condconv:
            pooled_inputs = self._avg_pooling(inputs)
            routing_weights = self._routing_fn(pooled_inputs)
            # Capture routing weights as additional input to CondConv layers
            fused_conv_fn = functools.partial(
                self._fused_conv, routing_weights=routing_weights)
            expand_conv_fn = functools.partial(
                self._expand_conv, routing_weights=routing_weights)
            depthwise_conv_fn = functools.partial(
                self._depthwise_conv, routing_weights=routing_weights)
            project_conv_fn = functools.partial(
                self._project_conv, routing_weights=routing_weights)

        # creates conv 2x2 kernel
        if self._block_args.super_pixel == 1:
            with tf.variable_scope('super_pixel'):
                x = self._relu_fn(
                    self._bnsp(self._superpixel(x), training=training))
            logging.info(
                'Block start with SuperPixel: %s shape: %s', x.name, x.shape)

        if self._block_args.fused_conv:
            # If use fused mbconv, skip expansion and use regular conv.
            x = self._relu_fn(self._bn1(fused_conv_fn(x), training=training))
            logging.info('Conv2D: %s shape: %s', x.name, x.shape)
        else:
            # Otherwise, first apply expansion and then apply depthwise conv.
            if self._block_args.expand_ratio != 1:
                x = self._relu_fn(self._bn0(expand_conv_fn(x), training=training))
                # logging.info('Expand: %s shape: %s', x.name, x.shape)

            x = self._relu_fn(self._bn1(depthwise_conv_fn(x), training=training))
            # logging.info('DWConv: %s shape: %s', x.name, x.shape)

        self.endpoints = {'expansion_output': x}

        x = self._bn2(project_conv_fn(x), training=training)
        # Add identity so that quantization-aware training can insert quantization
        # ops correctly.
        x = tf.identity(x)
        if self._clip_projection_output:
            x = tf.clip_by_value(x, -6, 6)
        if self._block_args.id_skip:
            if all(
                    s == 1 for s in self._block_args.strides
            ) and inputs.get_shape().as_list()[-1] == x.get_shape().as_list()[-1]:
                x = tf.add(x, inputs)
        return x

    def set_torch_weights(self, torch_weights):
        if self._block_args.expand_ratio != 1:
            default_set_torch_weights(self._bn0, submodel_state_dict(torch_weights, 'bn1.'))
            default_set_torch_weights(self._bn1, submodel_state_dict(torch_weights, 'bn2.'))
            default_set_torch_weights(self._bn2, submodel_state_dict(torch_weights, 'bn3.'))
            default_set_torch_weights(self._expand_conv, submodel_state_dict(torch_weights, 'conv_pw.'))
            default_set_torch_weights(self._depthwise_conv, submodel_state_dict(torch_weights, 'conv_dw.'), weights_permute_order=(2, 3, 0, 1))
            default_set_torch_weights(self._project_conv, submodel_state_dict(torch_weights, 'conv_pwl.'))
        else:
            default_set_torch_weights(self._bn1, submodel_state_dict(torch_weights, 'bn1.'))
            default_set_torch_weights(self._bn2, submodel_state_dict(torch_weights, 'bn2.'))
            default_set_torch_weights(self._project_conv, submodel_state_dict(torch_weights, 'conv_pw.'))
            default_set_torch_weights(self._depthwise_conv, submodel_state_dict(torch_weights, 'conv_dw.'), weights_permute_order=(2, 3, 0, 1))


class MBConvBlockWithoutDepthwise(MBConvBlock):
    """MBConv-like block without depthwise convolution and squeeze-and-excite."""

    def _build(self):
        """Builds block according to the arguments."""
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = tf.keras.layers.Conv2D(
                filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding='same',
                use_bias=False)
            self._bn0 = self._batch_norm(
                axis=self._channel_axis,
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon)

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=self._block_args.strides,
            padding='same',
            use_bias=False)
        self._bn1 = self._batch_norm(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon)

    def call(self, inputs, training=True, survival_prob=None):
        """Implementation of call().
        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
          survival_prob: float, between 0 to 1, drop connect rate.
        Returns:
          A output tensor.
        """
        logging.info('Block input: %s shape: %s', inputs.name, inputs.shape)
        if self._block_args.expand_ratio != 1:
            x = self._relu_fn(self._bn0(self._expand_conv(inputs), training=training))
        else:
            x = inputs
        logging.info('Expand: %s shape: %s', x.name, x.shape)

        self.endpoints = {'expansion_output': x}

        x = self._bn1(self._project_conv(x), training=training)
        # Add identity so that quantization-aware training can insert quantization
        # ops correctly.
        x = tf.identity(x)
        if self._clip_projection_output:
            x = tf.clip_by_value(x, -6, 6)

        if self._block_args.id_skip:
            if all(
                    s == 1 for s in self._block_args.strides
            ) and self._block_args.input_filters == self._block_args.output_filters:
                # Apply only if skip connection presents.
                x = tf.add(x, inputs)
        logging.info('Project: %s shape: %s', x.name, x.shape)
        return x

    def set_torch_weights(self, torch_weights):
        default_set_torch_weights(self._bn0, submodel_state_dict(torch_weights, 'bn1.'))
        default_set_torch_weights(self._bn1, submodel_state_dict(torch_weights, 'bn2.'))
        default_set_torch_weights(self._fused_conv, submodel_state_dict(torch_weights, 'conv_pw.'))
        default_set_torch_weights(self._project_conv, submodel_state_dict(torch_weights, 'conv_pwl.'))


class EfficientNetLite(tf.keras.layers.Layer):
    """A class implements tf.keras.Model for MNAS-like model.
      Reference: https://arxiv.org/abs/1807.11626
    """

    def __init__(self, model_name, input_shape=(320, 256, 3), override_params=None, out_embeddings_list=None, **kwargs):
        """Initializes an `Model` instance.
        Args:
          blocks_args: A list of BlockArgs to construct block modules.
          global_params: GlobalParams, a set of global parameters.
        Raises:
          ValueError: when blocks_args is not specified as a list.
        """
        super(EfficientNetLite, self).__init__()
        self.model_name = model_name
        blocks_args, global_params = get_model_params(model_name, override_params)
        if not isinstance(blocks_args, list):
            raise ValueError('blocks_args should be a list.')
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._relu_fn = global_params.relu_fn
        self._batch_norm = global_params.batch_norm
        self._fix_head_stem = global_params.fix_head_stem

        self.endpoints = None
        self.out_embeddings_list = out_embeddings_list

        self._build_layers()

        self._build_model(input_shape)

    def _get_conv_block(self, conv_type):
        conv_block_map = {0: MBConvBlock, 1: MBConvBlockWithoutDepthwise}
        return conv_block_map[conv_type]

    def _build_layers(self):
        """Builds a model."""
        self._blocks = []
        batch_norm_momentum = self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._global_params.batch_norm_epsilon
        if self._global_params.data_format == 'channels_first':
            channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            channel_axis = -1
            self._spatial_dims = [1, 2]

        # Stem part.
        self._conv_stem = Conv2dPad(
            filters=round_filters(32, self._global_params, self._fix_head_stem),
            kernel_size=[3, 3],
            strides=[2, 2],
            padding=(1, 1),
            data_format=self._global_params.data_format,
            use_bias=False)
        self._bn0 = self._batch_norm(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)

        # Builds blocks.
        for i, block_args in enumerate(self._blocks_args):
            assert block_args.num_repeat > 0
            assert block_args.super_pixel in [0, 1, 2]
            # Update block input and output filters based on depth multiplier.
            input_filters = round_filters(block_args.input_filters,
                                          self._global_params)

            output_filters = round_filters(block_args.output_filters,
                                           self._global_params)
            kernel_size = block_args.kernel_size
            if self._fix_head_stem and (i == 0 or i == len(self._blocks_args) - 1):
                repeats = block_args.num_repeat
            else:
                repeats = round_repeats(block_args.num_repeat, self._global_params)
            block_args = block_args._replace(
                input_filters=input_filters,
                output_filters=output_filters,
                num_repeat=repeats)

            # The first block needs to take care of stride and filter size increase.
            conv_block = self._get_conv_block(block_args.conv_type)
            if not block_args.super_pixel:  # no super_pixel at all
                self._blocks.append(conv_block(block_args, self._global_params))
            else:
                # if superpixel, adjust filters, kernels, and strides.
                depth_factor = int(4 / block_args.strides[0] / block_args.strides[1])
                block_args = block_args._replace(
                    input_filters=block_args.input_filters * depth_factor,
                    output_filters=block_args.output_filters * depth_factor,
                    kernel_size=((block_args.kernel_size + 1) // 2 if depth_factor > 1
                                 else block_args.kernel_size))
                # if the first block has stride-2 and super_pixel trandformation
                if (block_args.strides[0] == 2 and block_args.strides[1] == 2):
                    block_args = block_args._replace(strides=[1, 1])
                    self._blocks.append(conv_block(block_args, self._global_params))
                    block_args = block_args._replace(  # sp stops at stride-2
                        super_pixel=0,
                        input_filters=input_filters,
                        output_filters=output_filters,
                        kernel_size=kernel_size)
                elif block_args.super_pixel == 1:
                    self._blocks.append(conv_block(block_args, self._global_params))
                    block_args = block_args._replace(super_pixel=2)
                else:
                    self._blocks.append(conv_block(block_args, self._global_params))
            if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
            for _ in xrange(block_args.num_repeat - 1):
                self._blocks.append(conv_block(block_args, self._global_params))

    def _build_model(self, input_shape):
        """Implementation of call().
        Args:
          inputs: input tensors.
          training: boolean, whether the model is constructed for training.
          features_only: build the base feature network only.
          pooled_features_only: build the base network for features extraction
            (after 1x1 conv layer and global pooling, but before dropout and fc
            head).
        Returns:
          output tensors.
        """

        self.feature_channels = []
        inputs = tf.keras.Input(input_shape)
        outputs = inputs
        self.endpoints = {}
        reduction_idx = 0
        # Calls Stem layers
        outputs = self._relu_fn(self._bn0(self._conv_stem(outputs)))
        self.endpoints['stem'] = outputs

        # Calls blocks.

        for idx, block in enumerate(self._blocks):
            is_reduction = False  # reduction flag for blocks after the stem layer
            # If the first block has super-pixel (space-to-depth) layer, then stem is
            # the first reduction point.
            if (block.block_args().super_pixel == 1 and idx == 0):
                reduction_idx += 1
                self.endpoints['reduction_%s' % reduction_idx] = outputs

            elif ((idx == len(self._blocks) - 1) or
                  self._blocks[idx + 1].block_args().strides[0] > 1):
                is_reduction = True
                reduction_idx += 1

            outputs = block(outputs)
            self.endpoints['block_%s' % idx] = outputs
            if (self.out_embeddings_list is not None) and (idx in self.out_embeddings_list):
                self.feature_channels.append(block.block_args().output_filters)
            if is_reduction:
                self.endpoints['reduction_%s' % reduction_idx] = outputs
            if block.endpoints:
                for k, v in six.iteritems(block.endpoints):
                    self.endpoints['block_%s/%s' % (idx, k)] = v
                    if is_reduction:
                        self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
        self.endpoints['features'] = outputs

        if self.out_embeddings_list is not None:
            result = [inputs, self.endpoints['stem']] + [self.endpoints['block_{}'.format(i-1)] for i in self.out_embeddings_list]
        else:
            result = outputs

        self.model = tf.keras.models.Model(inputs, result, name=self.model_name)

    def call(self, x, **kwargs):
        return self.model.call(x, **kwargs)


    def set_torch_weights(self, torch_weights):
        torch_blocks_list = sorted({layer_name[:len('layers.x.y.')] for layer_name in torch_weights.keys() if layer_name.startswith('layers.')})
        for block_id in range(1, len(self._blocks)):
            self._blocks[block_id].set_torch_weights(submodel_state_dict(torch_weights, torch_blocks_list[block_id + 2]))
        self._blocks[0].set_torch_weights(submodel_state_dict(torch_weights, 'layers.2.0.0.'))
        default_set_torch_weights(self._conv_stem, submodel_state_dict(torch_weights, 'layers.1.0.'))
        default_set_torch_weights(self._bn0, submodel_state_dict(torch_weights, 'layers.1.1.'))



@registry.TF_BACKBONES.register("tf-efficientnet-lite-b0")
def EfficientNetLiteB0(out_embeddings_list=(3, 5, 11, 16), **kwargs):
    return EfficientNetLite('efficientnet-lite0', out_embeddings_list=out_embeddings_list, **kwargs)
