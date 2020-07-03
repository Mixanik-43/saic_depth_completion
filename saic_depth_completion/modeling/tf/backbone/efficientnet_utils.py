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
"""Model utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python.tpu import tpu_function  # pylint:disable=g-direct-tensorflow-import


class TpuBatchNormalization(tf.layers.BatchNormalization):
    # class TpuBatchNormalization(tf.layers.BatchNormalization):
    """Cross replica batch normalization."""

    def __init__(self, fused=False, **kwargs):
        if fused in (True, None):
            raise ValueError('TpuBatchNormalization does not support fused=True.')
        super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

    def _cross_replica_average(self, t, num_shards_per_group):
        """Calculates the average value of input tensor across TPU replicas."""
        num_shards = tpu_function.get_tpu_context().number_of_shards
        group_assignment = None
        if num_shards_per_group > 1:
            if num_shards % num_shards_per_group != 0:
                raise ValueError('num_shards: %d mod shards_per_group: %d, should be 0'
                                 % (num_shards, num_shards_per_group))
            num_groups = num_shards // num_shards_per_group
            group_assignment = [[
                x for x in range(num_shards) if x // num_shards_per_group == y
            ] for y in range(num_groups)]
        return tf.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
            num_shards_per_group, t.dtype)

    def _moments(self, inputs, reduction_axes, keep_dims):
        """Compute the mean and variance: it overrides the original _moments."""
        shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
            inputs, reduction_axes, keep_dims=keep_dims)

        num_shards = tpu_function.get_tpu_context().number_of_shards or 1
        if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
            num_shards_per_group = 1
        else:
            num_shards_per_group = max(8, num_shards // 8)
        logging.info('TpuBatchNormalization with num_shards_per_group %s',
                     num_shards_per_group)
        if num_shards_per_group > 1:
            # Compute variance using: Var[X]= E[X^2] - E[X]^2.
            shard_square_of_mean = tf.math.square(shard_mean)
            shard_mean_of_square = shard_variance + shard_square_of_mean
            group_mean = self._cross_replica_average(
                shard_mean, num_shards_per_group)
            group_mean_of_square = self._cross_replica_average(
                shard_mean_of_square, num_shards_per_group)
            group_variance = group_mean_of_square - tf.math.square(group_mean)
            return (group_mean, group_variance)
        else:
            return (shard_mean, shard_variance)


class BatchNormalization(tf.layers.BatchNormalization):
    """Fixed default name of BatchNormalization to match TpuBatchNormalization."""

    def __init__(self, name='tpu_batch_normalization', **kwargs):
        super(BatchNormalization, self).__init__(name=name, **kwargs)


def drop_connect(inputs, is_training, survival_prob):
    """Drop the entire conv with given survival probability."""
    # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    if not is_training:
        return inputs

    # Compute tensor.
    batch_size = tf.shape(inputs)[0]
    random_tensor = survival_prob
    random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no addition compute is
    # needed at test time.
    output = tf.div(inputs, survival_prob) * binary_tensor
    return output


class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.layers.Layer):
    """Wrap keras DepthwiseConv2D to tf.layers."""

    pass


class Conv2D(tf.layers.Conv2D):
    """Wrapper for Conv2D with specialization for fast inference."""

    def _bias_activation(self, outputs):
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _can_run_fast_1x1(self, inputs):
        batch_size = inputs.shape.as_list()[0]
        return (self.data_format == 'channels_first' and
                batch_size == 1 and
                self.kernel_size == (1, 1))

    def _call_fast_1x1(self, inputs):
        # Compute the 1x1 convolution as a matmul.
        inputs_shape = tf.shape(inputs)
        flat_inputs = tf.reshape(inputs, [inputs_shape[1], -1])
        flat_outputs = tf.matmul(
            tf.squeeze(self.kernel),
            flat_inputs,
            transpose_a=True)
        outputs_shape = tf.concat([[1, self.filters], inputs_shape[2:]], axis=0)
        outputs = tf.reshape(flat_outputs, outputs_shape)

        # Handle the bias and activation function.
        return self._bias_activation(outputs)

    def call(self, inputs):
        if self._can_run_fast_1x1(inputs):
            return self._call_fast_1x1(inputs)
        return super(Conv2D, self).call(inputs)
