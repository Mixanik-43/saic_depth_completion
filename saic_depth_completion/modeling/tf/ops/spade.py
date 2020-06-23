import torch
from torch import nn
from torch.nn import functional as F

import tensorflow as tf

class SPADE(tf.keras.layers.Layer):

    def __init__(self, x_ch, y_ch, kernel_size=3, upsample='nearest'):
        super(SPADE, self).__init__()
        self.eps = 1e-5
        assert upsample in ['nearest', 'bilinear']
        self.upsample = upsample

        self.gamma = tf.keras.layers.Conv2D(x_ch, kernel_size, padding='same', use_bias=False)
        self.beta = tf.keras.layers.Conv2D(x_ch, kernel_size, padding='same', use_bias=False)

        # we assume that there is a some distribution at each cell in tensor
        # => we need to compute stats over batch only
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, inputs, **kwargs):
        x, y = inputs

        y = tf.image.resize(y, size=x.shape[1:3], method=self.upsample)
        x_normalized = self.bn(x)

        # do not need relu !!! We should be able to sub from signal
        gamma = self.gamma(y)
        beta = self.beta(y)
        return (1 + gamma) * x_normalized + beta

    def set_torch_weights(self, torch_weights):
        self.beta.set_weights([torch_weights['beta.weight'].permute(2, 3, 1, 0)])
        self.gamma.set_weights([torch_weights['gamma.weight'].permute(2, 3, 1, 0)])
        self.bn.set_weights([torch_weights['bn.running_mean'], torch_weights['bn.running_var']])
