import tensorflow as tf
from saic_depth_completion.modeling.tf import ops
from .checkpoint_utils import submodel_state_dict
import numpy as np


class CRPBlock(tf.keras.layers.Layer):
    def conv1x1(self, in_planes, out_planes, stride=1, bias=False):
        return tf.keras.layers.Conv2D(filters=out_planes, kernel_size=(1, 1), strides=stride, use_bias=bias)

    def __init__(self, in_planes, out_planes, n_stages=4):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(
                self, '{}_{}'.format(i + 1, 'crp'),
                self.conv1x1(
                    in_planes if (i == 0) else out_planes,
                    out_planes, stride=1, bias=False
                )
            )
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=5, strides=1, padding="same")

    def call(self, x, **kwargs):
        top = x
        for i in range(self.n_stages):
            #             print(top.shape)
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'crp'))(top)
            x = top + x
        return x

    def set_torch_weights(self, torch_weights):
        tf_weights = [torch_weights[k].permute(2, 3, 1, 0) for k in sorted(torch_weights.keys())]
        self.set_weights(tf_weights)


class FusionBlock(tf.keras.layers.Layer):
    def __init__(
            self, hidden_dim, small_planes, activation=("ReLU", []), upsample="bilinear",
    ):
        super(FusionBlock, self).__init__()
        self.act      = ops.ACTIVATION_LAYERS[activation[0]](*activation[1])
        self.upsample = upsample
        self.conv1    = tf.keras.layers.Conv2D(filters=hidden_dim, kernel_size=(1, 1), use_bias=True)
        self.conv2    = tf.keras.layers.Conv2D(filters=hidden_dim, kernel_size=(1, 1), use_bias=True)

    def call(self, inputs, **kwargs):
        input1, input2 = inputs
        x1 = self.conv1(input1)
        x2 = self.conv2(input2)
        x2 = tf.compat.v1.image.resize(x2, size=x1.shape[1:3], method=self.upsample, align_corners=True)
        return self.act(x1 + x2)

    def set_torch_weights(self, torch_weights):
        tf_weights = []
        tf_biases = []
        for k in sorted(torch_weights.keys()):
            if k.endswith('bias'):
                tf_biases.append(torch_weights[k])
            else:
                tf_weights.append(torch_weights[k].permute(2, 3, 1, 0))

        tf_params = [param for layer_params in zip (tf_weights, tf_biases) for param in layer_params]

        self.set_weights(tf_params)


class SharedEncoder(tf.keras.layers.Layer):
    def __init__(
            self, out_channels, scales, in_channels=1, kernel_size=3, upsample="bilinear", activation=("ReLU", [])
    ):
        super(SharedEncoder, self).__init__()
        self.scales = scales
        self.upsample = upsample
        self.feature_extractor = tf.keras.Sequential()
        self.feature_extractor.add(tf.keras.layers.Conv2D(32, kernel_size, padding='same'))
        self.feature_extractor.add(ops.ACTIVATION_LAYERS[activation[0]](*activation[1]))
        self.feature_extractor.add(tf.keras.layers.Conv2D(64, kernel_size, padding='same'))
        self.feature_extractor.add(ops.ACTIVATION_LAYERS[activation[0]](*activation[1]))

        self.predictors = []
        for oup in out_channels:
            predictor = tf.keras.Sequential()
            predictor.add(tf.keras.layers.Conv2D(oup, 3, padding='valid'))
            predictor.add(ops.ACTIVATION_LAYERS[activation[0]](*activation[1]))
            self.predictors.append(predictor)

    def call(self, x, **kwargs):
        features = self.feature_extractor(x)
        res = []
        for it, scale in enumerate(self.scales):
            size = np.array(features.shape[1:3]) // scale
            features_scaled = tf.image.resize(features, size=size, method=self.upsample)
            res.append(self.predictors[it](features_scaled))
        return tuple(res)

    def set_torch_weights(self, torch_weights):
        tf_weights = []
        tf_biases = []
        for k in sorted(torch_weights.keys()):
            if k.endswith('bias'):
                tf_biases.append(torch_weights[k])
            else:
                tf_weights.append(torch_weights[k].permute(2, 3, 1, 0))

        tf_params = [param for layer_params in zip (tf_weights, tf_biases) for param in layer_params]

        self.set_weights(tf_params)


class AdaptiveBlock(tf.keras.layers.Layer):
    def __init__(
            self, x_in_ch, x_out_ch, y_ch, modulation="spade", activation=("ReLU", []), upsample='bilinear'
    ):
        super(AdaptiveBlock, self).__init__()

        x_hidden_ch = min(x_in_ch, x_out_ch)
        self.learned_res = x_in_ch != x_out_ch

        if self.learned_res:
            self.residual = tf.keras.layers.Conv2D(x_out_ch, kernel_size=1, padding='valid', use_bias=False)

        self.modulation1 = ops.MODULATION_LAYERS[modulation](x_ch=x_in_ch, y_ch=y_ch, upsample=upsample)
        self.act1        = ops.ACTIVATION_LAYERS[activation[0]](*activation[1])
        self.conv1       = tf.keras.layers.Conv2D(x_hidden_ch, kernel_size=3, padding='same', use_bias=True)
        self.modulation2 = ops.MODULATION_LAYERS[modulation](x_ch=x_hidden_ch, y_ch=y_ch, upsample=upsample)
        self.act2        = ops.ACTIVATION_LAYERS[activation[0]](*activation[1])
        self.conv2       = tf.keras.layers.Conv2D(x_out_ch, kernel_size=3, padding='same', use_bias=True)

    def call(self, inputs, **kwargs):
        x, skip = inputs
        if self.learned_res:
            res = self.residual(x)
        else:
            res = x

        x = self.modulation1([x, skip])
        x = self.act1(x)
        x = self.conv1(x)
        x = self.modulation2([x, skip])
        x = self.act2(x)
        x = self.conv2(x)

        return x + res


    def set_torch_weights(self, torch_weights):
        self.conv1.set_weights([torch_weights['conv1.weight'].permute(2, 3, 1, 0),
                                torch_weights['conv1.bias']])
        self.conv2.set_weights([torch_weights['conv2.weight'].permute(2, 3, 1, 0),
                                torch_weights['conv2.bias']])
        if self.learned_res:
            self.residual.set_weights([torch_weights['residual.weight'].permute(2, 3, 1, 0)])
        self.modulation1.set_torch_weights(submodel_state_dict(torch_weights, 'modulation1.'))
        self.modulation2.set_torch_weights(submodel_state_dict(torch_weights, 'modulation2.'))


class Predictor(tf.keras.layers.Layer):
    def __init__(self):
        super(Predictor, self).__init__()

        self.predictor = tf.keras.Sequential()
        self.predictor.add(tf.keras.layers.DepthwiseConv2D(kernel_size=1))
        self.predictor.add(tf.keras.layers.Conv2D(1, kernel_size=3, padding='same'))

    def call(self, inputs, **kwargs):
        return self.predictor(inputs)


    def set_torch_weights(self, torch_weights):
        self.predictor.set_weights([torch_weights['0.weight'].permute(2, 3, 0, 1),
                                    torch_weights['0.bias'],
                                    torch_weights['1.weight'].permute(2, 3, 1, 0),
                                    torch_weights['1.bias']])
