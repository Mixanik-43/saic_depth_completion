import tensorflow as tf
from saic_depth_completion.modeling.tf import backbone
from saic_depth_completion.modeling.tf.checkpoint_utils import submodel_state_dict, default_set_torch_weights
from saic_depth_completion.modeling.tf.blocks import AdaptiveBlock, FusionBlock, CRPBlock, SharedEncoder, Predictor
from saic_depth_completion.utils import registry
from saic_depth_completion.modeling.tf import ops
from saic_depth_completion.metrics import LOSSES


@registry.TF_MODELS.register("DM-LRN")
class DM_LRN(tf.keras.layers.Layer):
    def __init__(self, model_cfg, input_shape=None):
        super(DM_LRN, self).__init__()

        self.stem = tf.keras.Sequential()
        self.stem.add(tf.keras.layers.Conv2D(3, kernel_size=7, padding='same'))
        self.stem.add(tf.keras.layers.BatchNormalization())
        self.stem.add(tf.keras.layers.ReLU())

        self.backbone               = registry.TF_BACKBONES[model_cfg.backbone.arch](input_shape=input_shape)
        self.feature_channels       = self.backbone.feature_channels

        self.predict_log_depth      = model_cfg.predict_log_depth
        self.activation             = model_cfg.activation
        if self.activation[0] == "LeakyReLU":
            self.activation = (self.activation[0], [self.activation[1][0]])
        self.modulation             = model_cfg.modulation
        self.channels               = model_cfg.max_channels
        self.upsample               = model_cfg.upsample
        self.use_crp                = model_cfg.use_crp
        self.mask_encoder_ksize     = model_cfg.mask_encoder_ksize

        self.modulation32 = AdaptiveBlock(
            self.channels, self.channels, self.channels,
            modulation=self.modulation, activation=self.activation,
            upsample=self.upsample
        )
        self.modulation16 = AdaptiveBlock(
            self.channels // 2, self.channels // 2, self.channels // 2,
            modulation=self.modulation, activation=self.activation,
            upsample=self.upsample
        )
        self.modulation8  = AdaptiveBlock(
            self.channels // 4, self.channels // 4, self.channels // 4,
            modulation=self.modulation, activation=self.activation,
            upsample=self.upsample
        )
        self.modulation4  = AdaptiveBlock(
            self.channels // 8, self.channels // 8, self.channels // 8,
            modulation=self.modulation, activation=self.activation,
            upsample=self.upsample
        )

        self.modulation4_1 = AdaptiveBlock(
            self.channels // 8, self.channels // 16, self.channels // 8,
            modulation=self.modulation, activation=self.activation,
            upsample=self.upsample
        )
        self.modulation4_2 = AdaptiveBlock(
            self.channels // 16, self.channels // 16, self.channels // 16,
            modulation=self.modulation, activation=self.activation,
            upsample=self.upsample
        )



        self.mask_encoder = SharedEncoder(
            out_channels=(
                self.channels, self.channels // 2, self.channels // 4,
                self.channels // 8, self.channels // 8, self.channels // 16
            ),
            scales=(32, 16, 8, 4, 2, 1),
            upsample=self.upsample,
            activation=self.activation,
            kernel_size=self.mask_encoder_ksize
        )

        self.fusion_32x16 = FusionBlock(self.channels // 2, self.channels, upsample=self.upsample)
        self.fusion_16x8  = FusionBlock(self.channels // 4, self.channels // 2, upsample=self.upsample)
        self.fusion_8x4   = FusionBlock(self.channels // 8, self.channels // 4, upsample=self.upsample)

        self.adapt1 = tf.keras.layers.Conv2D(self.channels, kernel_size=1, use_bias=False)
        self.adapt2 = tf.keras.layers.Conv2D(self.channels // 2, kernel_size=1, use_bias=False)
        self.adapt3 = tf.keras.layers.Conv2D(self.channels // 4, kernel_size=1, use_bias=False)
        self.adapt4 = tf.keras.layers.Conv2D(self.channels // 8, kernel_size=1, use_bias=False)

        if self.use_crp:
            self.crp1 = CRPBlock(self.channels, self.channels)
            self.crp2 = CRPBlock(self.channels // 2, self.channels // 2)
            self.crp3 = CRPBlock(self.channels // 4, self.channels // 4)
            self.crp4 = CRPBlock(self.channels // 8, self.channels // 8)


        self.predictor = Predictor()

        if not self.predict_log_depth:
            self.act = ops.ACTIVATION_LAYERS[self.activation[0]](*self.activation[1])

    def call(self, batch, **kwargs):
        color, raw_depth, mask = batch

        x = tf.concat([color, raw_depth], axis=-1)
        mask = mask + 1.0
        x = self.stem(x, **kwargs)

        features = self.backbone(x, **kwargs)[::-1]
        if self.use_crp:
            f1 = self.crp1(self.adapt1(features[0], **kwargs), **kwargs)
        else:
            f1 = self.adapt1(features[0], **kwargs)
        f2 = self.adapt2(features[1], **kwargs)
        f3 = self.adapt3(features[2], **kwargs)
        f4 = self.adapt4(features[3], **kwargs)

        mask_features = self.mask_encoder(mask, **kwargs)

        x = self.modulation32([f1, mask_features[0]], **kwargs)
        x = self.fusion_32x16([f2, x], **kwargs)
        x = self.crp2(x, **kwargs) if self.use_crp else x

        x = self.modulation16([x, mask_features[1]], **kwargs)
        x = self.fusion_16x8([f3, x], **kwargs)
        x = self.crp3(x, **kwargs) if self.use_crp else x

        x = self.modulation8([x, mask_features[2]], **kwargs)
        x = self.fusion_8x4([f4, x], **kwargs)
        x = self.crp4(x, **kwargs) if self.use_crp else x

        x = self.modulation4([x, mask_features[3]], **kwargs)

        x = tf.image.resize(x, size=(x.shape[1] * 2, x.shape[2] * 2), method=self.upsample)
        x = self.modulation4_1([x, mask_features[4]], **kwargs)
        x = tf.image.resize(x, size=(x.shape[1] * 2, x.shape[2] * 2), method=self.upsample)
        x = self.modulation4_2([x, mask_features[5]], **kwargs)

        if not self.predict_log_depth: return self.act(self.predictor(x, **kwargs))
        return self.predictor(x, **kwargs)

    def set_torch_weights(self, torch_weights):
        children = []
        children.extend(['stem', 'backbone', 'mask_encoder', 'predictor'])
        children.extend([f'modulation{x}' for x in [32, 16, 8, 4, '4_1', '4_2']])
        children.extend([f'fusion_{x}' for x in ['32x16', '16x8', '8x4']])
        children.extend([f'adapt{x}' for x in [1, 2, 3, 4]])
        if self.use_crp:
            children.extend([f'crp{x}' for x in [1, 2, 3, 4]])

        for child_name in children:
            child = getattr(self, child_name)
            child_state_dict = submodel_state_dict(torch_weights, child_name + '.')
            if hasattr(child, 'set_torch_weights'):
                child.set_torch_weights(child_state_dict)
            else:
                default_set_torch_weights(child, child_state_dict)
