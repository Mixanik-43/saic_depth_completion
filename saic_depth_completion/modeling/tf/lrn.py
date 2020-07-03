import tensorflow as tf
from saic_depth_completion.modeling.tf import backbone
from saic_depth_completion.modeling.tf.checkpoint_utils import submodel_state_dict, default_set_torch_weights
from saic_depth_completion.modeling.tf.blocks import FusionBlock, CRPBlock, SharedEncoder, Predictor
from saic_depth_completion.utils import registry
from saic_depth_completion.modeling.tf import ops



@registry.TF_MODELS.register("LRN")
class LRN(tf.keras.layers.Layer):
    def __init__(self, model_cfg, input_shape=None):
        super(LRN, self).__init__()

        self.predict_log_depth      = model_cfg.predict_log_depth
        self.activation             = model_cfg.activation
        self.channels               = model_cfg.max_channels
        self.upsample               = model_cfg.upsample
        self.use_crp                = model_cfg.use_crp
        self.concat_mask             = model_cfg.input_mask

        self.stem = tf.keras.Sequential()
        self.stem.add(tf.keras.layers.Conv2D(3, kernel_size=7, padding='same'))
        self.stem.add(tf.keras.layers.BatchNormalization())
        self.stem.add(tf.keras.layers.ReLU())


        self.backbone = registry.TF_BACKBONES[model_cfg.backbone.arch](input_shape=input_shape)
        self.feature_channels = self.backbone.feature_channels


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



        self.__setattr__('convs.0', tf.keras.layers.Conv2D(self.channels // 8, 3, padding='same'))
        self.__setattr__('convs.1', tf.keras.layers.Conv2D(self.channels // 16, 3, padding='same'))
        self.__setattr__('convs.2', tf.keras.layers.Conv2D(self.channels // 16, 3, padding='same'))
        self.__setattr__('convs.3', tf.keras.layers.Conv2D(self.channels // 32, 3, padding='same'))

        self.acts = [
            ops.ACTIVATION_LAYERS[self.activation[0]](self.activation[1][0]),
            ops.ACTIVATION_LAYERS[self.activation[0]](self.activation[1][0]),
            ops.ACTIVATION_LAYERS[self.activation[0]](self.activation[1][0]),
            ops.ACTIVATION_LAYERS[self.activation[0]](self.activation[1][0]),
        ]

        self.predictor = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')


    def call(self, batch, **kwargs):

        color, raw_depth, mask = batch

        if self.concat_mask:
            x = tf.concat([color, raw_depth, mask], axis=-1)
        else:
            x = tf.concat([color, raw_depth], axis=-1)

        x = self.stem(x, **kwargs)

        features = self.backbone(x)[::-1]
        if self.use_crp:
            f1 = self.crp1(self.adapt1(features[0]))
        else:
            f1 = self.adapt1(features[0])
        f2 = self.adapt2(features[1])
        f3 = self.adapt3(features[2])
        f4 = self.adapt4(features[3])

        x = self.fusion_32x16([f2, f1])
        x = self.crp2(x) if self.use_crp else x

        x = self.fusion_16x8([f3, x])
        x = self.crp3(x) if self.use_crp else x

        x = self.fusion_8x4([f4, x])
        x = self.crp4(x) if self.use_crp else x


        x = tf.image.resize(x, size=(x.shape[1] * 2, x.shape[2] * 2), method=self.upsample)
        x = self.acts[0](getattr(self, 'convs.0')(x))
        x = self.acts[1](getattr(self, 'convs.1')(x))

        x = tf.image.resize(x, size=(x.shape[1] * 2, x.shape[2] * 2), method=self.upsample)
        x = self.acts[2](getattr(self, 'convs.2')(x))
        x = self.acts[3](getattr(self, 'convs.3')(x))

        return self.predictor(x)

    def set_torch_weights(self, torch_weights):
        children = []
        children.extend(['stem', 'backbone', 'predictor'])
        children.extend([f'fusion_{x}' for x in ['32x16', '16x8', '8x4']])
        children.extend([f'adapt{x}' for x in [1, 2, 3, 4]])
        children.extend([f'convs.{x}' for x in [0, 1, 2, 3]])

        if self.use_crp:
            children.extend([f'crp{x}' for x in [1, 2, 3, 4]])

        for child_name in children:
            child = getattr(self, child_name)
            child_state_dict = submodel_state_dict(torch_weights, child_name + '.')
            if hasattr(child, 'set_torch_weights'):
                child.set_torch_weights(child_state_dict)
            else:
                default_set_torch_weights(child, child_state_dict)
