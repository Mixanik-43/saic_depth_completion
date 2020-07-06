import tensorflow as tf
import numpy as np
from .checkpoint_utils import submodel_state_dict
from .dm_lrn import DM_LRN
from .lrn import LRN

from saic_depth_completion.utils import registry
# refactor this to


def preprocess(cfg, batch, dtype=np.float32):
    return [batch[key].permute(0, 2, 3, 1).detach().numpy().astype(dtype()) for key in ["color", "mask", "raw_depth"]]


class MetaModel(tf.keras.layers.Layer):
    def __init__(self, cfg, device, input_shape=None, fuse_pprocess=False):
        super(MetaModel, self).__init__()
        self.model = registry.TF_MODELS[cfg.model.arch](cfg.model, input_shape=input_shape)
        self.device = device
        if isinstance(self.device, str):
            self.device = tf.device(self.device)

        self.cfg = cfg
        self.rgb_mean = np.array(cfg.train.rgb_mean).reshape(1, 1, 1, 3) * 255.
        self.rgb_std = np.array(cfg.train.rgb_std).reshape(1, 1, 1, 3) * 255.

    def call(self, batch, **kwargs):
        with self.device:
            batch[0] = (batch[0] - self.rgb_mean) / self.rgb_std
            rd_mask = batch[1] != 0
            batch[1][rd_mask] = (batch[1][rd_mask] - self.cfg.train.depth_mean) / self.cfg.train.depth_std
            return self.model(batch, **kwargs)

    def preprocess(self, batch):
        with self.device:
            return preprocess(self.cfg, batch)

    def postprocess(self, input):
        if self.cfg.model.predict_log_depth:
            return tf.math.exp(input):
        else:
            return input
            

    def criterion(self, pred, gt):
        return self.model.criterion(pred, gt)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def set_torch_weights(self, torch_weights):
        self.model.set_torch_weights(submodel_state_dict(torch_weights, 'model.'))
