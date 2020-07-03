import tensorflow as tf
import numpy as np
from .checkpoint_utils import submodel_state_dict
from .dm_lrn import DM_LRN
from .lrn import LRN

from saic_depth_completion.utils import registry
# refactor this to


def preprocess(cfg, batch, dtype=np.float32):
    color = batch["color"].permute(0, 2, 3, 1).detach().numpy().astype(dtype) - np.array(cfg.train.rgb_mean).reshape(1, 1, 1, 3)
    color = color / np.array(cfg.train.rgb_std).reshape(1, 1, 1, 3)

    mask = batch["mask"].permute(0, 2, 3, 1).detach().numpy()
    raw_depth = batch["raw_depth"].permute(0, 2, 3, 1).detach().numpy()
    rd_mask = raw_depth != 0
    raw_depth[rd_mask] = raw_depth[rd_mask] - cfg.train.depth_mean
    raw_depth[rd_mask] = raw_depth[rd_mask] / cfg.train.depth_std
    return [color.astype(dtype), raw_depth.astype(dtype), mask.astype(dtype)]


def postprocess(cfg, pred):
    if cfg.model.predict_log_depth:
        return pred.exp()
    else:
        return pred


class MetaModel(tf.keras.layers.Layer):
    def __init__(self, cfg, device, input_shape=None):
        super(MetaModel, self).__init__()
        self.model = registry.TF_MODELS[cfg.model.arch](cfg.model, input_shape=input_shape)
        self.device = device
        if isinstance(self.device, str):
            self.device = tf.device(self.device)

        self.cfg = cfg

    def call(self, batch, **kwargs):
        with self.device:
            return self.model(batch, **kwargs)

    def preprocess(self, batch):
        with self.device:
            return preprocess(self.cfg, batch)

    def postprocess(self, input):
        return postprocess(self.cfg, input)

    def criterion(self, pred, gt):
        return self.model.criterion(pred, gt)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def set_torch_weights(self, torch_weights):
        self.model.set_torch_weights(submodel_state_dict(torch_weights, 'model.'))
