import tensorflow as tf
import numpy as np
from .checkpoint_utils import submodel_state_dict
from .dm_lrn import DM_LRN

from saic_depth_completion.utils import registry
# refactor this to


def preprocess(cfg, batch):
    batch["color"] = batch["color"].permute(0, 2, 3, 1).detach().numpy() - np.array(cfg.train.rgb_mean).reshape(1, 1, 1, 3)
    batch["color"] = batch["color"] / np.array(cfg.train.rgb_std).reshape(1, 1, 1, 3)

    mask = batch["raw_depth"].permute(0, 2, 3, 1).detach().numpy() != 0
    batch["raw_depth"][mask] = batch["raw_depth"][mask].permute(0, 2, 3, 1).detach().numpy() - cfg.train.depth_mean
    batch["raw_depth"][mask] = batch["raw_depth"][mask] / cfg.train.depth_std
    return batch['color'], batch['raw_depth'], batch['mask']


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
