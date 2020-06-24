import tensorflow as tf
import numpy as np
from .checkpoint_utils import submodel_state_dict
from .dm_lrn import DM_LRN

from saic_depth_completion.utils import registry
# refactor this to
class MetaModel(tf.keras.layers.Layer):
    def __init__(self, cfg, device):
        super(MetaModel, self).__init__()
        self.model = registry.TF_MODELS[cfg.model.arch](cfg.model)
        self.device = device
        if isinstance(self.device, str):
            self.device = tf.device(self.device)

        self.rgb_mean = np.array(cfg.train.rgb_mean).reshape(1, 1, 1, 3)
        self.rgb_std = np.array(cfg.train.rgb_std).reshape(1, 1, 1, 3)

        self.depth_mean = cfg.train.depth_mean
        self.depth_std = cfg.train.depth_std

    def call(self, batch):
        with self.device:
            return self.model(batch)

    def preprocess(self, batch):
        with self.device:
            batch["color"] = batch["color"] - self.rgb_mean
            batch["color"] = batch["color"] / self.rgb_std

            mask = batch["raw_depth"] != 0
            batch["raw_depth"][mask] = batch["raw_depth"][mask] - self.depth_mean
            batch["raw_depth"][mask] = batch["raw_depth"][mask] / self.depth_std
            return batch

    def postprocess(self, input):
        return self.model.postprocess(input)
    def criterion(self, pred, gt):
        return self.model.criterion(pred, gt)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def set_torch_weights(self, torch_weights):
        self.model.set_torch_weights(submodel_state_dict(torch_weights, 'model.'))
