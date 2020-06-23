from functools import partial
import tensorflow as tf
from .spade import SPADE

from saic_depth_completion.utils.registry import Registry

MODULATION_LAYERS = Registry()
NORM_LAYERS = Registry()
ACTIVATION_LAYERS = Registry()

ACTIVATION_LAYERS["ReLU"] = tf.keras.layers.ReLU
ACTIVATION_LAYERS["LeakyReLU"] = tf.keras.layers.LeakyReLU

MODULATION_LAYERS["SPADE"] = SPADE

NORM_LAYERS["BatchNorm2d"] = tf.keras.layers.BatchNormalization
# NORM_LAYERS["FrozenBatchNorm2d"] = FrozenBatchNorm2d