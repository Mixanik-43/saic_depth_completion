import torch
import tensorflow as tf
import numpy as np
import argparse

from saic_depth_completion.data.datasets.matterport import Matterport
from saic_depth_completion.data.datasets.nyuv2_test import NyuV2Test
from saic_depth_completion.engine.inference import inference
from saic_depth_completion.utils.tensorboard import Tensorboard
from saic_depth_completion.utils.logger import setup_logger
from saic_depth_completion.utils.experiment import setup_experiment
from saic_depth_completion.utils.snapshoter import Snapshoter
from saic_depth_completion.modeling.meta import MetaModel
from saic_depth_completion.modeling.tf.meta import MetaModel as TFMetaModel
from saic_depth_completion.config import get_default_config
from saic_depth_completion.data.collate import default_collate
from saic_depth_completion.metrics import Miss, SSIM, DepthL2Loss, DepthL1Loss, DepthRel

def main():
    parser = argparse.ArgumentParser(description="Some training params.")

    parser.add_argument(
        "--default_cfg", dest="default_cfg", type=str, default="arch0", help="Default config"
    )
    parser.add_argument(
        "--config_file", default="", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--weights", default="", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--save_path", default="", type=str, help="Save path for tf model"
    )

    args = parser.parse_args()

    cfg = get_default_config(args.default_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch_model = MetaModel(cfg, device)
    logger = setup_logger()
    snapshoter = Snapshoter(torch_model, logger=logger)
    snapshoter.load(args.weights)

    tf_device = tf.device("/gpu:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "/cpu:0")
    tf_block = TFMetaModel(cfg, tf_device)
    input_shapes = {"color": (224, 224, 3), "raw_depth": (224, 224, 1), "mask": (224, 224, 1)}
    input = {key: tf.keras.layers.Input(shape, name=f'input_{key}') for key, shape in input_shapes.items()}
    output = tf_block(input)
    tf_model = tf.keras.models.Model(inputs=input,
                                     outputs=output,
                                     name='tf_model')
    tf_model.layers[-1].set_torch_weights(torch_model.state_dict())
    tf_model.save_weights(args.save_path)


if __name__ == "__main__":
    main()