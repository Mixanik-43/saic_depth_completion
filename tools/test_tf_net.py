import torch
import tensorflow as tf

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
        "--save_dir", default="", type=str, help="Save dir for predictions"
    )
    parser.add_argument(
        "--weights", default="", type=str, metavar="FILE", help="path to config file"
    )
    args = parser.parse_args()

    cfg = get_default_config(args.default_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    logger = setup_logger()

    device = tf.device("/gpu:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "/cpu:0")
    tf_block = TFMetaModel(cfg, device)
    input_shapes = {"color": (224, 224, 3), "raw_depth": (224, 224, 1), "mask": (224, 224, 1)}
    input = {key: tf.keras.layers.Input(shape, name=f'input_{key}') for key, shape in input_shapes.items()}
    output = tf_block(input)
    model = tf.keras.models.Model(inputs=input,
                                  outputs=output,
                                  name='tf_model')
    model.load_weights(args.weights)

    metrics = {
        'mse': DepthL2Loss(),
        'mae': DepthL1Loss(),
        'd105': Miss(1.05),
        'd110': Miss(1.10),
        'd125_1': Miss(1.25),
        'd125_2': Miss(1.25**2),
        'd125_3': Miss(1.25**3),
        'rel': DepthRel(),
        'ssim': SSIM(),
    }

    test_datasets = {
        "test_matterport": Matterport(split="test"),
        # "official_nyu_test": NyuV2Test(split="official_test")
    }
    test_loaders = {
        k: torch.utils.data.DataLoader(
            dataset=v,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=default_collate
        )
        for k, v in test_datasets.items()
    }

    inference(
        model,
        test_loaders,
        save_dir=args.save_dir,
        logger=logger,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()