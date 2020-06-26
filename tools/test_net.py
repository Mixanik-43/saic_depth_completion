import torch
import tensorflow as tf

import argparse

from saic_depth_completion.data.datasets.matterport import Matterport
from saic_depth_completion.data.datasets.nyuv2_test import NyuV2Test
from saic_depth_completion.engine.inference import inference, tf_inference, tflite_inference
from saic_depth_completion.utils.tensorboard import Tensorboard
from saic_depth_completion.utils.logger import setup_logger
from saic_depth_completion.utils.experiment import setup_experiment
from saic_depth_completion.utils.snapshoter import Snapshoter
from saic_depth_completion.modeling.meta import MetaModel
from saic_depth_completion.modeling.tf.meta import preprocess, postprocess
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
        "--saved_model", default="", type=str, metavar="FILE", help="path to pytorch state_dict or  tensorflow saved model"
    )
    parser.add_argument(
        "--framework", default='pytorch', type=str, help="'pytorch', 'tf' or 'tflite'"
    )

    args = parser.parse_args()



    logger = setup_logger()
    frameworks_list = ['pytorch', 'tf', 'tflite']
    assert args.framework in frameworks_list, 'Supported frameworks are {}, got {}'.format(frameworks_list, args.framework)
    cfg = get_default_config(args.default_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if args.framework == "pytorch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MetaModel(cfg, device)
        snapshoter = Snapshoter(model, logger=logger)
        snapshoter.load(args.saved_model)
        inference_procedure = inference
        preprocess_func = model.preprocess
        postprocess_func = model.postprocess
    elif args.framework == "tf":
        model = tf.saved_model.load(args.saved_model)

        inference_procedure = tf_inference
        preprocess_func = lambda batch: preprocess(cfg, batch)
        postprocess_func = lambda pred: postprocess(cfg, pred)

    else:
        model = tf.lite.Interpreter(model_path=args.saved_model)
        model.allocate_tensors()
        inference_procedure = tflite_inference
        preprocess_func = lambda batch: preprocess(cfg, batch)
        postprocess_func = lambda pred: postprocess(cfg, pred)

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
        # "official_nyu_test": NyuV2Test(split="official_test"),
        #
        # # first
        # "1gr10pv1pd": NyuV2Test(split="1gr10pv1pd"),
        # "1gr10pv2pd": NyuV2Test(split="1gr10pv2pd"),
        # "1gr10pv5pd": NyuV2Test(split="1gr10pv5pd"),
        #
        # "1gr25pv1pd": NyuV2Test(split="1gr25pv1pd"),
        # "1gr25pv2pd": NyuV2Test(split="1gr25pv2pd"),
        # "1gr25pv5pd": NyuV2Test(split="1gr25pv5pd"),
        #
        # "1gr40pv1pd": NyuV2Test(split="1gr40pv1pd"),
        # "1gr40pv2pd": NyuV2Test(split="1gr40pv2pd"),
        # "1gr40pv5pd": NyuV2Test(split="1gr40pv5pd"),
        #
        # #second
        # "4gr10pv1pd": NyuV2Test(split="4gr10pv1pd"),
        # "4gr10pv2pd": NyuV2Test(split="4gr10pv2pd"),
        # "4gr10pv5pd": NyuV2Test(split="4gr10pv5pd"),
        #
        # "4gr25pv1pd": NyuV2Test(split="4gr25pv1pd"),
        # "4gr25pv2pd": NyuV2Test(split="4gr25pv2pd"),
        # "4gr25pv5pd": NyuV2Test(split="4gr25pv5pd"),
        #
        # "4gr40pv1pd": NyuV2Test(split="4gr40pv1pd"),
        # "4gr40pv2pd": NyuV2Test(split="4gr40pv2pd"),
        # "4gr40pv5pd": NyuV2Test(split="4gr40pv5pd"),
        #
        # # third
        # "8gr10pv1pd": NyuV2Test(split="8gr10pv1pd"),
        # "8gr10pv2pd": NyuV2Test(split="8gr10pv2pd"),
        # "8gr10pv5pd": NyuV2Test(split="8gr10pv5pd"),
        #
        # "8gr25pv1pd": NyuV2Test(split="8gr25pv1pd"),
        # "8gr25pv2pd": NyuV2Test(split="8gr25pv2pd"),
        # "8gr25pv5pd": NyuV2Test(split="8gr25pv5pd"),
        #
        # "8gr40pv1pd": NyuV2Test(split="8gr40pv1pd"),
        # "8gr40pv2pd": NyuV2Test(split="8gr40pv2pd"),
        # "8gr40pv5pd": NyuV2Test(split="8gr40pv5pd"),

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

    inference_procedure(
        model,
        test_loaders,
        preprocess_func=preprocess_func,
        postprocess_func=postprocess_func,
        save_dir=args.save_dir,
        logger=logger,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
