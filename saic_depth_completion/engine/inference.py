import os
import time
import datetime
import torch
from tqdm import tqdm
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from saic_depth_completion.utils.meter import AggregatedMeter
from saic_depth_completion.utils.meter import Statistics as LossMeter
from saic_depth_completion.utils import visualize


def inference(
        model, test_loaders, metrics, save_dir="", logger=None,
        preprocess_func=lambda x: x, postprocess_func=lambda x: x
):

    model.eval()
    metrics_meter = AggregatedMeter(metrics, maxlen=20)
    for subset, loader in test_loaders.items():
        idx = 0
        logger.info(
            "Inference: subset -- {}. Total number of batches: {}.".format(subset, len(loader))
        )

        metrics_meter.reset()
        # loop over dataset
        for batch in tqdm(loader):
            batch = preprocess_func(batch)
            pred = model(batch)

            with torch.no_grad():
                post_pred = postprocess_func(pred)
                if save_dir:
                    B = batch["color"].shape[0]
                    for it in range(B):
                        fig = visualize.figure(
                            batch["color"][it], batch["raw_depth"][it],
                            batch["mask"][it], batch["gt_depth"][it],
                            post_pred[it], close=True
                        )
                        fig.savefig(
                            os.path.join(save_dir, "result_{}.png".format(idx)), dpi=fig.dpi
                        )

                        idx += 1

                metrics_meter.update(post_pred, batch["gt_depth"])

        state = "Inference: subset -- {} | ".format(subset)
        logger.info(state + metrics_meter.suffix)


def tf_inference(model, test_loaders, metrics, save_dir="", logger=None,
                 preprocess_func=lambda x: x, postprocess_func=lambda x: x):
    metrics_meter = AggregatedMeter(metrics, maxlen=20)
    for subset, loader in test_loaders.items():
        idx = 0
        logger.info(
            "Inference: subset -- {}. Total number of batches: {}.".format(subset, len(loader))
        )

        metrics_meter.reset()
        # loop over dataset
        for batch in tqdm(loader):
            batch_for_inference = preprocess_func(batch)
            pred = model.signatures["serving_default"](**{'input_{}'.format(i): tf.constant(input_tensor, dtype=tf.float32) for i, input_tensor in enumerate(batch_for_inference)})
            with torch.no_grad():
                pred = torch.Tensor(pred['meta_model'].numpy()).permute(0, 3, 1, 2)
                post_pred = postprocess_func(pred)
                if save_dir:
                    B = batch["color"].shape[0]
                    for it in range(B):
                        fig = visualize.figure(
                            batch["color"][it], batch["raw_depth"][it],
                            batch["mask"][it], batch["gt_depth"][it],
                            post_pred[it], close=True
                        )
                        fig.savefig(
                            os.path.join(save_dir, "result_{}.png".format(idx)), dpi=fig.dpi
                        )

                        idx += 1

                metrics_meter.update(post_pred, batch["gt_depth"])

        state = "Inference: subset -- {} | ".format(subset)
        logger.info(state + metrics_meter.suffix)


def tflite_inference(model, test_loaders, metrics, save_dir="", logger=None,
                     preprocess_func=lambda x: x, postprocess_func=lambda x: x):
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    metrics_meter = AggregatedMeter(metrics, maxlen=20)
    for subset, loader in test_loaders.items():
        idx = 0
        logger.info(
            "Inference: subset -- {}. Total number of batches: {}.".format(subset, len(loader))
        )

        metrics_meter.reset()
        # loop over dataset
        for batch in tqdm(loader):
            batch_for_inference = preprocess_func(batch)
            for input_id in range(len(batch_for_inference)):
                model.set_tensor(input_details[input_id]['index'], batch_for_inference[input_id])
            model.invoke()
            pred = model.get_tensor(output_details[0]['index'])

            with torch.no_grad():
                pred = torch.Tensor(pred).permute(0, 3, 1, 2)
                post_pred = postprocess_func(pred)
                if save_dir:
                    B = batch["color"].shape[0]
                    for it in range(B):
                        fig = visualize.figure(
                            batch["color"][it], batch["raw_depth"][it],
                            batch["mask"][it], batch["gt_depth"][it],
                            post_pred[it], close=True
                        )
                        fig.savefig(
                            os.path.join(save_dir, "result_{}.png".format(idx)), dpi=fig.dpi
                        )

                        idx += 1

                metrics_meter.update(post_pred, batch["gt_depth"])

        state = "Inference: subset -- {} | ".format(subset)
        logger.info(state + metrics_meter.suffix)


