import os
import time
import datetime
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from saic_depth_completion.utils.meter import AggregatedMeter
from saic_depth_completion.utils.meter import Statistics as LossMeter
from saic_depth_completion.utils import visualize


def inference(
        model, test_loaders, metrics, save_dir="", logger=None
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
            batch = model.preprocess(batch)
            pred = model(batch)

            with torch.no_grad():
                post_pred = model.postprocess(pred)
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


def tf_inference(model, test_loaders, metrics, save_dir="", logger=None):
    metrics_meter = AggregatedMeter(metrics, maxlen=20)
    for subset, loader in test_loaders.items():
        idx = 0
        logger.info(
            "Inference: subset -- {}. Total number of batches: {}.".format(subset, len(loader))
        )

        metrics_meter.reset()
        # loop over dataset
        for batch in tqdm(loader):
            batch_for_inference = {k: batch[k].permute(0, 2, 3, 1).detach().numpy() for k in ["color", "raw_depth", "mask"]}
            batch_for_inference = model.layers[-1].preprocess(batch_for_inference)
            pred = model(batch_for_inference, training=False)

            with torch.no_grad():
                pred = torch.Tensor(pred.numpy()).permute(0, 3, 1, 2)
                post_pred = model.layers[-1].postprocess(pred)
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
