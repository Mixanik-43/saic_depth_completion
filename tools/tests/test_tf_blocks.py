import torch
import numpy as np
from saic_depth_completion.modeling.tf.blocks import *
from saic_depth_completion.modeling.tf.ops import *
from saic_depth_completion.modeling.tf.backbone.efficientnet import EfficientNetB0
from saic_depth_completion.modeling.tf.backbone.efficientnet_lite import EfficientNetLiteB0
from saic_depth_completion.modeling.tf.meta import MetaModel as TFMetaModel
from saic_depth_completion.modeling.tf.dm_lrn import DM_LRN
from saic_depth_completion.utils.logger import setup_logger
from saic_depth_completion.utils.snapshoter import Snapshoter
from saic_depth_completion.modeling.meta import MetaModel
from saic_depth_completion.config import get_default_config
from saic_depth_completion.modeling.tf.checkpoint_utils import default_set_torch_weights
from termcolor import colored


def load_torch_model():
    cfg = get_default_config('DM-LRN')
    cfg.merge_from_file('configs/dm_lrn/DM-LRN_efficientnet-b0_camila.yaml')
    cfg.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = MetaModel(cfg, device).eval()
    logger = setup_logger()
    snapshoter = Snapshoter(torch_model, logger=logger)
    snapshoter.load('weights/arch2_efficientnet-b0_camila.pth')
    return torch_model


def run_tests(raise_errors=True, **kwargs):
    torch_model = load_torch_model()
    for test_func in globals():
        if test_func.startswith('test_') and callable(eval(test_func)):
            if raise_errors:
                eval(test_func)(torch_model, **kwargs)
            else:
                try:
                    eval(test_func)(torch_model, **kwargs)
                except Exception as e:
                    print(e)


def run_block_test(torch_block, tf_block, input_shapes, test_name, max_mape=1, torch_input_shapes=None):
    np.random.seed(0)
    if torch_input_shapes is None:
        torch_input_shapes = input_shapes
    if len(input_shapes) == 1:
        input = tf.keras.layers.Input(input_shapes[0], name=f'input')
    elif isinstance(input_shapes, dict):
        input = {key: tf.keras.layers.Input(shape, name=f'input_{key}') for key, shape in input_shapes.items()}
    else:
        input = [tf.keras.layers.Input(input_shapes[i], name=f'input{i + 1}') for i in range(len(input_shapes))]
    output = tf_block(input)
    tf_model = tf.keras.models.Model(inputs=input,
                                     outputs=output,
                                     name='tf_model')
    if hasattr(tf_model.layers[-1], 'set_torch_weights'):
        tf_model.layers[-1].set_torch_weights(torch_block.state_dict())
    else:
        default_set_torch_weights(tf_block, torch_block.state_dict())

    if isinstance(torch_input_shapes, dict):
        keys_list = ['color', 'raw_depth', 'mask']
        assert set(torch_input_shapes.keys()) == set(keys_list)
        input_tensors = {key: np.random.normal(size=(1, shape[2], shape[0], shape[1])) for key, shape in torch_input_shapes.items()}
        tf_multiple_out = tf_model.predict_step([input_tensors[key].transpose(0, 2, 3, 1) for key in keys_list])
        torch_multiple_out = torch_block({key: torch.Tensor(input_tensor) for key, input_tensor in input_tensors.items()})
    else:
        input_tensors = [np.random.normal(size=(1, shape[2], shape[0], shape[1])) for shape in input_shapes]
        if len(input_tensors) == 1:
            tf_multiple_out = tf_model.predict_step(input_tensors[0].transpose(0, 2, 3, 1))
        else:
            tf_multiple_out = tf_model.predict_step([input_tensor.transpose(0, 2, 3, 1) for input_tensor in input_tensors])
        torch_multiple_out = torch_block(*[torch.Tensor(input_tensor) for input_tensor in input_tensors])
    if isinstance(torch_multiple_out, torch.Tensor):
        assert isinstance(tf_multiple_out, tf.Tensor)
        torch_multiple_out = [torch_multiple_out]
        tf_multiple_out = [tf_multiple_out]

    for one_tf_out, one_torch_out in zip(tf_multiple_out, torch_multiple_out):
        mape = (np.abs(one_torch_out.permute(0, 2, 3, 1).detach().numpy() - one_tf_out.numpy())
                / (np.abs(one_torch_out.permute(0, 2, 3, 1).detach().numpy()) + np.abs(one_tf_out.numpy()) + 1e-5)
                ).mean() * 100
        if mape > max_mape:
            raise Exception('{} test status: '.format(test_name) + colored('ERROR', 'red') + '\tmape ={}'.format(mape))

    print('{} test status: '.format(test_name), colored('OK', 'green'))


def test_crp(torch_model, **kwargs):
    torch_block = torch_model.model.crp1
    tf_block = CRPBlock(256, 256)
    input_shapes = [(10, 10, 256)]
    test_name = 'CRP block'
    run_block_test(torch_block, tf_block, input_shapes, test_name, **kwargs)


def test_fusion(torch_model, **kwargs):
    torch_block = torch_model.model.fusion_16x8
    tf_block = FusionBlock(64, 128)
    input_shapes = [(16, 16, 64), (8, 8, 128)]
    test_name = 'Fusion block'
    run_block_test(torch_block, tf_block, input_shapes, test_name, **kwargs)


def test_shared_encoder(torch_model, **kwargs):
    torch_block = torch_model.model.mask_encoder
    tf_block = SharedEncoder(
            out_channels=(256, 128, 64, 32, 32, 16),
            scales=(32, 16, 8, 4, 2, 1),
            upsample='bilinear',
            activation=("LeakyReLU", [0.2] ),
            kernel_size=3
        )
    input_shapes = [(160, 192, 1)]
    test_name = 'Shared enoder'
    run_block_test(torch_block, tf_block, input_shapes, test_name, **kwargs)


def test_spade(torch_model, **kwargs):
    torch_block = torch_model.model.modulation32.modulation1
    tf_block = SPADE(256, 256, kernel_size=3, upsample='bilinear')
    input_shapes = [(8, 8, 256), (16, 16, 256)]
    test_name = 'SPADE'
    run_block_test(torch_block, tf_block, input_shapes, test_name, **kwargs)


def test_adaptive_spade(torch_model, **kwargs):
    torch_block = torch_model.model.modulation32
    tf_block = AdaptiveBlock(
            256, 256, 256,
            modulation='SPADE', activation=("LeakyReLU", [0.2] ),
            upsample='bilinear'
        )
    input_shapes = [(8, 8, 256), (16, 16, 256)]
    test_name = 'Adaptive block with SPADE modulation'
    run_block_test(torch_block, tf_block, input_shapes, test_name, **kwargs)


def test_efficientnet(torch_model, **kwargs):
    torch_block = torch_model.model.backbone
    tf_block = EfficientNetB0(weights=None, include_top=False)
    input_shapes = [(224, 224, 3)]
    test_name = 'Efficientnet encoder'
    run_block_test(torch_block, tf_block, input_shapes, test_name, **kwargs)


def test_efficientnet_lite(torch_model, **kwargs):
    cfg = get_default_config('DM-LRN')
    cfg.merge_from_file('configs/dm_lrn/DM-LRN_tf_efficientnet-l0_emma.yaml')
    cfg.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = MetaModel(cfg, device).eval()
    torch_block = torch_model.model.backbone
    tf_block = EfficientNetLiteB0()
    input_shapes = [(320, 256, 3)]
    test_name = 'Efficientnet-lite encoder'
    run_block_test(torch_block, tf_block, input_shapes, test_name, **kwargs)

def test_dm_lrn(torch_model, **kwargs):
    torch_block = torch_model
    cfg = get_default_config('DM-LRN')
    cfg.merge_from_file('configs/dm_lrn/DM-LRN_efficientnet-b0_camila.yaml')
    cfg.freeze()
    device = tf.device("/gpu:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "/cpu:0")
    tf_block = TFMetaModel(cfg, device)
    input_shapes = [(224, 224, 3), (224, 224, 1), (224, 224, 1)]
    torch_input_shapes = {"color": (224, 224, 3), "raw_depth": (224, 224, 1), "mask": (224, 224, 1)}
    test_name = 'DM_LRN'
    run_block_test(torch_block, tf_block, input_shapes, test_name, torch_input_shapes=torch_input_shapes, **kwargs)


def test_lrn(torch_model, **kwargs):
    cfg = get_default_config('LRN')
    cfg.merge_from_file('configs/lrn/LRN_tf_efficientnet-l0_gabriella.yaml')
    cfg.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = MetaModel(cfg, device).eval()
    torch_block = torch_model

    device = tf.device("/gpu:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "/cpu:0")
    tf_block = TFMetaModel(cfg, device, input_shape=(224, 224, 3))
    input_shapes = [(224, 224, 3), (224, 224, 1), (224, 224, 1)]
    torch_input_shapes = {"color": (224, 224, 3), "raw_depth": (224, 224, 1), "mask": (224, 224, 1)}
    test_name = 'LRN'
    run_block_test(torch_block, tf_block, input_shapes, test_name, torch_input_shapes=torch_input_shapes, **kwargs)

if __name__ == '__main__':
    run_tests(raise_errors=True)
    # run_tests(raise_errors=False, max_mape=1e-2)
