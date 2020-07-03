import argparse
import tensorflow as tf
import tqdm
from saic_depth_completion.data.datasets.tf_datagen import tf_datagen
from saic_depth_completion.data.datasets.matterport import Matterport


def main():
    parser = argparse.ArgumentParser(description="Some training params.")
    parser.add_argument(
        "--tf_path", type=str, help="Path to load tensorflow saved_model"
    )
    parser.add_argument(
        "--tflite_path", type=str, help="Path to save tflite model"
    )
    parser.add_argument(
        "--dtype", default='int8', type=str, help="unt8/float16 quantization type"
    )


    args = parser.parse_args()
    converter = tf.lite.TFLiteConverter.from_saved_model(args.tf_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if args.dtype == 'float16':
        converter.target_spec.supported_types=[tf.float16]
    elif args.dtype == 'int8':
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8
        converter.representative_dataset = lambda: tf_datagen(tqdm.tqdm(Matterport(split='val')), 1000)
    tflite_model = converter.convert()
    open(args.tflite_path, "wb").write(tflite_model)


if __name__ == '__main__':
    main()
