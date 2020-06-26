import argparse
import tensorflow as tf



def main():
    parser = argparse.ArgumentParser(description="Some training params.")
    parser.add_argument(
        "--tf_path", type=str, help="Path to load tensorflow saved_model"
    )
    parser.add_argument(
        "--tflite_path", type=str, help="Path to save tflite model"
    )
    parser.add_argument(
        "--quantize", default=False, type=bool, help="Whether to quantize model"
    )
    parser.add_argument(
        "--float16", default=False, type=bool, help="Whether to use float16 weights"
    )


    args = parser.parse_args()
    converter = tf.lite.TFLiteConverter.from_saved_model(args.tf_path)
    if args.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if args.float16:
            converter.target_spec.supported_types=[tf.float16]
    tflite_model = converter.convert()
    open(args.tflite_path, "wb").write(tflite_model)


if __name__ == '__main__':
    main()
