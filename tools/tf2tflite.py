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

    args = parser.parse_args()
    converter = tf.lite.TFLiteConverter.from_saved_model(args.tf_path)
    if args.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open(args.tflite_path, "wb").write(tflite_model)


if __name__ == '__main__':
    main()