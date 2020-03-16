#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import argparse
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

DEFAULT_OUTPUTS = 8
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT
DEFAULT_KERNEL_HEIGHT = 3
DEFAULT_KERNEL_WIDTH = DEFAULT_KERNEL_HEIGHT
DEFAULT_PADDING = 'same'
DEFAULT_PATH = Path(__file__).parent.joinpath(
    'debug', 'depthwise_conv2d').resolve()


class DepthwiseConv2D(common.OpTestDefaultModel):
    def build_core_model(
            self, K_h, K_w, height, width, output_channels, *,
            padding, strides, **inits):
        assert output_channels % 4 == 0, "# of output channels must be multiple of 4"
        self.input_init = inits['input_init']
        try:
            self.core_model = tf.keras.Sequential(
                name=self.name,
                layers=[
                    tf.keras.layers.DepthwiseConv2D(kernel_size=(K_h, K_w),
                                                    depth_multiplier=1,
                                                    padding=padding,
                                                    strides=strides,
                                                    input_shape=(height, width, output_channels),
                                                    bias_initializer=inits['bias_init'],
                                                    depthwise_initializer=inits['weight_init'])
                ]
            )
            # for layer in self.core_model.layers:
            #     logging.debug(f"WEIGHT DATA SAMPLE:\n{layer.get_weights()[0][1]}")
            #     logging.debug(f"BIAS DATA SAMPLE:\n{layer.get_weights()[1]}")
        except ValueError as e:
            if e.args[0].startswith("Negative dimension size caused by"):
                raise ValueError(
                    "Negative dimension size (Hint: if using 'valid' padding "
                    "verify that the kernel is at least the size of input image)"
                ) from e
            else:
                raise e from None


def main(raw_args=None):
    parser = common.OpTestConvParser(
        conflict_handler='resolve',
        defaults={
            'path': DEFAULT_PATH,
            'inputs': -1,
            'outputs': DEFAULT_OUTPUTS,
            'width': DEFAULT_WIDTH,
            'height': DEFAULT_HEIGHT,
            'padding': DEFAULT_PADDING,
            'kernel_width': DEFAULT_KERNEL_WIDTH,
            'kernel_height': DEFAULT_KERNEL_HEIGHT,
            'inits': {
                'input_init': {
                    'type': common.OpTestInitializers.UNIF,
                    'help': "Initializer for input data distribution."
                },
                'weight_init': {
                    'type': common.OpTestInitializers.UNIF,
                    'help': "Initializer for weight distribution."
                },
                'bias_init': {
                    'type': common.OpTestInitializers.CONST,
                    'help': "Initializer for bias distribution."
                }
            }
        }
    )
    parser.add_argument(
        "-in", "--inputs", type=int, default=-1, choices=[-1],
        help=argparse.SUPPRESS
    )
    parser.add_argument(  # TODO: use the a better parser for this after the conv2d enhancements
        "-st", "--strides", nargs=2, type=int, default=[1, 1],
        help=f"Strides, vertical first",
    )
    args = parser.parse_args(raw_args)
    args.strides = tuple(args.strides)  # TODO: fix this

    model = DepthwiseConv2D('depthwise_conv2d', args.path)
    model.build(args.kernel_height, args.kernel_width,
                args.height, args.width,
                args.outputs,
                padding=args.padding,
                strides=args.strides,
                **args.inits)
    model.run()


if __name__ == "__main__":
    main()
