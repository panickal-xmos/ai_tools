#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

from generate_avgpool2d import (
    DEFAULT_INPUTS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_POOL_SIZE,
    DEFAULT_PADDING,
    DEFAULT_STRIDES,
)

DEFAULT_PATH = Path(__file__).parent.joinpath('debug', 'maxpool2d').resolve()


class MaxPool2d(common.DefaultOpTestModel):
    def build_core_model(self, height, width, input_channels, *, pool_size,
                         strides, padding):
        assert input_channels % 4 == 0, "# of input channels must be multiple of 4"
        if padding.lower() == 'same':
            assert (height % 2 == 0 and width % 2 == 0 and pool_size[0] == 2
                    and pool_size[1] == 2 and strides[0] == 2
                    and strides[1] == 2
                    ), "same padding is only allowed for the common 2x2 case"
        else:
            assert padding.lower(
            ) == 'valid', f"invalid padding mode '{padding}'"

        self.core_model = tf.keras.Sequential(
            name=self.name,
            layers=[
                tf.keras.layers.MaxPool2D(pool_size=pool_size,
                                          strides=strides,
                                          padding=padding,
                                          input_shape=(height, width,
                                                       input_channels))
            ])


def main(path=DEFAULT_PATH,
         *,
         input_channels=DEFAULT_INPUTS,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH,
         pool_size=DEFAULT_POOL_SIZE,
         strides=DEFAULT_STRIDES,
         padding=DEFAULT_PADDING):
    model = MaxPool2d('maxpool2d', Path(path))
    model.build(height,
                width,
                input_channels,
                pool_size=pool_size,
                strides=strides,
                padding=padding)
    model.run()


if __name__ == "__main__":
    parser = common.OpTestPoolStridesParser(
        defaults={
            "path": DEFAULT_PATH,
            "inputs": DEFAULT_INPUTS,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "padding": DEFAULT_PADDING,
            "strides": DEFAULT_STRIDES,
            "pool_size": DEFAULT_POOL_SIZE
        })
    args = parser.parse_args()

    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    strides_pool = common.strides_pool_arg_handler(args)

    main(path=args.path,
         input_channels=args.inputs,
         height=args.height,
         width=args.width,
         padding=args.padding,
         **strides_pool)
