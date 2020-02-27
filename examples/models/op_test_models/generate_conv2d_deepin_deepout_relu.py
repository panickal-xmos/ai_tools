#!/usr/bin/env python
#
# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
from pathlib import Path
from tflite2xcore.model_generation import utils
import tensorflow as tf
import op_test_models_common as common

DEFAULT_INPUTS = 32
DEFAULT_OUTPUTS = 16
DEFAULT_HEIGHT = 5
DEFAULT_WIDTH = DEFAULT_HEIGHT
DEFAULT_KERNEL_HEIGHT = 3
DEFAULT_KERNEL_WIDTH = DEFAULT_KERNEL_HEIGHT
DEFAULT_PADDING = 'same'
DEFAULT_PATH = Path(__file__).parent.joinpath(
    'debug', 'conv2d_deepin_deepout_relu').resolve()
DEFAULT_NUM_THREADS = 1


class Conv2dDeepinDeepoutRelu(common.DefaultOpTestConvModel):
    def build_core_model(self, *args, **kwargs):
        input_channels = args[4]
        assert input_channels % 32 == 0, "# of input channels must be multiple of 32"
        super().build_core_model(*args, **kwargs)


def main(path=DEFAULT_PATH, *,
         num_threads=DEFAULT_NUM_THREADS,
         input_channels=DEFAULT_INPUTS,
         output_channels=DEFAULT_OUTPUTS,
         height=DEFAULT_HEIGHT,
         width=DEFAULT_WIDTH,
         K_h=DEFAULT_KERNEL_HEIGHT,
         K_w=DEFAULT_KERNEL_WIDTH,
         padding=DEFAULT_PADDING,
         inits=common.DEFAULT_INITS):
    kwargs = {
        'name': 'conv2d_deepin_deepout_relu',
        'path': path if path else DEFAULT_PATH
    }
    model = Conv2dDeepinDeepoutRelu(**kwargs)
    model.run(num_threads=num_threads,
              input_channels=input_channels,
              output_channels=output_channels,
              height=height,
              width=width,
              K_h=K_h,
              K_w=K_w,
              padding=padding,
              inits=inits)


if __name__ == "__main__":
    parser = common.OpTestConvParser(defaults={
        'path': DEFAULT_PATH,
        'inputs': DEFAULT_INPUTS,
        'outputs': DEFAULT_OUTPUTS,
        'width': DEFAULT_WIDTH,
        'height': DEFAULT_HEIGHT,
        'padding': DEFAULT_PADDING,
        'kernel_width': DEFAULT_KERNEL_WIDTH,
        'kernel_height': DEFAULT_KERNEL_HEIGHT,
        'inits': {
            'input_init': common.OpTestInitializers.UNIF,
            'bias_init': common.OpTestInitializers.CONST,
            'weight_init': common.OpTestInitializers.UNIF}
    })
    parser.add_argument(
        '-par', '--par_num_threads', type=int, default=DEFAULT_NUM_THREADS,
        help='Number of parallel threads for xcore.ai optimization.')
    args = parser.parse_args()
    utils.set_verbosity(args.verbose)
    utils.set_gpu_usage(False, args.verbose)

    main(path=args.path,
         num_threads=args.par_num_threads,
         input_channels=args.inputs,
         output_channels=args.outputs,
         K_h=args.kernel_height,
         K_w=args.kernel_width,
         height=args.height,
         width=args.width,
         padding=args.padding,
         inits=args.inits)
