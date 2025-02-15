# Copyright 2019-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import os
import logging
import tempfile
import subprocess

from pathlib import Path
from typing import Optional, Union, Any

from tflite2xcore.pass_manager import PassManager
from tflite2xcore.xcore_model import XCOREModel
from tflite2xcore import transformation_passes as passes


XFORMER2_PATH = (
    Path(__file__).resolve().parents[3]
    / "experimental"
    / "xformer"
    / "bazel-bin"
    / "xcore-opt"
)


class CleanupManager(PassManager):
    def __init__(self, model: Optional[XCOREModel] = None, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.register_pass(passes.EliminateDeadOperatorsPass())
        self.register_pass(passes.EliminateDeadTensorsPass())
        self.register_pass(passes.EliminateDeadBuffersPass())


class BasicCanonicalizationManager(PassManager):
    def __init__(
        self,
        model: Optional[XCOREModel] = None,
        *,
        remove_float_interface: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.register_pass(passes.CanonicalizeEmptyBuffersPass())

        if remove_float_interface:
            self.register_pass(passes.CanonicalizeQuantizedInputPass())
            self.register_pass(passes.CanonicalizeQuantizedOutputPass())

        # start with a round of constant folding
        self.register_pass(passes.ConstantPropagationPass())

        # canonicalize single pixel convolution
        # 1x1 convolutions acting on 1x1 inputs (without padding) map trivially
        # to a fully connected, so we canoncalize these to a builtin FULLY_CONNECTED
        self.register_pass(passes.CanonicalizeSinglePixelConv2DPass())

        # canonicalize reshape
        # this ensures that RESHAPE has a single input tensor
        # (no dynamic reshapes are currently supported)
        self.register_pass(passes.CanonicalizeReshapePass())
        self.register_passes(CleanupManager())  # this is needed

        # canonicalize fully connected shapes
        # the FC implementation flattens implicitly, so we remove RESHAPES before
        # and after FULLY_CONNECTED ops
        self.register_pass(passes.RemovePrecedingReshapePass())
        self.register_pass(passes.RemoveSubsequentReshapePass())

        # canonicalize single channel depthwise convolutions
        # depthwise convolutions with one input channel map trivially to ordinary
        # convolutions with `depth_multiplier` output channels
        self.register_pass(passes.CanonicalizeSingleinDepthwiseConv2DPass())
        self.register_pass(passes.LegalizeSingleinConv2DPass())

        # canonicalize quantize ops
        # two consecutive quantize ops have no effect besides adding error
        self.register_pass(passes.RemoveRedundantInt8RequantizationPass())
        # the TFLM interpreter does not support newer versions of quantized
        # so we downgrade where safe
        self.register_pass(passes.LegalizeQuantizeVersionPass())

        # need to cleanup after the intial canonicalization
        self.register_passes(CleanupManager())


class WordAlignmentCanonicalizationManager(PassManager):
    def __init__(self, model: Optional[XCOREModel] = None, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)

        # canonicalize word alignment of inputs
        # we insert explicit channel-wise padding to ensure that
        # input channel counts to convolutions are divisible by 4
        # (this is currently required by our kernels)
        self.register_pass(passes.CanonicalizeConv2DInputChannels())


class ActivationLoweringManager(PassManager):
    def __init__(self, model: Optional[XCOREModel] = None, experimental_xformer2: bool = False, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)

        # first we match ops and replace them
        if not experimental_xformer2:
            self.register_pass(passes.ReplaceReLUPass())
            self.register_pass(passes.ReplaceReLU6Pass())
            self.register_pass(passes.ReplaceTanhPass())
            self.register_pass(passes.ReplaceLogisticPass())
            # second we legalize the op by calculating the LUT
            self.register_pass(passes.LegalizeXCLookupTablePass())


class PoolingLoweringManager(PassManager):
    def __init__(self, model: Optional[XCOREModel] = None, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)

        self.register_pass(passes.ReplaceMaxPool2D2x2Pass())
        self.register_pass(passes.ReplaceMaxPool2DPass())
        self.register_pass(passes.ReplaceAveragePool2D2x2Pass())
        self.register_pass(passes.ReplaceAveragePool2DPass())
        self.register_pass(passes.ReplaceGlobalAveragePool2DPass())


class ParametricOperatorLoweringManager(PassManager):
    def __init__(
        self,
        model: Optional[XCOREModel] = None,
        *,
        experimental_xformer2: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)

        # first we match ops and replace them
        if not experimental_xformer2:
            self.register_pass(passes.ReplaceFullyConnectedPass())
        self.register_pass(passes.Replace1x1Conv2dPass())
        self.register_pass(passes.ReplaceShallowinConv2dPass())
        self.register_pass(passes.ReplaceDepthwiseConv2dPass())
        self.register_pass(passes.ReplaceDeepConv2dPass())

        # second we legalize them by reshaping weight/bias tensors,
        # calculating parameters specific to our kernels,
        # and populating the custom options
        if not experimental_xformer2:
            self.register_pass(passes.LegalizeXCFullyConnectedPass())
        self.register_pass(passes.LegalizeXC1x1ConvPass())
        self.register_pass(passes.LegalizeXCShallowinConvPass())
        self.register_pass(passes.LegalizeXCDepthwiseConvPass())
        self.register_pass(passes.LegalizeXCDeepConvPass())


class PaddingOptimizationManager(PassManager):
    def __init__(
        self,
        model: Optional[XCOREModel] = None,
        *,
        remove_input_alignment_pad: bool,
        experimental_xformer2: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)

        # canonicalize by ensuring that spatial and other dims are decoupled
        # first fuse consecutive PAD ops
        # (injected by word alignment, bconv2d padding legalization, etc.)
        self.register_pass(passes.FuseConsecutivePadsPass())
        # second split batch/channel-wise padding from spatial padding
        self.register_pass(passes.SplitPaddingPass())

        # we optimize the convolutions by fusing it with spatial padding
        self.register_pass(passes.FuseConv2dPaddingPass())
        if remove_input_alignment_pad:
            # remove word alignment padding on the input
            self.register_pass(passes.RemovePaddingInputPass())
        # replace with optimized implementation where possible
        self.register_pass(passes.ReplacePadPass())

        # Fuse back any remaining PAD operators
        self.register_pass(passes.FuseConsecutivePadsPass())


class ParallelizationManager(PassManager):
    def __init__(
        self, model: Optional[XCOREModel] = None, *, num_threads: int, **kwargs: Any
    ) -> None:
        super().__init__(model, **kwargs)

        self.register_pass(
            passes.ParallelizeFullyConnectedPass(num_threads=num_threads)
        )
        self.register_pass(passes.ParallelizeConv2dPass(num_threads=num_threads))
        self.register_pass(
            passes.ParallelizeDepthwiseConv2dPass(num_threads=num_threads)
        )
        self.register_pass(passes.ParallelizePooling2DPass(num_threads=num_threads))
        self.register_pass(
            passes.ParallelizeGlobalAveragePool2DPass(num_threads=num_threads)
        )

        self.register_pass(passes.ParallelizeBConv2dBinPass(num_threads=num_threads))
        self.register_pass(passes.ParallelizeBConv2dInt8Pass(num_threads=num_threads))
        self.register_pass(passes.ParallelizeAddPass(num_threads=num_threads))
        self.register_pass(passes.ParallelizeLUTPass(num_threads=num_threads))
        # pass_mgr.register_pass(passes.ParallelizeRequant16To8Pass(num_threads=num_threads))  # intentionally disabled

        # NOTE: scratch memory passes must be registered after parallelization passes
        # TODO: it would be better if scratch memory calculation could be decoupled from parallelization
        self.register_pass(passes.ScratchMemoryConv2dPass())
        self.register_pass(passes.ScratchMemoryConv2d1x1Pass())
        self.register_pass(passes.ScratchMemoryDepthwiseConv2dPass())
        self.register_pass(passes.ScratchMemoryFullyConnectedPass())


class BinarizedOperatorLoweringManager(PassManager):
    def __init__(self, model: Optional[XCOREModel] = None, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)

        # map LceQuantize to our bsign op
        self.register_pass(passes.ReplaceLceQuantizePass())

        # match bconv2d ops and replace them
        self.register_pass(passes.ReplaceBconv2DBitpackedDeepInPass())
        self.register_pass(passes.ReplaceBconv2DBitpackedPass())
        self.register_pass(passes.ReplaceBconv2DInt8DeepInDeepOutPass())
        self.register_pass(passes.ReplaceBconv2DInt8Pass())

        # we legalize the padding by injecting an explicit PAD where needed
        self.register_pass(passes.LegalizeXCBconv2DPaddingPass())

        # legalize the parameter tensors and custom options
        self.register_pass(passes.LegalizeBconv2dBitpackedDeepInPass())
        self.register_pass(passes.LegalizeBconv2dBitpackedPass())
        self.register_pass(passes.LegalizeBconv2dInt8DeepInDeepOutPass())
        self.register_pass(passes.LegalizeBconv2dInt8Pass())


class ExternalMemoryOptimizationManager(PassManager):
    def __init__(self, model: Optional[XCOREModel] = None, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.register_pass(passes.InsertExternalMemoryFetchPass())


class FinalizationManager(PassManager):
    def __init__(
        self,
        model: Optional[XCOREModel] = None,
        *,
        cleanup: bool,
        minification: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        if cleanup:
            self.register_passes(CleanupManager())

        # TODO: this is actually a canonicalization pass
        self.register_pass(passes.LegalizeOperatorOutputTensorNamePass())

        self.register_pass(passes.FloatingPointWarningPass())

        self.register_pass(passes.UnifyEmptyBuffersPass())
        self.register_pass(passes.EliminateDeadBuffersPass())

        if minification:
            self.register_pass(passes.MinifyQuantInfoPass())
            self.register_pass(passes.MinifyTensorNamesPass())


def optimize_for_xcore(
    model: XCOREModel,
    *,
    cleanup: bool = True,
    minification: bool = False,
    num_threads: Optional[int] = None,
    intermediates_path: Optional[Union[str, Path]] = None,
    remove_input_alignment_pad: bool = False,
    remove_float_interface: bool = False,
    external_memory: bool = False,
    experimental_xformer2: bool = False,
) -> XCOREModel:
    num_threads = num_threads or 1
    intermediates_path = Path(intermediates_path) if intermediates_path else None

    pass_mgr = PassManager(model, keep_intermediates=bool(intermediates_path))

    # canonicalization
    pass_mgr.register_passes(
        BasicCanonicalizationManager(remove_float_interface=remove_float_interface)
    )
    pass_mgr.register_passes(WordAlignmentCanonicalizationManager())

    # lowering to the xcore ops
    pass_mgr.register_passes(ActivationLoweringManager(experimental_xformer2=experimental_xformer2))
    pass_mgr.register_passes(PoolingLoweringManager())
    pass_mgr.register_passes(BinarizedOperatorLoweringManager())

    if experimental_xformer2:
        try:
            pass_mgr.run_passes()
            model.sanity_check()
        finally:
            if intermediates_path:
                pass_mgr.save_intermediates(intermediates_path / "pre_xformer2")
                intermediates_path /= "post_xformer2"

        with tempfile.TemporaryDirectory(suffix=str(os.getpid())) as dirname:
            input_path = Path(dirname) / "input.tflite"
            model.write_flatbuffer(input_path)

            output_path = Path(dirname) / "output.tflite"
            cmd = [str(XFORMER2_PATH), str(input_path), "-o", str(output_path)]
            p = subprocess.run(cmd, capture_output=True, check=True)
            logging.debug(p.stdout)

            model = XCOREModel.read_flatbuffer(output_path)

        pass_mgr = PassManager(model, keep_intermediates=bool(intermediates_path))

    pass_mgr.register_passes(
        ParametricOperatorLoweringManager(experimental_xformer2=experimental_xformer2)
    )

    # TODO: finish these and find a manager for them:
    pass_mgr.register_pass(passes.ReplaceAddPass())

    # optimizations on xcore ops
    pass_mgr.register_passes(
        PaddingOptimizationManager(
            remove_input_alignment_pad=remove_input_alignment_pad, experimental_xformer2=experimental_xformer2
        )
    )
    pass_mgr.register_passes(ParallelizationManager(num_threads=num_threads))
    if external_memory:
        pass_mgr.register_passes(ExternalMemoryOptimizationManager())

    # finalize (cleanup, minification, renaming, etc.)
    pass_mgr.register_passes(
        FinalizationManager(minification=minification, cleanup=cleanup)
    )

    try:
        pass_mgr.run_passes()
        model.sanity_check()
    finally:
        if intermediates_path:
            pass_mgr.save_intermediates(intermediates_path)

    model.description = model.description + " + XMOS optimized."

    return model


def convert(
    tflite_input_path: Union[str, Path],
    tflite_output_path: Union[str, Path],
    **kwargs: Any,
) -> None:
    model = XCOREModel.read_flatbuffer(tflite_input_path)
    model = optimize_for_xcore(model, **kwargs)
    model.write_flatbuffer(tflite_output_path)
