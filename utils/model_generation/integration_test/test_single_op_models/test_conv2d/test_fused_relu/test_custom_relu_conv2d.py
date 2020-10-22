# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest

from tflite2xcore.xcore_schema import XCOREModel, BuiltinOpCodes

from ..test_conv2d import Conv2dTestModelGenerator
from . import FusedCustomReluMixin
from ..test_conv2d import converted_op_code  # pylint: disable=unused-import
from . import (  # pylint: disable=unused-import
    test_output,
    # test_converted_single_op_model,  # TODO: enable
    test_idempotence,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class CustomReluConv2dTestModelGenerator(
    FusedCustomReluMixin, Conv2dTestModelGenerator
):
    pass


GENERATOR = CustomReluConv2dTestModelGenerator

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture  # type: ignore
def output_tolerance() -> None:
    return


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


def test_reference_model_regression(reference_model: XCOREModel) -> None:
    operators = reference_model.subgraphs[0].operators

    opcodes = [op.operator_code.code for op in operators]
    expected_opcodes = [
        BuiltinOpCodes.CONV_2D,
        BuiltinOpCodes.QUANTIZE,
        BuiltinOpCodes.QUANTIZE,
        BuiltinOpCodes.MINIMUM,
        BuiltinOpCodes.QUANTIZE,
        BuiltinOpCodes.MAXIMUM,
    ]
    assert opcodes == expected_opcodes


if __name__ == "__main__":
    pytest.main()
