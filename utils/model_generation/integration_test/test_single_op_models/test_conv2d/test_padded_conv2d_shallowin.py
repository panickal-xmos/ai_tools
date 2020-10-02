# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore

from .test_conv2d_shallowin import Conv2dShallowinTestModelGenerator
from . import ExplicitPaddingMixin
from .test_conv2d_shallowin import converted_op_code  # pylint: disable=unused-import
from . import (  # pylint: disable=unused-import
    test_output,
    test_converted_single_op_model,
    test_idempotence,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class PaddedConv2dShallowinTestModelGenerator(
    ExplicitPaddingMixin, Conv2dShallowinTestModelGenerator
):
    pass


GENERATOR = PaddedConv2dShallowinTestModelGenerator


if __name__ == "__main__":
    pytest.main()
