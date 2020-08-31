# Copyright (c) 2020, XMOS Ltd, All rights reserved

import pytest  # type: ignore

from .test_conv2d_shallowin import converted_op_code, Conv2dShallowinTestModelGenerator
from . import (
    ExplicitPaddingMixin,
    test_output as _test_output,
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


#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------

# TODO: fix this when issue #187 is fixed
def test_output(run, request):  # type: ignore
    if request.node.name in ("test_output[CONFIGS[1]]", "test_output[CONFIGS[6]]"):
        request.applymarker(pytest.mark.xfail(run=False))
    _test_output(run, request)


if __name__ == "__main__":
    pytest.main()
