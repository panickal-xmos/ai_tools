# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest

from tflite2xcore.model_generation import Configuration

from . import (
    BinarizedSingleOpRunner,
    BConv2dGenericTestModelGenerator,
    LarqSingleOpConverter,
)

from . import (  # pylint: disable=unused-import
    test_output,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class BConv2dInt8TestModelGenerator(BConv2dGenericTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        cfg.setdefault("padding", "valid")
        super()._set_config(cfg)


GENERATOR = BConv2dInt8TestModelGenerator

#  ----------------------------------------------------------------------------
#                                   RUNNERS
#  ----------------------------------------------------------------------------


class BConv2dInt8TestRunner(BinarizedSingleOpRunner):
    def make_lce_converter(self) -> LarqSingleOpConverter:
        return LarqSingleOpConverter(self, self.get_built_model, strip=True)


RUNNER = BConv2dInt8TestRunner

#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture
def bitpacked_outputs() -> bool:
    return False


if __name__ == "__main__":
    pytest.main()
