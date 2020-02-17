# Copyright (c) 2019, XMOS Ltd, All rights reserved

import pytest
import itertools

from tflite2xcore.transformation_passes import ReplaceAveragePool2DPass

from .model_builders import build_avgpool


from .test_ReplaceAveragePool2D2x2Pass import (
    input_height, input_width, input_channels, input_shape,
    MATCHING_POOL_SIZE, MATCHING_STRIDES
)

MATCHING_PADDING = ['VALID']


@pytest.fixture()
def trf_pass_safe():
    return ReplaceAveragePool2DPass(safe_mode=True)


@pytest.fixture()
def trf_pass_unsafe():
    return ReplaceAveragePool2DPass(safe_mode=False)


@pytest.fixture(params=MATCHING_PADDING)
def padding(request):
    return request.param


@pytest.fixture()
def model(input_shape, padding):
    return build_avgpool(input_shape=input_shape, padding=padding,
                         pool_size=MATCHING_POOL_SIZE, strides=MATCHING_STRIDES)


def test_matching(trf_pass_unsafe, model):
    assert trf_pass_unsafe.match(model.subgraphs[0].operators[-1])


def test_non_matching(trf_pass_safe, model):
    assert not trf_pass_safe.match(model.subgraphs[0].operators[-1])


if __name__ == "__main__":
    pytest.main()