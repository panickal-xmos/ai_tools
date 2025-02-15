# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import os
import pytest
import tensorflow as tf
from typing import Optional, Tuple, Any
from tensorflow.python.keras.utils import (  # pylint: disable=no-name-in-module
    data_utils,
)

from tflite2xcore.model_generation import Configuration

from . import IntegrationTestModelGenerator
from . import (  # pylint: disable=unused-import
    test_output,
)

#  ----------------------------------------------------------------------------
#                                   HELPERS
#  ----------------------------------------------------------------------------

BASE_WEIGHT_URL = (
    "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/"
)


def _MobileNet_safe(
    input_shape: Optional[Tuple[int, int, int]] = None,
    alpha: float = 1.0,
    depth_multiplier: int = 1,
    dropout: float = 1e-3,
    include_top: bool = True,
    weights: Optional[str] = "imagenet",
    input_tensor: Optional[tf.Tensor] = None,
    pooling: Optional[str] = None,
    classes: int = 1000,
    *args: Any,
    **kwargs: Any,
) -> tf.keras.Model:
    if weights == "imagenet" and (not include_top or classes == 1000):
        input_shape = input_shape or (224, 224, 3)
        rows = input_shape[0]
        if (
            input_shape[0] == input_shape[1]
            and depth_multiplier == 1
            and alpha in [0.25, 0.50, 0.75, 1.0]
            and rows in [128, 160, 192, 224]
        ):
            if alpha == 1.0:
                alpha_text = "1_0"
            elif alpha == 0.75:
                alpha_text = "7_5"
            elif alpha == 0.50:
                alpha_text = "5_0"
            else:
                alpha_text = "2_5"

            model_name = f"mobilenet_{alpha_text}_{rows}_tf"
            if not include_top:
                model_name += "_no_top"
            model_name += ".h5"
            weight_url = BASE_WEIGHT_URL + model_name
            weights = data_utils.get_file(
                model_name,
                weight_url,
                cache_subdir=f"/tmp/.keras/{os.getpid()}/",
            )

    return tf.keras.applications.MobileNet(
        input_shape,
        alpha,
        depth_multiplier,
        dropout,
        include_top,
        weights,
        pooling,
        classes,
        *args,
        **kwargs,
    )


def MobileNet(*args: Any, **kwargs: Any) -> tf.keras.Model:
    """ Wrapper for tf.keras.applications.MobileNet to work around h5 multiprocess issues. """
    try:
        return tf.keras.applications.MobileNet(*args, **kwargs)
    except (KeyError, OSError) as e:
        if e.args[0].startswith("Unable to open"):
            return _MobileNet_safe(*args, **kwargs)
        else:
            raise


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class MobileNetV1ModelGenerator(IntegrationTestModelGenerator):
    def _set_config(self, cfg: Configuration) -> None:
        self._config["input_size"] = cfg.pop("input_size")
        self._config["alpha"] = cfg.pop("alpha")
        super()._set_config(cfg)

    def _build_core_model(self) -> tf.keras.Model:
        input_size = self._config["input_size"]
        return MobileNet(
            input_shape=(input_size, input_size, 3), alpha=self._config["alpha"]
        )


GENERATOR = MobileNetV1ModelGenerator

#  ----------------------------------------------------------------------------
#                                   CONFIGS
#  ----------------------------------------------------------------------------


CONFIGS = {
    "default": {0: {"input_size": 128, "alpha": 0.25, "skip_on_device": True}},
}


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture
def abs_output_tolerance() -> None:
    return


# TODO: try removing this when global average pool is improved
@pytest.fixture
def implicit_tolerance_margin() -> float:
    return 0.15


if __name__ == "__main__":
    pytest.main()
