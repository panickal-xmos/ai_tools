# Copyright (c) 2020, XMOS Ltd, All rights reserved

import tensorflow as tf  # type: ignore
from abc import abstractmethod
from typing import Callable, Optional, Tuple

from .. import (
    ChannelPreservingOpTestModelGenerator,
    test_output,
    test_converted_single_op_model,
)


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class ActivationOPTestModelGenerator(ChannelPreservingOpTestModelGenerator):
    def _check_channel_count(self, channels):
        pass


class LUTActivationOpTestModelGenerator(ActivationOPTestModelGenerator):
    @property
    @abstractmethod
    def act_fun(self) -> Callable[[tf.Tensor], tf.Tensor]:
        raise NotImplementedError()

    def _op_layer(
        self, *, input_shape: Optional[Tuple[int, int, int]] = None
    ) -> tf.keras.layers.Layer:
        kwargs = {"input_shape": input_shape} if input_shape else {}
        return tf.keras.layers.Lambda(self.act_fun, **kwargs)
