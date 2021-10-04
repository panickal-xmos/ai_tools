# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import pytest
import tensorflow as tf
from typing import Type
from tensorflow.keras import layers,models

from tflite2xcore.xcore_schema import (
    XCOREModel,
    XCOREOpCodes,
    BuiltinOpCodes,
    OperatorCode,
    TensorType,
)

from . import (  # pylint: disable=unused-import
    test_output_operator_splitting,
)
from .. import IntegrationTestModelGenerator, OperatorSplittingTestRunner


#  ----------------------------------------------------------------------------
#                                   GENERATORS
#  ----------------------------------------------------------------------------


class OperatorSplittingModelGenerator(IntegrationTestModelGenerator):
	def _build_core_model(self) -> tf.keras.Model:
		inputs = tf.keras.Input(shape=(160,160, 3),)
		x0 = tf.keras.layers.Conv2D(16,(3, 3), strides=(2,2),activation='relu',padding='same')(inputs)
		x_skip = x0
		x1 = tf.keras.layers.DepthwiseConv2D(3,padding='same')(x0)
		x2 = tf.keras.layers.Conv2D(16,1)(x1)
		x3 = tf.keras.layers.Add()([x2,x_skip])
		x4 = tf.keras.layers.Flatten()(x3)
		outputs = tf.keras.layers.Dense(10, activation='relu')(x4)

		model = tf.keras.Model(inputs=inputs,outputs=outputs)

		return model

GENERATOR = OperatorSplittingModelGenerator


#  ----------------------------------------------------------------------------


RUNNER = OperatorSplittingTestRunner

#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


@pytest.mark.skip_on_device
def test_converted_model(xcore_model: XCOREModel, experimental_xformer2: bool) -> None:
	opcode_cnt = xcore_model.count_operator_codes()
	assert opcode_cnt[OperatorCode(BuiltinOpCodes.CONCATENATION)] == 1
    
if __name__ == "__main__":
    pytest.main()
