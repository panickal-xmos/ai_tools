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

		x = tf.keras.layers.Conv2D(16,(3, 3), strides=(2,2),padding='same')(inputs)
		x_skip = x

		x = tf.keras.layers.DepthwiseConv2D(16,padding='same')(x)
		x = tf.keras.layers.Conv2D(16,1)(x)
		x = tf.keras.layers.Add()([x,x_skip])

		x = tf.keras.layers.Conv2D(96,(1, 1), padding='same')(x)
		x = tf.keras.layers.DepthwiseConv2D(3,strides=(2,2),padding='same')(x)
		x = tf.keras.layers.Conv2D(16,1)(x)
		x_skip = x
		
		x = tf.keras.layers.Conv2D(96,(1, 1), padding='same')(x)
		x = tf.keras.layers.DepthwiseConv2D(3,padding='same')(x)
		x = tf.keras.layers.Conv2D(16,1)(x)

		x = tf.keras.layers.Add()([x,x_skip])

		x = tf.keras.layers.Flatten()(x)
		outputs = tf.keras.layers.Dense(10, activation='relu')(x)

		model = tf.keras.Model(inputs=inputs,outputs=outputs)

		# model = tf.keras.Sequential()
		# model.add(tf.keras.Input(shape=(10,10, 16),))#80,40,20,10,5
		# model.add(tf.keras.layers.Conv2D(96,(1, 1), strides=(1,1),activation='relu',padding='same'))
		# # model.add(tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2,2),activation='relu',padding='same'))
		# model.add(tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1,1),activation='relu',padding='same'))
		# model.add(tf.keras.layers.Conv2D(16,(1, 1), strides=(1,1),activation='relu',padding='same'))
		# model.add(tf.keras.layers.Flatten())
		# model.add(tf.keras.layers.Dense(10, activation='relu'))

		return model

GENERATOR = OperatorSplittingModelGenerator


#  ----------------------------------------------------------------------------


RUNNER = OperatorSplittingTestRunner

#  ----------------------------------------------------------------------------
#                                   TESTS
#  ----------------------------------------------------------------------------


@pytest.mark.skip_on_device
def test_converted_model(operator_splitting_model: XCOREModel, experimental_xformer2: bool) -> None:
	opcode_cnt = operator_splitting_model.count_operator_codes()
	assert opcode_cnt[OperatorCode(BuiltinOpCodes.CONCATENATION)] == 2
    
if __name__ == "__main__":
    pytest.main()
