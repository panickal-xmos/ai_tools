# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
import numpy as np

from tflite2xcore.transformation_passes.transformation_passes import (
    OperatorMatchingPass,
    QuantizedOperatorMatchingPass,
    TensorMatchingPass,
)
from tflite2xcore.xcore_model import Operator, Tensor
from tflite2xcore.xcore_schema import (
    OperatorCode,
    XCOREOpCodes,
    TensorType,
    BuiltinOpCodes,
    Subgraph,
)
from .pooling_passes import (
    ReplaceAveragePool2DPass,
    ReplaceGlobalAveragePool2DPass,
)

def find_largest_address_in_persistent_buffer(subgraph: Subgraph) -> int:
    largest_address = 0
    for operator in subgraph.operators:
        if 'prev_data_address' in operator.custom_options:
            prev_data_address = operator.custom_options.get("prev_data_address")
            prev_data_size = operator.custom_options.get("prev_data_size")
            if (prev_data_address+prev_data_size+1) > largest_address:
                largest_address = prev_data_address+prev_data_size+1
    return largest_address

def insert_ringbuffer(ringbuffer_time_dim: int, op: Operator) -> Operator:
    subgraph = op.subgraph

    ringbuffer_shape = list(op.inputs[0].shape)
    ringbuffer_shape[1] = ringbuffer_time_dim

    ringbuffer_tensor = subgraph.create_tensor(
        f"{op.name}/ringbuffer",
        TensorType.INT8,
        consumers=[op],
        shape=ringbuffer_shape,
        quantization=op.inputs[0].quantization,
        custom_options={"tdnn":True},
    )

    prev_data_address = find_largest_address_in_persistent_buffer(subgraph)
    prev_data_shape = ringbuffer_shape
    prev_data_shape[1] = prev_data_shape[1] - 1
    prev_data_size = np.prod(prev_data_shape)

    prev_data_address_size = subgraph.create_tensor(
        f"{op.name}/prev_data_address_size", 
        TensorType.INT8,
        shape=(2,),
        custom_options={"tdnn":True},
    )

    # disconnect input from op
    op.inputs[0].consumers.pop(0)
    # create and connect ring buffer op
    ringbuffer_op = subgraph.create_operator(
        OperatorCode(XCOREOpCodes.XC_ringbuffer),
        # inputs=[op.inputs[0], prev_data_tensor, persistent_buffer_number],
        inputs=[op.inputs[0],  prev_data_address_size],
        outputs=[ringbuffer_tensor],
        custom_options={"prev_data_address":prev_data_address,"prev_data_size":prev_data_size},
    )
    # connect op to ring buffer
    op.inputs[0] = ringbuffer_tensor

    for input_tensor in op.inputs:
        input_tensor.add_custom_options(tdnn=True)

    params = np.int32([prev_data_address,prev_data_size])
    ringbuffer_op.inputs[1].buffer.data = params

    return op

class TdnnShallowinConv2dPass(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.CONV_2D

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and "tdnn" not in op.custom_options
            and len(op.inputs[1].shape) >= 3
        )
     
    def mutate(self, op: Operator) -> Operator:
        op.add_custom_options(tdnn=True)

        # kernel_size[0]
        ringbuffer_time_dim = op.inputs[1].shape[1]
        op = insert_ringbuffer(ringbuffer_time_dim, op)

        return op

class TdnnMaxPool2DPass(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.MAX_POOL_2D

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and "tdnn" not in op.custom_options
        )
            
    def mutate(self, op: Operator) -> Operator:
        op.add_custom_options(tdnn=True)

        options = op.builtin_options

        ringbuffer_time_dim = options["filter_height"]

        op = insert_ringbuffer(ringbuffer_time_dim,op)
        
        return op

class TdnnAveragePool2DPass(ReplaceAveragePool2DPass):
    def mutate(self, op: Operator) -> Operator:
        new_op = super().mutate(op)

        ringbuffer_time_dim = new_op.custom_options["pool"][0]

        new_op = insert_ringbuffer(ringbuffer_time_dim, new_op)

        return new_op

class TdnnReshapePass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and op.operator_code.code is BuiltinOpCodes.RESHAPE
            and "tdnn" not in op.custom_options
        )

    def mutate(self, op: Operator) -> Operator:
        op.add_custom_options(tdnn=True)

        ringbuffer_time_dim = op.inputs[0].shape[1]

        op = insert_ringbuffer(ringbuffer_time_dim, op)
        
        new_op = super().mutate(op)

        return new_op 

class TdnnTensorPass(TensorMatchingPass):
    def match(self, tensor: Tensor) -> bool:
        return (
            super().match(tensor) 
            and "tdnn" not in tensor.custom_options
            #checks if tensor is 4d
            and len(tensor.shape) == 4
        )

    def mutate(self, tensor: Tensor) -> Tensor:
        tensor.add_custom_options(tdnn=True)

        shape = list(tensor.shape)
        shape[1] = 1
        tensor.shape = tuple(shape)

        return tensor
    
class TdnnCleanupPass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and "tdnn" in op.custom_options
        )

    def mutate(self, op: Operator) -> bool:
        op.custom_options.pop('tdnn')
        op.custom_options.pop('prev_data_address', None)
        op.custom_options.pop('prev_data_size', None)
        return op

class TdnnPersistentBufferSizePass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and op.operator_code.code is XCOREOpCodes.XC_ringbuffer
            and "persistent_buffer_size" not in op.custom_options
        )

    def mutate(self, op: Operator) -> bool:
        largest_address = find_largest_address_in_persistent_buffer(op.subgraph)
        op.add_custom_options(persistent_buffer_size=largest_address)
        return op

# class TdnnGlobalAveragePool2DPass(ReplaceGlobalAveragePool2DPass):
#     def mutate(self, op: Operator) -> Operator:
#         new_op = super().mutate(op)

#         ringbuffer_time_dim = new_op.inputs[0].shape[1]

#         new_op = insert_ringbuffer(ringbuffer_time_dim, new_op)

#         return new_op
