# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
from copy import deepcopy
import numpy as np
from math import floor
from tensorflow.python.ops.gen_array_ops import strided_slice

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

def insert_slice_concat(op: Operator,num_slices :int) -> Operator:
    subgraph = op.subgraph
    
    next_op = op.outputs[0].consumers[0]
    
    concat_tensor = subgraph.create_tensor(
        f"{op.name}/concat",
        TensorType.INT8,
        consumers=[next_op],
        shape=op.outputs[0].shape,
        quantization=op.outputs[0].quantization,
    )

    # create and connect concat op
    concat_op = subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.CONCATENATION),
        outputs=[concat_tensor],
    )
    concat_op.builtin_options = {
        "axis": 2,
    }

    next_op.inputs[0] = concat_tensor

    # disconnect output from op
    op.outputs[0].consumers.pop(0)

    quarter_dim2 = op.outputs[0].shape[2]//num_slices
    quarter_dim2_remainder = op.outputs[0].shape[2]%num_slices

    # slicer op needs to know where to begin slice, end of prev slice
    prev_end_params = 0

    for i in range(num_slices):
        slice_tensor_shape = list(op.outputs[0].shape)
        # handles dim that are not divisible by num_slices
        if i < quarter_dim2_remainder:
            fraction_handler=1
        else:
            fraction_handler=0
        slice_tensor_shape[2] = quarter_dim2+fraction_handler
        slice_tensor = subgraph.create_tensor(
            f"slice_tensor_{i}",
            TensorType.INT8,
            consumers=[concat_op],
            shape=slice_tensor_shape,
            quantization=op.outputs[0].quantization,
        )

        concat_op.inputs.append(slice_tensor)

        # create and connect slice op
        slicer = subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.STRIDED_SLICE),
            inputs=[op.outputs[0]],
            outputs=[slice_tensor],
        )

        # constexpr int kBeginTensor = 1;
        # constexpr int kEndTensor = 2;
        # constexpr int kStridesTensor = 3;

        begin_params = list(op.outputs[0].shape)
        begin_params[0] = 0
        begin_params[1] = 0
        begin_params[2] =prev_end_params
        begin_params[3] = 0
        begin_params = np.int32(begin_params)
        begin_tensor = subgraph.create_tensor(
            f"{slicer.name}/begin",
            TensorType.INT8,
            consumers=[slicer],
            shape=begin_params.shape,
        )
        slicer.inputs.append(begin_tensor)
        slicer.inputs[1].buffer.data = begin_params

        end_params = list(op.outputs[0].shape)
        end_params[2] = prev_end_params+quarter_dim2+fraction_handler
        prev_end_params = end_params[2]
        end_params = np.int32( end_params)
        end_tensor = subgraph.create_tensor(
            f"{slicer.name}/end",
            TensorType.INT8,
            consumers=[slicer],
            shape=end_params.shape,
        )
        slicer.inputs.append(end_tensor)
        slicer.inputs[2].buffer.data = end_params

        strides_params = np.int32([1,1,1,1])
        strides_tensor = subgraph.create_tensor(
            f"{slicer.name}/strides",
            TensorType.INT8,
            consumers=[slicer],
            shape=strides_params.shape,
        )
        slicer.inputs.append(strides_tensor)
        slicer.inputs[3].buffer.data = strides_params

    return op
 
def move_slice_above_op(op: Operator,num_slices :int) -> Operator:
    subgraph = op.subgraph
    
    # shorter names also preserves reference
    slicer_0=op.outputs[0].consumers[0]
    slice_tensor_0=slicer_0.outputs[0]
    concat_op=slice_tensor_0.consumers[0]
    block_input = op.inputs[0]
    filter_size = op.inputs[1].shape[2]
    assert op.inputs[1].shape[1] == op.inputs[1].shape[2],"op splitting only implemented for square filters"
    assert op.inputs[1].shape[1]%2!=0,"op splitting with padding only implemented for odd shaped filters"
    assert op.inputs[0].shape[1] == op.inputs[0].shape[2],"op splitting with padding only implemented for square inputs"
    
    # changes strided slice's input tensor
    slicer_0.inputs[0] = op.inputs[0]
    op.inputs[0].consumers[0]=slicer_0

    op.inputs[0]=slice_tensor_0
    slice_tensor_0.consumers[0]=op 

    op.outputs[0].shape = slice_tensor_0.shape
    concat_op.inputs[0]=op.outputs[0]
    op.outputs[0].consumers[0]=concat_op
    slice_tensor_0.quantization=slicer_0.inputs[0].quantization

    if op.builtin_options.get('padding').name == 'SAME':
        pad_size = (filter_size-1)//2

        padded_tensor_shape = list(slice_tensor_0.shape)
        padded_tensor_shape[1] = padded_tensor_shape[1]+pad_size*2
        padded_tensor_shape[2] = padded_tensor_shape[2]+pad_size*2
        padded_tensor = subgraph.create_tensor(
            f"padded_tensor_0",
            TensorType.INT8,
            consumers=[op],
            shape=padded_tensor_shape,
            quantization=slice_tensor_0.quantization,
        )

        #batch,[top,bottom],[left,right],depth
        paddings_data = [[0,0],[pad_size, pad_size,], [pad_size, 0],[0,0]]
        paddings_data = np.array(paddings_data)
        paddings = subgraph.create_tensor(
            f"paddings_0",
            TensorType.INT8,
            shape=paddings_data.shape,
        )
        paddings_data = np.int8(paddings_data)
        paddings.buffer.data=paddings_data

        constant_value_data = [0]
        constant_value_data = np.array(constant_value_data)
        contant_value = subgraph.create_tensor(
            f"paddings_0",
            TensorType.INT8,
            shape=constant_value_data.shape,
        )
        constant_value_data = np.int8(constant_value_data)
        contant_value.buffer.data = constant_value_data

        padder = subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.PAD),
            inputs=[slice_tensor_0,paddings,contant_value],
            outputs=[padded_tensor],
        )
        op.inputs[0]=padded_tensor

    # adjustments for filter size >1 
    op_output_shape = list(op.outputs[0].shape)
    op_output_shape[2] = op_output_shape[2]-filter_size+1
    op.outputs[0].shape = op_output_shape

    slice_tensor_0_shape = list(slice_tensor_0.shape)
    slice_tensor_0_shape[1] = slice_tensor_0_shape[1]+filter_size-1
    slice_tensor_0.shape = slice_tensor_0_shape

    #int32 so step 4
    begin_params = list(slicer_0.inputs[1].buffer.data)[::4]
    end_params = list(slicer_0.inputs[2].buffer.data)[::4]
    end_params[1] = end_params[1] +filter_size-1
    end_params = np.int32(end_params)
    slicer_0.inputs[2].buffer.data = end_params

    end_offset = 0
    begin_offset = 0
    for i in range(1,num_slices):
        # shorter names
        slicer_i=op.outputs[0].consumers[i]
        # op.outputs[0].consumers.pop(1)
        slice_tensor_i=slicer_i.outputs[0]
        concat_op=slice_tensor_i.consumers[0]
        
            
        new_op_output = subgraph.create_tensor(
            f"{op.name}/output{i}",
            TensorType.INT8,
            consumers=[concat_op],
            shape=slice_tensor_i.shape,
            quantization=op.outputs[0].quantization,
        ) 

        # resize expands tensors to the left
        # but left most tensor cannnot expand to the left
        # so redistribute left most tensor size change
        # offset accumulates across slices
        if i > num_slices-filter_size:
            new_op_output_shape = list(new_op_output.shape)
            new_op_output_shape[2] = new_op_output_shape[2]+1
            new_op_output.shape = new_op_output_shape
            end_offset = end_offset+1
            begin_offset = end_offset-1
            

        # resizes for filter size >1
        new_op_output_shape = list(new_op_output.shape)
        new_op_output_shape[2] = new_op_output_shape[2]+filter_size-1
        new_op_output_shape[1] = new_op_output_shape[1]+filter_size-1
        slice_tensor_i.shape = new_op_output_shape
        
        begin_param = list(slicer_i.inputs[1].buffer.data)[8]
        begin_param = begin_param-filter_size+1+begin_offset
        begin_params = [0,0,begin_param,0]
        slicer_i.inputs[1].buffer.data = np.int32(begin_params)

        end_params = list(slicer_i.inputs[2].buffer.data)[::4]
        end_params[1] = end_params[1] +filter_size-1
        end_params[2] = end_params[2] +end_offset
        end_params = np.int32(end_params)
        slicer_i.inputs[2].buffer.data = end_params

        # changes slicer's input tensor
        slicer_i.inputs[0] = block_input
        block_input.consumers.append(slicer_i)
        slice_tensor_i.quantization=slice_tensor_0.quantization

        new_op_input_1 = subgraph.create_tensor(
            f"{op.name}/input_1",
            op.inputs[1].type,
            op.inputs[1].shape,
            quantization=deepcopy(op.inputs[1].quantization),
        )
        new_op_input_1.buffer.data= op.inputs[1].buffer.data
        new_op_input_2 = subgraph.create_tensor(
            f"{op.name}/input_2",
            op.inputs[2].type,
            op.inputs[2].shape,
            quantization=deepcopy(op.inputs[2].quantization),
        )
        new_op_input_2.buffer.data= op.inputs[2].buffer.data
        new_op = subgraph.create_operator(
            OperatorCode(op.operator_code.code),
            inputs=[slice_tensor_i,new_op_input_1,new_op_input_2],
            outputs=[new_op_output],
        ) 
        new_op.add_custom_options(op_splitting=True)
        # new_op.builtin_options.update(fused_activation_function=op.builtin_options.get('fused_activation_function'))
        new_op.builtin_options=op.builtin_options

        # concat_op.inputs.append(new_op_output)
        concat_op.inputs[i]=new_op_output
        slice_tensor_i.consumers.pop(0)
        new_op_output.producers[0]=new_op

    op.outputs[0].consumers = op.outputs[0].consumers[:1]
    return op



class OperatorSplittingPass(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.CONV_2D

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and "op_splitting" not in op.custom_options
        )
     
    def mutate(self, op: Operator) -> Operator:
        op.add_custom_options(op_splitting=True)

        num_slices = 1
        op = insert_slice_concat(op,num_slices)
        op = move_slice_above_op(op,num_slices)

class OperatorSplittingCleanupPass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and "op_splitting" in op.custom_options
        )

    def mutate(self, op: Operator) -> bool:
        op.custom_options.pop('op_splitting')
        return op
