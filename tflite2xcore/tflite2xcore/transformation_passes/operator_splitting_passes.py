# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
from copy import deepcopy
import numpy as np

from tflite2xcore.transformation_passes.transformation_passes import (
    OperatorMatchingPass,
    QuantizedOperatorMatchingPass,
)
from tflite2xcore.xcore_model import Operator, Tensor
from tflite2xcore.xcore_schema import (
    OperatorCode,
    TensorType,
    BuiltinOpCodes,
)
from tflite2xcore.xcore_schema.misc_enums import Padding

def insert_slice_concat(op: Operator,num_slices :int) -> Operator:
    subgraph = op.subgraph
    
    next_ops = op.outputs[0].consumers
    
    concat_tensor = subgraph.create_tensor(
        f"{op.name}/concat",
        TensorType.INT8,
        consumers=next_ops,
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

    for i in range(len(next_ops)):
        for j in range(len(next_ops[i].inputs)):
            if next_ops[i].inputs[j].producers: 
                if next_ops[i].inputs[j].producers[0] is op: 
                    next_ops[i].inputs[j] = concat_tensor


    # disconnect output from op
    op.outputs[0].consumers = []

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

        #   op_params.begin_mask = op_context->params->begin_mask;
        #   op_params.ellipsis_mask = 0;
        #   op_params.end_mask = op_context->params->end_mask;
        #   op_params.new_axis_mask = 0;
        #   op_params.shrink_axis_mask = op_context->params->shrink_axis_mask;
#         tf.strided_slice(
#     input_, begin, end, strides=None, begin_mask=0, end_mask=0, ellipsis_mask=0,
#     new_axis_mask=0, shrink_axis_mask=0, var=None, name=None
# )
  
        # create and connect slice op
        slicer = subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.STRIDED_SLICE),
            inputs=[op.outputs[0]],
            outputs=[slice_tensor],
            builtin_options= {"begin_mask":0,"end_mask":0,  "shrink_axis_mask":0,},
        )

        # strided slice input param tensors
        # BeginTensor = 1;
        # EndTensor = 2;
        # StridesTensor = 3;

        begin_params = list(op.outputs[0].shape)
        begin_params[0] = 0
        begin_params[1] = 0
        begin_params[2] =prev_end_params
        begin_params[3] = 0
        begin_params = np.int32(begin_params)
        begin_tensor = subgraph.create_tensor(
            f"{slicer.name}/begin",
            TensorType.INT32,
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
            TensorType.INT32,
            consumers=[slicer],
            shape=end_params.shape,
        )
        slicer.inputs.append(end_tensor)
        slicer.inputs[2].buffer.data = end_params

        strides_params = np.int32([1,1,1,1])
        strides_tensor = subgraph.create_tensor(
            f"{slicer.name}/strides",
            TensorType.INT32,
            consumers=[slicer],
            shape=strides_params.shape,
        )
        slicer.inputs.append(strides_tensor)
        slicer.inputs[3].buffer.data = strides_params

    return op
 
def move_slice_above_first_op(op: Operator,num_slices :int) -> Operator:
    subgraph = op.subgraph
    
    # shorter names also preserves reference
    op_input_shape = op.inputs[0].shape
    slicer_0=op.outputs[0].consumers[0]
    slice_tensor_0=slicer_0.outputs[0]
    concat_op=slice_tensor_0.consumers[0]
    block_input = op.inputs[0]
    filter_size = op.inputs[1].shape[2]
    stride_w = op.builtin_options.get('stride_w')
    
    assert op.inputs[1].shape[1] == op.inputs[1].shape[2],"op splitting only implemented for square filters"
    assert op.inputs[1].shape[1] == 3 or op.inputs[1].shape[1] == 1,"op splitting only implemented for 3x3 or 1x1 filters"
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

    # adjustments for filter size >1 
    op_output_shape = list(op.outputs[0].shape)
    op_output_shape[2] = (op_output_shape[2]-filter_size+stride_w)//stride_w
    op.outputs[0].shape = op_output_shape

    slice_tensor_0_shape = list(slice_tensor_0.shape)
    slice_tensor_0_shape[1] = op_input_shape[1]
    slice_tensor_0_shape[3] = block_input.shape[3]
    slice_tensor_0.shape = slice_tensor_0_shape

    #int32 so step 4
    begin_params = list(slicer_0.inputs[1].buffer.data)[::4]
    end_params = list(op_input_shape)
    end_params[3] = block_input.shape[3]
    end_params = np.int32(end_params)
    slicer_0.inputs[2].buffer.data = end_params

    # checks for same padding and filter size >1
    same_padding = False
    if op.builtin_options.get('padding').name == 'SAME' and op.inputs[1].shape[1]!=1:
        same_padding = True
        op = extract_padding(op,left=1)

    for i in range(1,num_slices):
        # shorter names
        slicer_i=op.outputs[0].consumers[i]
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
        if i >= (num_slices-filter_size+stride_w)//stride_w:
            new_op_output_shape = list(new_op_output.shape)
            new_op_output_shape[2] = new_op_output_shape[2]+1
            new_op_output.shape = new_op_output_shape
            

        # resizes for filter size >1
        new_slice_tensor_shape = list(new_op_output.shape)
        new_slice_tensor_shape[1] = op_input_shape[1]
        new_slice_tensor_shape[2] = new_slice_tensor_shape[2]*stride_w+filter_size-stride_w
        new_slice_tensor_shape[3] = block_input.shape[3]
        slice_tensor_i.shape = new_slice_tensor_shape
        

        begin_param = end_params[2]+(-filter_size+stride_w)//stride_w

        begin_params = [0,0,begin_param,0]
        slicer_i.inputs[1].buffer.data = np.int32(begin_params)

        end_params = list(slicer_i.inputs[2].buffer.data)[::4]
        end_params[1] = op_input_shape[1]
        end_params[2] = begin_param+new_slice_tensor_shape[2]
        end_params[3] = block_input.shape[3]
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
        new_op.builtin_options=op.builtin_options

        concat_op.inputs[i]=new_op_output
        slice_tensor_i.consumers.pop(0)
        new_op_output.producers[0]=new_op

        # checks for same padding
        if same_padding:
            right=0
            if i==num_slices-1:
                right=1
            new_op = extract_padding(new_op,right=right)

    op.outputs[0].consumers = op.outputs[0].consumers[:1]
    return op

def move_slice_above_add(op: Operator,num_slices :int) -> Operator:
    subgraph = op.subgraph
    
    # shorter names also preserves reference
    op_input_shape = op.inputs[0].shape
    slicer_0=op.outputs[0].consumers[0]
    slice_tensor_0=slicer_0.outputs[0]
    concat_op=slice_tensor_0.consumers[0]
    block_input = op.inputs[0]
    residual_connection = op.inputs[1]
    
    # changes strided slice's input tensor
    slicer_0.inputs[0] = op.inputs[0]
    op.inputs[0].consumers[0]=slicer_0

    # changes op's inputs
    op.inputs[0]=slice_tensor_0
    slice_tensor_0.consumers[0]=op 

    op.outputs[0].shape = slice_tensor_0.shape
    concat_op.inputs[0]=op.outputs[0]
    op.outputs[0].consumers[0]=concat_op
    slice_tensor_0.quantization=slicer_0.inputs[0].quantization

    slice_tensor_0_shape = list(slice_tensor_0.shape)
    slice_tensor_0_shape[1] = op_input_shape[1]
    slice_tensor_0_shape[3] = block_input.shape[3]
    slice_tensor_0.shape = slice_tensor_0_shape

    #int32 so step 4
    begin_params = list(slicer_0.inputs[1].buffer.data)[::4]
    end_params = list(op_input_shape)
    end_params[2] = end_params[2]//num_slices
    end_params = np.int32(end_params)
    slicer_0.inputs[2].buffer.data = end_params
    
    # adds slicer to residual connection
    slice_tensor_2 = subgraph.create_tensor(
            f"slice_tensor_{2}",
            TensorType.INT8,
            consumers=[op],
            shape=slice_tensor_0_shape,
            quantization=residual_connection.quantization,
        ) 

    op.inputs[1]=slice_tensor_2

    # create and connect slice op
    slicer_2 = subgraph.create_operator(
            OperatorCode(BuiltinOpCodes.STRIDED_SLICE),
            inputs=[residual_connection],
            outputs=[slice_tensor_2],
            builtin_options= {"begin_mask":0,"end_mask":0,  "shrink_axis_mask":0,},
        )
    
    #begin params  
    begin_tensor = subgraph.create_tensor(
            f"{slicer_2.name}/begin",
            TensorType.INT32,
            consumers=[slicer_2],
            shape=slicer_0.inputs[1].shape,
        )
    slicer_2.inputs.append(begin_tensor)
    slicer_2.inputs[1].buffer.data = slicer_0.inputs[1].buffer.data
    #end params
    end_tensor = subgraph.create_tensor(
            f"{slicer_2.name}/end",
            TensorType.INT32,
            consumers=[slicer_2],
            shape=slicer_0.inputs[2].shape,
        )
    slicer_2.inputs.append(end_tensor)
    slicer_2.inputs[2].buffer.data = slicer_0.inputs[2].buffer.data
    #strides params
    strides_tensor = subgraph.create_tensor(
            f"{slicer_2.name}/strides",
            TensorType.INT32,
            consumers=[slicer_2],
            shape=slicer_0.inputs[3].shape,
        )
    slicer_2.inputs.append(strides_tensor)
    slicer_2.inputs[3].buffer.data = slicer_0.inputs[3].buffer.data

    residual_connection.consumers.remove(op)
    residual_connection.consumers.append(slicer_2)
    
    for i in range(1,num_slices):
        # shorter names
        slicer_i=op.outputs[0].consumers[i]
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
        if i >= num_slices:
            new_op_output_shape = list(new_op_output.shape)
            new_op_output_shape[2] = new_op_output_shape[2]+1
            new_op_output.shape = new_op_output_shape

        new_slice_tensor_shape = list(new_op_output.shape)
        new_slice_tensor_shape[1] = op_input_shape[1]
        new_slice_tensor_shape[2] = new_slice_tensor_shape[2]
        new_slice_tensor_shape[3] = block_input.shape[3]
        slice_tensor_i.shape = new_slice_tensor_shape

        begin_param = end_params[2]

        begin_params = [0,0,begin_param,0]
        slicer_i.inputs[1].buffer.data = np.int32(begin_params)

        end_params = list(slicer_i.inputs[2].buffer.data)[::4]
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
        new_op = subgraph.create_operator(
            OperatorCode(op.operator_code.code),
            inputs=[slice_tensor_i],
            outputs=[new_op_output],
        ) 
        new_op.add_custom_options(op_splitting=True)
        new_op.builtin_options=op.builtin_options

        concat_op.inputs[i]=new_op_output
        slice_tensor_i.consumers.pop(0)
        new_op_output.producers[0]=new_op

        # adds slicer to residual connection
        slice_tensor_iplus = subgraph.create_tensor(
                f"slice_tensor_{i+2}",
                TensorType.INT8,
                consumers=[new_op],
                shape=slice_tensor_0_shape,
                quantization=residual_connection.quantization,
            ) 

        new_op.inputs.append(slice_tensor_iplus)

        # create and connect slice op
        slicer_iplus = subgraph.create_operator(
                OperatorCode(BuiltinOpCodes.STRIDED_SLICE),
                inputs=[residual_connection],
                outputs=[slice_tensor_iplus],
                builtin_options= {"begin_mask":0,"end_mask":0,  "shrink_axis_mask":0,},
            )
        
        #begin params  
        begin_tensor = subgraph.create_tensor(
                f"{slicer_iplus.name}/begin",
                TensorType.INT32,
                consumers=[slicer_iplus],
                shape=slicer_0.inputs[1].shape,
            )
        slicer_iplus.inputs.append(begin_tensor)
        slicer_iplus.inputs[1].buffer.data = slicer_i.inputs[1].buffer.data
        #end params
        end_tensor = subgraph.create_tensor(
                f"{slicer_iplus.name}/end",
                TensorType.INT32,
                consumers=[slicer_iplus],
                shape=slicer_0.inputs[2].shape,
            )
        slicer_iplus.inputs.append(end_tensor)
        slicer_iplus.inputs[2].buffer.data = slicer_i.inputs[2].buffer.data
        #strides params
        strides_tensor = subgraph.create_tensor(
                f"{slicer_iplus.name}/strides",
                TensorType.INT32,
                consumers=[slicer_iplus],
                shape=slicer_0.inputs[3].shape,
            )
        slicer_iplus.inputs.append(strides_tensor)
        slicer_iplus.inputs[3].buffer.data = slicer_i.inputs[3].buffer.data

        residual_connection.consumers.append(slicer_iplus)

    op.outputs[0].consumers = op.outputs[0].consumers[:1]
    return op

def move_slice_above_op(op: Operator,num_slices :int) -> Operator:
    subgraph = op.subgraph
    
    # shorter names also preserves reference
    op_input_shape = op.inputs[0].shape
    slicer_0=op.outputs[0].consumers[0]
    slice_tensor_0=slicer_0.outputs[0]
    following_op=slice_tensor_0.consumers[0]
    block_input = op.inputs[0]
    filter_size = op.inputs[1].shape[2]
    stride_w = op.builtin_options.get('stride_w')
    
    
    # changes strided slice's input tensor
    slicer_0.inputs[0] = op.inputs[0]
    op.inputs[0].consumers[0]=slicer_0

    op.inputs[0]=slice_tensor_0
    slice_tensor_0.consumers[0]=op 

    op.outputs[0].shape = slice_tensor_0.shape
    following_op.inputs[0]=op.outputs[0]
    op.outputs[0].consumers[0]=following_op
    slice_tensor_0.quantization=slicer_0.inputs[0].quantization

    op.outputs[0].shape = slice_tensor_0.shape

    slice_tensor_0_shape = list(slice_tensor_0.shape)
    slice_tensor_0_shape[1] = op_input_shape[1]
    slice_tensor_0_shape[2] = slice_tensor_0_shape[2]*stride_w-stride_w+filter_size
    slice_tensor_0_shape[3] = block_input.shape[3]
    slice_tensor_0.shape = slice_tensor_0_shape

    if filter_size>1 or stride_w>1:
        #int32 so step 4
        end_params = list(slicer_0.inputs[2].buffer.data)[::4]
        end_params[1] = op_input_shape[1]
        end_params[2] = slice_tensor_0_shape[2]
        end_params[3] = block_input.shape[3]
        end_params = np.int32(end_params)
        slicer_0.inputs[2].buffer.data = end_params

    # checks for same padding and filter size >1
    same_padding = False
    if op.builtin_options.get('padding').name == 'SAME' and op.inputs[1].shape[1]!=1:
        same_padding = True
        op = extract_padding(op,left=1)

    for i in range(1,num_slices):
        # shorter names
        slicer_i=op.outputs[0].consumers[i]
        slice_tensor_i=slicer_i.outputs[0]
        following_op=slice_tensor_i.consumers[0]
        
            
        new_op_output = subgraph.create_tensor(
            f"{op.name}/output{i}",
            TensorType.INT8,
            consumers=[following_op],
            shape=slice_tensor_i.shape,
            quantization=op.outputs[0].quantization,
        ) 

        # resizes for filter size >1
        new_slice_tensor_shape = list(new_op_output.shape)
        new_slice_tensor_shape[1] = op_input_shape[1]
        new_slice_tensor_shape[2] = new_slice_tensor_shape[2]*stride_w+filter_size-stride_w
        new_slice_tensor_shape[3] = block_input.shape[3]
        slice_tensor_i.shape = new_slice_tensor_shape
        
        if filter_size>1 or stride_w>1:
            begin_param = end_params[2]+(-filter_size+stride_w)//stride_w
            begin_params = [0,0,begin_param,0]
            slicer_i.inputs[1].buffer.data = np.int32(begin_params)

            end_params = list(slicer_i.inputs[2].buffer.data)[::4]
            end_params[1] = op_input_shape[1]
            end_params[2] = begin_param+new_slice_tensor_shape[2]
            end_params[3] = block_input.shape[3]
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
        new_op.builtin_options=op.builtin_options

        following_op.inputs[0]=new_op_output
        slice_tensor_i.consumers.pop(0)
        new_op_output.producers[0]=new_op

        # checks for same padding
        if same_padding:
            right=0
            if i==num_slices-1:
                right=1
            new_op = extract_padding(new_op,right=right)

    op.outputs[0].consumers = op.outputs[0].consumers[:1]
    return op

def extract_padding(op: Operator,left: int=0, right: int=0) -> Operator:

    # shorter names also preserves reference
    subgraph = op.subgraph
    slice_tensor_i=op.inputs[0]
    slicer_i = slice_tensor_i.producers[0]
    
    op.builtin_options.update({'padding':Padding.VALID})

    filter_size = op.inputs[1].shape[2]
    stride_w = op.builtin_options.get('stride_w')

    total_padding =(filter_size-stride_w) 
    top_left_pad_size = total_padding//2
    bottom_right_pad_size = total_padding-top_left_pad_size

    # vertical receptive field changes will be handled by padding
    padded_tensor_shape = list(slice_tensor_i.shape)
    padded_tensor_shape[1] = padded_tensor_shape[1]+total_padding
    
    padded_tensor = subgraph.create_tensor(
        f"padded_tensor_0",
        TensorType.INT8,
        consumers=[op],
        shape=padded_tensor_shape,
        quantization=slice_tensor_i.quantization,
    )

    slice_tensor_shape = list(slice_tensor_i.shape)

    if left:
        end_params = list(slicer_i.inputs[2].buffer.data)[::4]
        end_params[2] = end_params[2]-top_left_pad_size
        end_params = np.int32(end_params)
        slicer_i.inputs[2].buffer.data = end_params

        slice_tensor_shape[2] = slice_tensor_shape[2]-top_left_pad_size

        #batch,[top,bottom],[left,right],depth
        paddings_data = [[0,0],[top_left_pad_size, bottom_right_pad_size], [top_left_pad_size,0],[0,0]]
        
    if not(left or right):
        begin_param = list(slicer_i.inputs[1].buffer.data)[8]
        begin_param = begin_param-top_left_pad_size
        begin_params = [0,0,begin_param,0]
        slicer_i.inputs[1].buffer.data = np.int32(begin_params) 

        end_params = list(slicer_i.inputs[2].buffer.data)[::4]
        end_params[2] = end_params[2]-top_left_pad_size
        end_params = np.int32(end_params)
        slicer_i.inputs[2].buffer.data = end_params
        slice_tensor_shape[2] = slice_tensor_shape[2]
        paddings_data = [[0,0],[top_left_pad_size, bottom_right_pad_size], [0,0],[0,0]]

    if right:
        begin_param = list(slicer_i.inputs[1].buffer.data)[8]
        begin_param = begin_param-top_left_pad_size
        begin_params = [0,0,begin_param,0]
        slicer_i.inputs[1].buffer.data = np.int32(begin_params)
        slice_tensor_shape[2] = slice_tensor_shape[2]-bottom_right_pad_size
        paddings_data = [[0,0],[top_left_pad_size, bottom_right_pad_size], [0,bottom_right_pad_size],[0,0]]
    
    slice_tensor_i.shape = slice_tensor_shape

    paddings_data = np.array(paddings_data)
    paddings_data = np.int32(paddings_data)
    paddings = subgraph.create_tensor(
        f"paddings_0",
        TensorType.INT32,
        shape=paddings_data.shape,
    )
    paddings.buffer.data=paddings_data
    
    constant_value_data =op.inputs[0].quantization.get('zero_point')
    constant_value_data = np.array(constant_value_data)
    constant_value = subgraph.create_tensor(
        f"paddings_0",
        TensorType.INT8,
        shape=constant_value_data.shape,
    )
    constant_value_data = np.int8(constant_value_data)
    constant_value.buffer.data = constant_value_data

    slice_tensor_i.consumers.pop(0)

    pad_op = subgraph.create_operator(
        OperatorCode(BuiltinOpCodes.PAD),
        inputs=[slice_tensor_i,paddings],#,constant_value],
        outputs=[padded_tensor],
    )
    # pad_op.inputs[2].quantization = pad_op.outputs[0].quantization
    op.inputs[0]=padded_tensor

    return op

class OperatorSplittingPass(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.CONV_2D

    def match(self, op: Operator) -> bool:
        input_tensor_producers = op.inputs[0].producers
        return (
            super().match(op)
            and input_tensor_producers # checks if prev op exits
            and input_tensor_producers[0].operator_code.code == BuiltinOpCodes.DEPTHWISE_CONV_2D
            and input_tensor_producers[0].inputs[0].producers # checks if prev op exits
            and input_tensor_producers[0].inputs[0].producers[0].operator_code.code == BuiltinOpCodes.CONV_2D 
            and np.prod(op.inputs[0].shape) in [(40*40*96)]
            # and np.prod(input_tensor_producers[0].inputs[0].shape) in [614400,(40*40*96)]
            and len(input_tensor_producers[0].inputs[0].producers[0].outputs) ==1
            and input_tensor_producers[0].inputs[0].producers[0].inputs[1].shape[2] == 1
            and "op_splitting" not in op.custom_options
        ) 

    def mutate(self, op: Operator) -> Operator:
        op.add_custom_options(op_splitting=True)

        if op.inputs[0].producers[0].inputs[0].shape[2] > 40: 
            num_slices = 6
        else:
            num_slices = 2
        preceding_op = op.inputs[0].producers[0]
        double_preceding_op = preceding_op.inputs[0].producers[0]
        op = insert_slice_concat(op,num_slices)
        op = move_slice_above_first_op(op,num_slices)
        op = move_slice_above_op(preceding_op,num_slices)
        op = move_slice_above_op(double_preceding_op,num_slices)

class AddOperatorSplittingPass(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.ADD

    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and np.prod(op.outputs[0].shape) == (80*80*16)
            and "op_splitting" not in op.custom_options
        ) 

    def mutate(self, op: Operator) -> Operator:
        op.add_custom_options(op_splitting=True)

        num_slices = 2
        preceding_op = op.inputs[0].producers[0]
        double_preceding_op = preceding_op.inputs[0].producers[0]
        
        op = insert_slice_concat(op,num_slices)
        op = move_slice_above_add(op,num_slices)
        op = move_slice_above_op(preceding_op,num_slices)
        op = move_slice_above_op(double_preceding_op,num_slices)
        

class TempOperatorSplittingPass(QuantizedOperatorMatchingPass):
    @property
    def matching_opcode(self):
        return BuiltinOpCodes.CONV_2D

    def match(self, op: Operator) -> bool:
        input_tensor_producers = op.inputs[0].producers
        return (
            super().match(op)
            and input_tensor_producers # checks if prev op exits
            and input_tensor_producers[0].operator_code.code == BuiltinOpCodes.DEPTHWISE_CONV_2D
            and input_tensor_producers[0].inputs[0].producers # checks if prev op exits
            and input_tensor_producers[0].inputs[0].producers[0].operator_code.code == BuiltinOpCodes.CONV_2D 
            and np.prod(input_tensor_producers[0].inputs[0].shape) in [(80*80*16)]
            # and len(input_tensor_producers[0].inputs[0].producers[0].outputs) ==1
            # and input_tensor_producers[0].inputs[0].producers[0].inputs[1].shape[2] == 1
            and "op_splitting" not in op.custom_options
        ) 

    def mutate(self, op: Operator) -> Operator:
        op.add_custom_options(op_splitting=True)

        num_slices = 2
        preceding_op = op.inputs[0].producers[0]
        op = insert_slice_concat(op,num_slices)
        op = move_slice_above_first_op(op,num_slices)
        op = move_slice_above_op(preceding_op,num_slices)

class OperatorSplittingCleanupPass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and "op_splitting" in op.custom_options
        )

    def mutate(self, op: Operator) -> bool:
        op.custom_options.pop('op_splitting')
        return op

class OperatorSplittingReshapeOptionsPass(OperatorMatchingPass):
    def match(self, op: Operator) -> bool:
        return (
            super().match(op)
            and  op.operator_code.code == BuiltinOpCodes.RESHAPE
            and  not op.builtin_options
        )

    def mutate(self, op: Operator) -> bool:
        op.builtin_options.update({"shape":np.array(op.outputs[0].shape)})

        return op
