# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from abc import ABC, abstractmethod
from asyncio import base_futures
import random
from typing import Sequence, Dict
from scipy import rand

from tflite2xcore import xcore_schema as xir
from tflite2xcore.xcore_schema import tensor
from zmq import EVENT_HANDSHAKE_FAILED_PROTOCOL


class ExecutionPlanner(ABC):
    def __init__(self, subgraph: xir.Subgraph):
        self._graph = subgraph

    @abstractmethod
    def make_plan(self) -> Sequence[xir.Operator]:
        raise NotImplementedError()


class ReverseDepthFirstPlanner(ExecutionPlanner):
    def make_plan(self) -> Sequence[xir.Operator]:
        # rely on dict's insertion order guarantee (CPython 3.6+)
        reverse_op_order: Dict[xir.Operator, None] = {}

        # initialize the op stack with a sentinel that we'll remove later
        sentinel_op = self._graph.create_operator(
            xir.OperatorCode(xir.XCOREOpCodes.DUMMY),
            inputs=self._graph.outputs,
        )
        sentinel_op.name = "SENTINEL"
        op_stack = [sentinel_op]

        # dependency counts to be used to resolve ops that have multiple consumers
        dependency_counts: Dict[xir.Operator, int] = {sentinel_op: 1}

        while op_stack:
            op = op_stack.pop(-1)
            if op in reverse_op_order:
                # op already scheduled
                continue

            if op not in dependency_counts:
                # this is the first time we see this op, so count the dependencies
                dependency_counts[op] = len(
                    {c for t in op.outputs for c in t.consumers}
                )

            if dependency_counts[op] <= 0:
                raise Exception(
                    "Found operator with 0 or fewer dependencies (the graph may be corrupted)"
                )

            dependency_counts[op] -= 1
            if dependency_counts[op]:
                # skip scheduling of op if there are other dependents
                continue

            reverse_op_order[op] = None
            for tin in sorted(op.inputs, key=lambda t: t.size):
                op_stack.extend(tin.producers)

        # remove sentinel op
        self._graph.remove_operator(sentinel_op)
        del reverse_op_order[sentinel_op]

        # return ops in reverse order
        return list(reversed(list(reverse_op_order.keys())))

class TensorsSizeMinimizerPlanner(ExecutionPlanner):
    def make_plan(self) -> Sequence[xir.Operator]:
        # rely on dict's insertion order guarantee (CPython 3.6+)
        op_order: Dict[xir.Operator, None] = {}

        # initialize the op stack with a sentinel that we'll remove later
        sentinel_op = self._graph.create_operator(
            xir.OperatorCode(xir.XCOREOpCodes.DUMMY),
            outputs=self._graph.inputs,
        )
        sentinel_op.name = "SENTINEL"
        op_stack = [sentinel_op]

        outputs_size=0
        for output in sentinel_op.outputs:
            outputs_size += output.size
        tensors_size_increase: Dict[xir.Operator, int] = {sentinel_op: outputs_size-0}
        not_ready = []
        while tensors_size_increase:
            ready_ops = tensors_size_increase
            for op in not_ready:
                ready_ops.pop(op)
            #find op that causes the smallest increase in tensor arena
            op = min(ready_ops, key=ready_ops.get)


            op_order[op] = None
            tensors_size_increase.pop(op)
            for output in op.outputs:
                for next_op in output.consumers:
                    outputs_size=0
                    inputs_size =0
                    for output in next_op.outputs:
                        outputs_size += output.size 
                    for in_tensor in next_op.inputs:
                        inputs_size += in_tensor.size 
                    tensors_size_increase[next_op]= outputs_size-inputs_size

            not_ready = set([])
            #check if op input tensors are ready
            for next_op in tensors_size_increase:
                for in_tensor in next_op.inputs:
                    for producers in in_tensor.producers:
                        if producers not in op_order:
                            not_ready.add(next_op)

        # remove sentinel op
        self._graph.remove_operator(sentinel_op)
        del op_order[sentinel_op]

        # return ops 
        return list(op_order.keys())


class RandomSequencePlanner(ExecutionPlanner):
    def make_plan(self) -> Sequence[xir.Operator]:
        # rely on dict's insertion order guarantee (CPython 3.6+)
        op_order: Dict[xir.Operator, None] = {}

        # initialize the op stack with a sentinel that we'll remove later
        sentinel_op = self._graph.create_operator(
            xir.OperatorCode(xir.XCOREOpCodes.DUMMY),
            outputs=self._graph.inputs,
        )
        sentinel_op.name = "SENTINEL"
        op_stack = [sentinel_op]

        not_ready = []
        ready_ops = op_stack[:]
        for op in not_ready:
            ready_ops.remove(op)

        num_optimizations = 0   
        while op_stack:
            if len(ready_ops)==1 or num_optimizations==2:
                op = ready_ops[0]
                op_order[op] = None
                op_stack.remove(op)
                for output in op.outputs:
                        for next_op in output.consumers:
                            op_stack.append(next_op) if next_op not in op_stack else op_stack

                #check if op input tensors are ready
                not_ready = set([])
                for next_op in op_stack:
                        for in_tensor in next_op.inputs:
                            for producers in in_tensor.producers:
                                if producers not in op_order:
                                    not_ready.add(next_op) if next_op not in not_ready else not_ready

                ready_ops = op_stack[:]
                for op in not_ready:
                    ready_ops.remove(op)
            
            else:

                num_optimizations += 1
                
                num_runs = 100**num_optimizations
                best_arena_size = 2**32

                for run in range(num_runs):
                    
                    # rely on dict's insertion order guarantee (CPython 3.6+)
                    temp_seq: Dict[xir.Operator, None] = op_order.copy()
                    temp_stack = op_stack[:]
                    temp_ready_ops = ready_ops[:]
                    temp_not_ready = set([])
                    current_arena_size = temp_stack[0].inputs[0].size
                    max_arena_size = current_arena_size

                    while len(temp_stack)>1:
                        temp_op = random.choice(temp_ready_ops)

                        temp_seq[temp_op] = None

                        temp_stack.remove(temp_op)

                        current_arena_size += temp_op.outputs[0].size 

                        if current_arena_size > max_arena_size:
                            max_arena_size = current_arena_size  
                        
                        # if all the consumers have been scheduled
                        if all(operators in temp_seq for operators in temp_op.inputs[0].consumers):
                            current_arena_size -= in_tensor.size 
                                 
                        for output in temp_op.outputs:
                            for next_op in output.consumers:
                                temp_stack.append(next_op) if next_op not in temp_stack else temp_stack
                        
                        #check if op input tensors are ready
                        temp_not_ready = set([])
                        for next_op in temp_stack:
                            for in_tensor in next_op.inputs:
                                for producer in in_tensor.producers:
                                    if producer not in temp_seq:
                                        temp_not_ready.add(next_op) if next_op not in temp_not_ready else temp_not_ready
                                    # else:
                                        # current_arena_size += in_tensor.size

                        temp_ready_ops = temp_stack[:]
                        for temp_op in temp_not_ready:
                            temp_ready_ops.remove(temp_op)

                    if max_arena_size < best_arena_size:
                        best_arena_size = max_arena_size
                        best_seq = temp_seq.copy()
                    print(max_arena_size)


                op_stack = temp_stack[:]
                ready_ops = temp_ready_ops

                op_order.update(best_seq)

        # remove sentinel op
        self._graph.remove_operator(sentinel_op)
        del op_order[sentinel_op]

        # return ops 
        return list(op_order.keys())

def update_arena_size(op_order):
    current_arena_size = list(op_order.keys())[0].outputs[0].size
    max_arena_size = -2**15 
    for op_index in range(1,len(op_order)):
        current_op = list(op_order.keys())[op_index]
                       
        for in_tensor in current_op.inputs:
            if not in_tensor.producers:
                current_arena_size += in_tensor.size
        current_arena_size += current_op.outputs[0].size 
      
        if current_arena_size > max_arena_size:
            max_arena_size = current_arena_size 

        # if all the consumers have been scheduled
        for in_tensor in current_op.inputs:
            tensor_deactivated = True
            for operator in in_tensor.consumers:
                if operator not in list(op_order.keys())[0:op_index+1]: 
                    tensor_deactivated = False
            if tensor_deactivated:
                current_arena_size -= in_tensor.size
    
    return max_arena_size
            
class DepthVsWidthPlanner(ExecutionPlanner):
    def make_plan(self) -> Sequence[xir.Operator]:
        # rely on dict's insertion order guarantee (CPython 3.6+)
        op_order: Dict[xir.Operator, None] = {}

        # initialize the op stack with a sentinel that we'll remove later
        sentinel_op = self._graph.create_operator(
            xir.OperatorCode(xir.XCOREOpCodes.DUMMY),
            outputs=self._graph.inputs,
        )
        sentinel_op.name = "SENTINEL"
        op_stack = [sentinel_op]

        not_ready = []
        ready_ops = op_stack[:]
        for op in not_ready:
            ready_ops.remove(op)

        num_optimizations = 0   
        while op_stack:
                if len(ready_ops)==1 or num_optimizations==2:
                    op = ready_ops[0]
                    op_order[op] = None
                    op_stack.remove(op)
                    for output in op.outputs:
                            for next_op in output.consumers:
                                op_stack.append(next_op) if next_op not in op_stack else op_stack

                    #check if op input tensors are ready
                    not_ready = set([])
                    for next_op in op_stack:
                            for in_tensor in next_op.inputs:
                                for producers in in_tensor.producers:
                                    if producers not in op_order:
                                        not_ready.add(next_op) if next_op not in not_ready else not_ready

                    ready_ops = op_stack[:]
                    for op in not_ready:
                        ready_ops.remove(op)
                
                else:

                    num_optimizations += 1
                    
                    #####
                    # Find ops to seq
                    #####
                    
                    # rely on dict's insertion order guarantee (CPython 3.6+)
                    temp_seq: Dict[xir.Operator, None] = op_order.copy()
                    temp_stack = op_stack[:]
                    temp_ready_ops = ready_ops[:]
                    temp_not_ready = set([])

                    while len(temp_stack)>1:
                        temp_op = random.choice(temp_ready_ops)

                        temp_seq[temp_op] = None

                        temp_stack.remove(temp_op)

                        for output in temp_op.outputs:
                            for next_op in output.consumers:
                                temp_stack.append(next_op) if next_op not in temp_stack else temp_stack
                        
                        #check if op input tensors are ready
                        temp_not_ready = set([])
                        for next_op in temp_stack:
                            for in_tensor in next_op.inputs:
                                for producer in in_tensor.producers:
                                    if producer not in temp_seq:
                                        temp_not_ready.add(next_op) if next_op not in temp_not_ready else temp_not_ready

                        temp_ready_ops = temp_stack[:]
                        for temp_op in temp_not_ready:
                            temp_ready_ops.remove(temp_op)

                    first_op_in_split = list(op_order.keys())[-1]
                    last_op_in_split = temp_stack[0]

                    ####
                    # Find Depth Order
                    ####
                    reverse_op_order: Dict[xir.Operator, None] = {}

                    reverse_stack = []
        
                    # dependency counts to be used to resolve ops that have multiple consumers
                    dependency_counts: Dict[xir.Operator, int] = {}

                    reverse_op_order[last_op_in_split] = None
                    for tin in sorted(last_op_in_split.inputs, key=lambda t: t.size):
                        reverse_stack.extend(tin.producers)

                    while reverse_stack:
                        op = reverse_stack.pop(-1)
                        if op in reverse_op_order:
                            # op already scheduled
                            continue

                        if op not in dependency_counts:
                            # this is the first time we see this op, so count the dependencies
                            dependency_counts[op] = len(
                                {c for t in op.outputs for c in t.consumers}
                            )

                        if dependency_counts[op] <= 0:
                            raise Exception(
                                "Found operator with 0 or fewer dependencies (the graph may be corrupted)"
                            )

                        dependency_counts[op] -= 1
                        if dependency_counts[op]:
                            # skip scheduling of op if there are other dependents
                            continue

                        reverse_op_order[op] = None
                        if op!=first_op_in_split:
                            for tin in sorted(op.inputs, key=lambda t: t.size):
                                reverse_stack.extend(tin.producers)


                    depth_op_order = dict(reversed(list(reverse_op_order.items())))

                    ####
                    # Find Breadth Order
                    ####
                    breadth_op_order: Dict[xir.Operator, None] = {}

                    breadth_queue = []
        
                    
                    breadth_op_order[first_op_in_split] = None
                    for tout in sorted(first_op_in_split.outputs, key=lambda t: t.size):
                        breadth_queue.extend(tout.consumers)

                    while breadth_queue:
                        op = breadth_queue.pop(0)

                        not_ready = False
                        for in_tensor in op.inputs:
                                for producer in in_tensor.producers:
                                    if producer not in breadth_op_order:
                                        not_ready = True
                                        
                        if not_ready:
                            breadth_queue.append(op)
                            continue
                                        
                        if op in breadth_op_order:
                            # op already scheduled
                            continue

                        breadth_op_order[op] = None
                        if op!=last_op_in_split:
                            for tout in sorted(op.outputs, key=lambda t: t.size):
                                breadth_queue.extend(tout.consumers)

                    ####
                    # Find Greedy Order TensorsSizeMinimizerPlanner
                    ####
                    greedy_op_order: Dict[xir.Operator, None] = {}

                    outputs_size=0
                    for output in first_op_in_split.outputs:
                        outputs_size += output.size

                    tensors_size_increase: Dict[xir.Operator, int] = {first_op_in_split: outputs_size-0}
                    not_ready = []
                    ready_ops = tensors_size_increase
                    for op in not_ready:
                        ready_ops.pop(op)
                    #find op that causes the smallest increase in tensor arena
                    op = min(ready_ops, key=ready_ops.get)
                    while op != last_op_in_split:
                        greedy_op_order[op] = None
                        tensors_size_increase.pop(op)
                        for output in op.outputs:
                            for next_op in output.consumers:
                                outputs_size=0
                                inputs_size =0
                                for output in next_op.outputs:
                                    outputs_size += output.size 
                                for in_tensor in next_op.inputs:
                                    inputs_size += in_tensor.size 
                                tensors_size_increase[next_op]= outputs_size-inputs_size

                        not_ready = set([])
                        #check if op input tensors are ready
                        for next_op in tensors_size_increase:
                            for in_tensor in next_op.inputs:
                                for producers in in_tensor.producers:
                                    if producers not in greedy_op_order:
                                        not_ready.add(next_op)
                                        
                        ready_ops = tensors_size_increase
                        for op in not_ready:
                            ready_ops.pop(op)
                        #find op that causes the smallest increase in tensor arena
                        op = min(ready_ops, key=ready_ops.get)

                    greedy_op_order[last_op_in_split] = None

                    ####
                    # Evaluate seqs
                    ####
                    best_op_order = greedy_op_order

                    greedy_arena_size = update_arena_size(greedy_op_order)

                    best_arena_size = greedy_arena_size

                    depth_arena_size = update_arena_size(depth_op_order)
                    
                    if depth_arena_size < best_arena_size:
                        best_op_order = depth_op_order
                        best_arena_size = depth_arena_size

                    # eval_op_order = breadth_op_order
                    # current_arena_size =0
                    # breadth_arena_size = -2**15 
                    # for op_index in range(len(eval_op_order)):
                    #     current_op = list(eval_op_order.keys())[op_index]
                    #     current_arena_size += current_op.outputs[0].size 
                    #     for in_tensor in current_op.inputs[1:]:
                    #         current_arena_size -= in_tensor.size

                    #     if current_arena_size > breadth_arena_size:
                    #         breadth_arena_size = current_arena_size  
                        
                    #     # if all the consumers have been scheduled
                    #     for in_tensor in current_op.inputs:
                    #         tensor_deactivated = True
                    #         for operator in in_tensor.consumers:
                    #             if operator not in list(eval_op_order.keys())[0:op_index+1]: 
                    #                 tensor_deactivated = False
                    #         if tensor_deactivated:
                    #             current_arena_size -= in_tensor.size

                    # if breadth_arena_size < best_arena_size:
                    #     best_op_order = breadth_op_order
                    #     best_arena_size = breadth_arena_size
                        

                    # best_arena_size = 999999 
                    # # number of exp
                    # for _ in range(100000):
                    #     random_op_order: Dict[xir.Operator, None] = {}
                    #     random_op_order[first_op_in_split] = None
                    #     current_arena_size = first_op_in_split.outputs[0].size
                    #     max_arena_size = current_arena_size
                    
                    #     for op_index in range(len(eval_op_order)-1):

                    #         new_ready = False
                    #         while not new_ready:
                    #             current_op = list(eval_op_order.keys())[random.randint(0,len(eval_op_order)-1)]
                    #             new_ready = True
                                
                    #             if current_op in random_op_order:
                    #                 new_ready = False 
                    #                 continue
                                
                    #             for in_tensor in current_op.inputs:
                    #                     for producer in in_tensor.producers:
                    #                         if producer not in random_op_order:
                    #                             new_ready = False 
                    #                             continue
                            
                    #         random_op_order[current_op] = None

                    #         current_arena_size,max_arena_size = update_arena_size(current_arena_size,current_op,max_arena_size,random_op_order)
                        
                    #     if max_arena_size < best_arena_size:
                    #         best_arena_size = max_arena_size
                    #         best_op_order = random_op_order

                    op_stack = temp_stack[:]
                    ready_ops = temp_ready_ops

                    op_order.update(best_op_order)

        # remove sentinel op
        self._graph.remove_operator(sentinel_op)
        del op_order[sentinel_op]

        # return ops 
        return list(op_order.keys())
