# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

from abc import ABC, abstractmethod
from asyncio import base_futures
from itertools import permutations
import random
from typing import Sequence, Dict
from scipy import rand

from tflite2xcore import xcore_schema as xir
from tflite2xcore.xcore_schema import BuiltinOpCodes
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

def update_arena_size(op_order):
    current_arena_size = list(op_order.keys())[0].outputs[0].size
    max_arena_size = -2**15 
    for op_index in range(1,len(op_order)):
        current_op = list(op_order.keys())[op_index]
        if current_op.operator_code.code != BuiltinOpCodes.PAD:
                       
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
            


class DepthVsGreedyPlanner(ExecutionPlanner):
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
                        temp_op = temp_ready_ops[0]

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

                    # del temp_seq[sentinel_op]
                    for op in list(op_order.keys())[:-1]:
                        del temp_seq[op]
                    temp_seq[last_op_in_split] = None

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
                    # Find Greedy Order 
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

                    op_stack = temp_stack[:]
                    ready_ops = temp_ready_ops

                    op_order.update(best_op_order)

        # remove sentinel op
        self._graph.remove_operator(sentinel_op)
        del op_order[sentinel_op]

        # return ops 
        return list(op_order.keys())
    
def permute(temp_seq):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    def backtrack(first = 0,scheduled=[],temp_seq=temp_seq):
        temp_seq_keys=list(temp_seq.keys())
        # if all integers are used up
        if first == n:  
            perms.append(nums[:])
            scheduled=[]
        for i in range(first, n):
            possible_ops = []
            scheduled_ops = [temp_seq_keys[i] for i in scheduled]
            
            for op in scheduled_ops:
                for output in op.outputs:
                        for next_op in output.consumers:
                            possible_ops.append(next_op) if next_op not in possible_ops else possible_ops

            #check if op input tensors are ready
            not_ready = set([])
            for next_op in possible_ops:
                    for in_tensor in next_op.inputs:
                        for producers in in_tensor.producers:
                            if producers not in scheduled_ops:
                                not_ready.add(next_op) if next_op not in not_ready else not_ready

            ready_ops = possible_ops
            for op in not_ready:
                ready_ops.remove(op)
                
            if not scheduled:
                ready_ops = [temp_seq_keys[0]]
                
            if (temp_seq_keys[nums[i]] in ready_ops):
                scheduled.append(nums[i])
                # place i-th integer first 
                # in the current permutation
                nums[first], nums[i] = nums[i], nums[first]
                # use next integers to complete the permutations
                backtrack(first + 1,scheduled)
                # backtrack
                scheduled.pop(-1) 
                nums[first], nums[i] = nums[i], nums[first]
            else:
                scheduled=[]	
                break

def permute_pre(list_producers):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    def backtrack(first = 0,scheduled=[],list_producers=list_producers):
        # if all integers are used up
        if first == n:  
            perms.append(nums[:])
            scheduled=[]
        for i in range(first, n):
            # scheduled_ops = [temp_seq_keys[i] for i in scheduled]
                
            # if (temp_seq_keys[nums[i]] in ready_ops) or not scheduled:
            # no producers or producers are scheduled 
            if (not list_producers[nums[i]]) or all( item in scheduled for item in list_producers[nums[i]] ):
                scheduled.append(nums[i])
                # place i-th integer first 
                # in the current permutation
                nums[first], nums[i] = nums[i], nums[first]
                # use next integers to complete the permutations
                backtrack(first + 1,scheduled)
                # backtrack
                scheduled.pop(-1) 
                nums[first], nums[i] = nums[i], nums[first]
            else:
                scheduled=[]	
                break
            

    nums = list(range(len(list_producers)))
    n = len(nums)
    perms = []
    backtrack()

    return perms

class PermutationsPlanner(ExecutionPlanner):
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
                    # temp_seq: Dict[xir.Operator, None] = op_order.copy()
                    temp_seq = list(op_order.keys())
                    temp_stack = op_stack[:]
                    temp_ready_ops = ready_ops[:]
                    temp_not_ready = set([])
                    list_producers = [[]]

                    while len(temp_stack)>1:
                        temp_op = temp_ready_ops[0]

                        temp_seq.append(temp_op)

                        op_producers = []
                        for in_tensor in temp_op.inputs:
                            for producer in in_tensor.producers:
                                op_producers.append(temp_seq.index(producer)-1)

                        list_producers.append(op_producers)

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

                    last_op_in_split = temp_stack[0]
                    
                    
                   

                    # for op in list(op_order.keys())[:-1]:
                    #     del temp_seq[op]
                    # temp_seq[last_op_in_split] = None

                    ####
                    # Find Pemutations
                    ####
                    permutations = permute_pre(list_producers)
                    # permutations = permute(temp_seq)

                    dict_for_size: Dict[xir.Operator, None] = {}
                    best_arena_size = 2**31
                    for perm in permutations: 
                        for i in perm:
                            dict_for_size[temp_seq[i]] = None    
                        perm_arena_size = update_arena_size(dict_for_size)
                        # print(perm_arena_size)
                        if perm_arena_size <= best_arena_size:
                            best_op_order = dict_for_size
                            best_arena_size = perm_arena_size
                        dict_for_size = {}    

                    op_stack = temp_stack[:]
                    ready_ops = temp_ready_ops

                    op_order.update(best_op_order)

        # remove sentinel op
        self._graph.remove_operator(sentinel_op)
        del op_order[sentinel_op]

        # return ops 
        return list(op_order.keys())
