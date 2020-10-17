# Copyright (c) 2020, XMOS Ltd, All rights reserved

import enum
import aenum
from typing import Type


class BuiltinOpCodes(enum.IntEnum):
    ADD: BuiltinOpCodes
    AVERAGE_POOL_2D: BuiltinOpCodes
    CONCATENATION: BuiltinOpCodes
    CONV_2D: BuiltinOpCodes
    DEPTHWISE_CONV_2D: BuiltinOpCodes
    DEPTH_TO_SPACE: BuiltinOpCodes
    DEQUANTIZE: BuiltinOpCodes
    EMBEDDING_LOOKUP: BuiltinOpCodes
    FLOOR: BuiltinOpCodes
    FULLY_CONNECTED: BuiltinOpCodes
    HASHTABLE_LOOKUP: BuiltinOpCodes
    L2_NORMALIZATION: BuiltinOpCodes
    L2_POOL_2D: BuiltinOpCodes
    LOCAL_RESPONSE_NORMALIZATION: BuiltinOpCodes
    LOGISTIC: BuiltinOpCodes
    LSH_PROJECTION: BuiltinOpCodes
    LSTM: BuiltinOpCodes
    MAX_POOL_2D: BuiltinOpCodes
    MUL: BuiltinOpCodes
    RELU: BuiltinOpCodes
    RELU_N1_TO_1: BuiltinOpCodes
    RELU6: BuiltinOpCodes
    RESHAPE: BuiltinOpCodes
    RESIZE_BILINEAR: BuiltinOpCodes
    RNN: BuiltinOpCodes
    SOFTMAX: BuiltinOpCodes
    SPACE_TO_DEPTH: BuiltinOpCodes
    SVDF: BuiltinOpCodes
    TANH: BuiltinOpCodes
    CONCAT_EMBEDDINGS: BuiltinOpCodes
    SKIP_GRAM: BuiltinOpCodes
    CALL: BuiltinOpCodes
    CUSTOM: BuiltinOpCodes
    EMBEDDING_LOOKUP_SPARSE: BuiltinOpCodes
    PAD: BuiltinOpCodes
    UNIDIRECTIONAL_SEQUENCE_RNN: BuiltinOpCodes
    GATHER: BuiltinOpCodes
    BATCH_TO_SPACE_ND: BuiltinOpCodes
    SPACE_TO_BATCH_ND: BuiltinOpCodes
    TRANSPOSE: BuiltinOpCodes
    MEAN: BuiltinOpCodes
    SUB: BuiltinOpCodes
    DIV: BuiltinOpCodes
    SQUEEZE: BuiltinOpCodes
    UNIDIRECTIONAL_SEQUENCE_LSTM: BuiltinOpCodes
    STRIDED_SLICE: BuiltinOpCodes
    BIDIRECTIONAL_SEQUENCE_RNN: BuiltinOpCodes
    EXP: BuiltinOpCodes
    TOPK_V2: BuiltinOpCodes
    SPLIT: BuiltinOpCodes
    LOG_SOFTMAX: BuiltinOpCodes
    DELEGATE: BuiltinOpCodes
    BIDIRECTIONAL_SEQUENCE_LSTM: BuiltinOpCodes
    CAST: BuiltinOpCodes
    PRELU: BuiltinOpCodes
    MAXIMUM: BuiltinOpCodes
    ARG_MAX: BuiltinOpCodes
    MINIMUM: BuiltinOpCodes
    LESS: BuiltinOpCodes
    NEG: BuiltinOpCodes
    PADV2: BuiltinOpCodes
    GREATER: BuiltinOpCodes
    GREATER_EQUAL: BuiltinOpCodes
    LESS_EQUAL: BuiltinOpCodes
    SELECT: BuiltinOpCodes
    SLICE: BuiltinOpCodes
    SIN: BuiltinOpCodes
    TRANSPOSE_CONV: BuiltinOpCodes
    SPARSE_TO_DENSE: BuiltinOpCodes
    TILE: BuiltinOpCodes
    EXPAND_DIMS: BuiltinOpCodes
    EQUAL: BuiltinOpCodes
    NOT_EQUAL: BuiltinOpCodes
    LOG: BuiltinOpCodes
    SUM: BuiltinOpCodes
    SQRT: BuiltinOpCodes
    RSQRT: BuiltinOpCodes
    SHAPE: BuiltinOpCodes
    POW: BuiltinOpCodes
    ARG_MIN: BuiltinOpCodes
    FAKE_QUANT: BuiltinOpCodes
    REDUCE_PROD: BuiltinOpCodes
    REDUCE_MAX: BuiltinOpCodes
    PACK: BuiltinOpCodes
    LOGICAL_OR: BuiltinOpCodes
    ONE_HOT: BuiltinOpCodes
    LOGICAL_AND: BuiltinOpCodes
    LOGICAL_NOT: BuiltinOpCodes
    UNPACK: BuiltinOpCodes
    REDUCE_MIN: BuiltinOpCodes
    FLOOR_DIV: BuiltinOpCodes
    REDUCE_ANY: BuiltinOpCodes
    SQUARE: BuiltinOpCodes
    ZEROS_LIKE: BuiltinOpCodes
    FILL: BuiltinOpCodes
    FLOOR_MOD: BuiltinOpCodes
    RANGE: BuiltinOpCodes
    RESIZE_NEAREST_NEIGHBOR: BuiltinOpCodes
    LEAKY_RELU: BuiltinOpCodes
    SQUARED_DIFFERENCE: BuiltinOpCodes
    MIRROR_PAD: BuiltinOpCodes
    ABS: BuiltinOpCodes
    SPLIT_V: BuiltinOpCodes
    UNIQUE: BuiltinOpCodes
    CEIL: BuiltinOpCodes
    REVERSE_V2: BuiltinOpCodes
    ADD_N: BuiltinOpCodes
    GATHER_ND: BuiltinOpCodes
    COS: BuiltinOpCodes
    WHERE: BuiltinOpCodes
    RANK: BuiltinOpCodes
    ELU: BuiltinOpCodes
    REVERSE_SEQUENCE: BuiltinOpCodes
    MATRIX_DIAG: BuiltinOpCodes
    QUANTIZE: BuiltinOpCodes
    MATRIX_SET_DIAG: BuiltinOpCodes
    ROUND: BuiltinOpCodes
    HARD_SWISH: BuiltinOpCodes
    IF: BuiltinOpCodes
    WHILE: BuiltinOpCodes
    NON_MAX_SUPPRESSION_V4: BuiltinOpCodes
    NON_MAX_SUPPRESSION_V5: BuiltinOpCodes
    SCATTER_ND: BuiltinOpCodes
    SELECT_V2: BuiltinOpCodes
    DENSIFY: BuiltinOpCodes
    SEGMENT_SUM: BuiltinOpCodes
    BATCH_MATMUL: BuiltinOpCodes


class ExternalOpCodes(aenum.Enum):  # type: ignore
    LceQuantize: ExternalOpCodes
    LceBconv2d: ExternalOpCodes
    LceDequantize: ExternalOpCodes

    @classmethod
    def add_new_opcode(cls: Type["ExternalOpCodes"], name: str) -> "ExternalOpCodes":
        ...


class XCOREOpCodes(enum.Enum):
    ...
