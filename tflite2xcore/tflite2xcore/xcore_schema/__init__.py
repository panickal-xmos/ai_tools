# Copyright (c) 2018-2020, XMOS Ltd, All rights reserved

from . import flexbuffers
from .ir_object import _IRObject
from .tensor_type import TensorType
from .op_codes import BuiltinOpCodes, ExternalOpCodes, XCOREOpCodes
from .operator_code import OperatorCode, ValidOpCodes, CustomOpCodes
from .buffer import Buffer, _BufferDataType, _BufferOwnerContainer
from .operator import _OpOptionsType, Operator
from .tensor import Tensor, _ShapeInputType
from .subgraph import Subgraph
from .metadata import Metadata

from .xcore_schema import (
    QuantizationDetails,
    ActivationFunctionType,
    FullyConnectedOptionsWeightsFormat,
    Padding,
    BuiltinOptions,
)

from .xcore_model import XCOREModel

from . import xcore_model
