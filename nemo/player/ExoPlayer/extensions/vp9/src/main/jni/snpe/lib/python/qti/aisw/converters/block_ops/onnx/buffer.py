# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""ONNX Buffer Block Operator code"""

import onnx
from onnxscript import script
from onnxscript.onnx_types import BOOL
from onnxscript.values import OnnxFunction
from qti.aisw.converters.block_ops.onnx.onnx_block_op_base import QnnOnnxBlockOp
from qti.aisw.converters.common.converter_ir.op_properties.buffer import (
    BUFFER_MODE_DEFAULT_VAL,
    BUFFER_STRIDE_DEFAULT_VAL,
)


class Buffer(QnnOnnxBlockOp):
    """ONNX Buffer Block Operator class

    This class represents a Buffer BlockOperator in ONNX. This operator is
    defined in onnxscript, so both getOnnxScriptFunc and getOnnxFuncProto are
    implemented.

    :ivar min_opset: Initialized to 8.
    :ivar min_opset: Initialized to 21.
    """

    def __init__(self, onnx_opset_version: int, aisw_opset_version: int = 1):
        self.name = "Buffer"
        self.min_opset = 8
        self.max_opset = 21
        super(Buffer, self).__init__(onnx_opset_version, aisw_opset_version)

    def getOnnxScriptFunc(self) -> OnnxFunction:
        opset = self.opset
        qti_aisw = self.aisw_opset

        @script(qti_aisw, default_opset=opset)
        def Buffer(
            input,
            reset: BOOL = opset.Constant(value=False),
            # Attributes
            buffer_size: int = None,
            buffer_dim: int = None,
            stride: int = BUFFER_STRIDE_DEFAULT_VAL,
            mode: int = BUFFER_MODE_DEFAULT_VAL,
        ):
            """Buffer operator in ONNX script.

            This is a dummy implementation for running it on our test framework.
            The output will not match the QNN op def and it should not be used
            as a golden reference implementation.
            """
            return opset.Identity(input)

        return Buffer

    def getOnnxFuncProto(self) -> onnx.FunctionProto:
        return self.getOnnxScriptFunc().to_function_proto()
