# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""ONNX StatefulLstm Block Operator code"""

import onnx
from onnx import helper
from onnxscript import script
from onnxscript.values import OnnxFunction
from onnxscript.onnx_types import BOOL
from qti.aisw.converters.block_ops.onnx.onnx_block_op_base import (
    QnnOnnxBlockOp,
    qcom_block_op_domain,
)
from qti.aisw.converters.common.converter_ir.op_properties.stateful_lstm import (
    IR_RESET_IDX,
    IR_TO_ONNX_INDICES,
)
ONNX_RESET_IDX = IR_TO_ONNX_INDICES[IR_RESET_IDX]

class StatefulLstm(QnnOnnxBlockOp):
    """ONNX StatefulLstm Block Operator class

    This class represents a StatefulLstm BlockOperator in ONNX. This operator is
    defined in onnxscript, so both getOnnxScriptFunc and getOnnxFuncProto are
    implemented.

    :ivar min_opset: Initialized to 10.
    :ivar min_opset: Initialized to 11.
    """

    def __init__(self, onnx_opset_version: int, aisw_opset_version: int = 1):
        self.name = "StatefulLstm"
        self.min_opset = 10
        self.max_opset = 11
        super(StatefulLstm, self).__init__(onnx_opset_version, aisw_opset_version)

    def getOnnxScriptFunc(self) -> OnnxFunction:
        opset = self.opset
        qti_aisw = self.aisw_opset

        @script(qti_aisw, default_opset=opset)
        def StatefulLstm(
            X,
            W,
            R,
            hidden_size: int,
            B=None,
            sequence_lens=None,
            initial_h=None,
            initial_c=None,
            P=None,
            # This parameter is always false in ONNX runtime
            reset: BOOL = opset.Constant(value=False),
            # Attributes
            clip: float = None,
            direction: str = "forward",
            input_forget: int = 0,
        ):
            """See ONNX LSTM documentation for op overview.

            StatefulLstm adds a reset input in addition to the standard ONNX
            LSTM inputs. See QNN MasterOpDef for documentation on the behavior
            of the reset input.

            NOTE: This op currently does not have the reset functionality in
            ONNX runtime.
            """
            return opset.LSTM(
                X,
                W,
                R,
                B=B,
                sequence_lens=sequence_lens,
                initial_h=initial_h,
                initial_c=initial_c,
                P=P,
                clip=clip,
                direction=direction,
                hidden_size=hidden_size,
                input_forget=input_forget,
            )

        return StatefulLstm

    def getOnnxFuncProto(self) -> onnx.FunctionProto:
        return self.getOnnxScriptFunc().to_function_proto()


def replaceOnnxLstmWithBlockOp(node: onnx.NodeProto):
    """Given an ONNX node, convert it to a StatefulLstm Block Op node if it is
       an ONNX LSTM node . `reset` input is added to the node, with the name
       <node_name>_reset.

    :param node: The node to modify.
    :type node: onnx.NodeProto
    :return: The modified node.
    :rtype: onnx.NodeProto
    :raises ValueError: If an unknown or malformed ONNX LSTM node is found while
        trying to transform.

    """

    if node.op_type != "LSTM":
        return node
    else:
        num_empty_args = ONNX_RESET_IDX - len(node.input)
        if num_empty_args < 0:
            raise ValueError(
                "Too many arguments provided to ONNX LSTM "
                f"node {node.name}. Expected at most "
                f"{ONNX_RESET_IDX} arguments, got "
                f"{len(node.input)}."
            )
        node.input.extend([""] * num_empty_args)
        node.input.extend([f"{node.name}_reset"])
        node.op_type = "StatefulLstm"
        node.domain = qcom_block_op_domain()
        print(f"Added reset tensor ({node.name}_reset) to {node.name}")
        return node


def replaceAllOnnxLstmWithBlockOp(model: onnx.ModelProto):
    """Given an onnx ModelProto, all ONNX LSTM nodes are replaced with StatefulLstm
       BlockOps. `reset` inputs are added to each StatefulLstm node, with the
       name <node_name>_reset. StatefulLstm ONNX function proto is added to the
       model if any nodes were replaced.

    :param model: The model to modify.
    :type model: onnx.ModelProto
    :return: The modified model.
    :rtype: onnx.ModelProto
    :raises ValueError: If an unknown or malformed ONNX LSTM node is found while
        trying to transform.
    :raises ValueError: If an ONNX opset version cannot be found in the model.
    """
    replaced_lstm_node = False
    for node in model.graph.node:
        if node.op_type == "LSTM":
            node = replaceOnnxLstmWithBlockOp(node)
            reset_name = node.input[ONNX_RESET_IDX]
            model.graph.input.append(
                onnx.helper.make_tensor_value_info(reset_name, 9, [])
            )
            replaced_lstm_node = True
    if replaced_lstm_node:
        model_version = None
        for opset in model.opset_import:
            # Empty string is ONNX domain
            if opset.domain == "":
                model_version = opset.version
                break
        if not model_version:
            raise ValueError("ONNX opset version could not be found in model")
        model.functions.extend([StatefulLstm(model_version).getOnnxFuncProto()])
        model.opset_import.extend([helper.make_opsetid(qcom_block_op_domain(), 1)])
        onnx.checker.check_model(model)
    return model
