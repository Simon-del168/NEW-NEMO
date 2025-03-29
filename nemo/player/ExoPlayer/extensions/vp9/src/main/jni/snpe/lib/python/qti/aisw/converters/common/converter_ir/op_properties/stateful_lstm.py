# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""StatefulLstm op properties

:var NUM_INPUTS: Number of inputs.
:type NUM_INPUTS: int
:var NUM_PARAMS: Number of parameters.
:type NUM_PARAMS: int
"""

NUM_INPUTS = 9
NUM_REQUIRED_INPUTS = 3
NUM_OUTPUTS = 3

# Indices of inputs after initial conversion to RolledLstm node
IR_NUM_INPUTS = 11
IR_INPUT_IDX = 0
IR_INITIAL_H_IDX = 1
IR_INITIAL_C_IDX = 2
IR_INPUT_WEIGHTS_IDX = 3
IR_HIDDEN_STATE_WEIGHTS_IDX = 4
IR_GATE_BIASES_IDX = 5
IR_NORM_WEIGHTS_IDX = 6
IR_CELL_STATE_WEIGHTS_IDX = 7
IR_PROJ_WEIGHTS_IDX = 8
IR_PROJ_BIAS_IDX = 9
IR_RESET_IDX = 10

# Map from IR indices to ONNX indices
IR_TO_ONNX_INDICES = {
    IR_INPUT_IDX : 0,
    IR_INPUT_WEIGHTS_IDX : 1,
    IR_HIDDEN_STATE_WEIGHTS_IDX : 2,
    IR_GATE_BIASES_IDX : 3,
    # ONNX input 4 is "sequence_lens", which is unused
    IR_INITIAL_H_IDX : 5,
    IR_INITIAL_C_IDX : 6,
    # Projection weights and biases combined in ONNX
    IR_PROJ_WEIGHTS_IDX : 7,
    IR_PROJ_BIAS_IDX : 7,
    IR_RESET_IDX : 8,
}
