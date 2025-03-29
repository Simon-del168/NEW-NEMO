# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import numpy as np

QNN_DTYPE_NUMPY_DTYPE_MAP = {
    "0x8": np.int8, # QNN_DATATYPE_INT_8 = 0x0008,
    "0x16": np.int16, # QNN_DATATYPE_INT_16 = 0x0016,
    "0x32": np.int32, # QNN_DATATYPE_INT_32 = 0x0032,
    "0x64": np.int64, # QNN_DATATYPE_INT_64 = 0x0064,

    "0x108": np.uint8, # QNN_DATATYPE_UINT_8  = 0x0108,
    "0x116": np.uint16, # QNN_DATATYPE_UINT_16 = 0x0116,
    "0x132": np.uint32, # QNN_DATATYPE_UINT_32 = 0x0132,
    "0x164": np.uint64, # QNN_DATATYPE_UINT_64 = 0x0164,

    "0x216": np.float16, # QNN_DATATYPE_FLOAT_16 = 0x0216,
    "0x232": np.float32, # QNN_DATATYPE_FLOAT_32 = 0x0232,
    "0x264": np.float64, # QNN_DATATYPE_FLOAT_64 = 0x0264,

    "0x304": None, # QNN_DATATYPE_SFIXED_POINT_4  = 0x0304,
    "0x308": np.int8, # QNN_DATATYPE_SFIXED_POINT_8  = 0x0308,
    "0x316": np.int16, # QNN_DATATYPE_SFIXED_POINT_16 = 0x0316,
    "0x332": np.int32, # QNN_DATATYPE_SFIXED_POINT_32 = 0x0332,

    "0x404": None, # QNN_DATATYPE_UFIXED_POINT_4  = 0x0404,
    "0x408": np.uint8, # QNN_DATATYPE_UFIXED_POINT_8  = 0x0408,
    "0x416": np.uint16, # QNN_DATATYPE_UFIXED_POINT_16 = 0x0416,
    "0x432": np.uint32, # QNN_DATATYPE_UFIXED_POINT_32 = 0x0432,

    "0x508": np.bool_ # QNN_DATATYPE_BOOL_8 = 0x0508
}

def verify_path(*paths):
    '''
    Verifies whether directory exists or not, if not then
    creates the directory and returns the path
    '''
    path = os.path.join(*paths)
    if os.path.exists(path) and os.path.isdir(path):
        return path
    os.makedirs(path)
    return path
