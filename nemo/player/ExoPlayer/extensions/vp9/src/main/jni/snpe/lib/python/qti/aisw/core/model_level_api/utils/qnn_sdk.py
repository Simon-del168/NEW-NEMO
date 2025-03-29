# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from os import getenv, path


def qnn_sdk_root():
    sdk_root = getenv('QNN_SDK_ROOT')
    if sdk_root is None:
        sdk_root = path.dirname(path.abspath(__file__ + '/../../../../../../../'))
    return sdk_root
