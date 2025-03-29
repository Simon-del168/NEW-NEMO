# =============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from .native_executor import py_net_run

NetRunErrorCode = py_net_run.NetRunErrorCode


class InferenceError(Exception):
    def __init__(self, error_code, message):
        self.error_code = error_code
        self.message = message


class ContextBinaryGenerationError(Exception):
    def __init__(self, error_code, message):
        self.error_code = error_code
        self.message = message


def return_code_to_netrun_error_enum(error_code):
    if error_code in NetRunErrorCode.__members__.values():
        return NetRunErrorCode(error_code)
    else:
        return None
