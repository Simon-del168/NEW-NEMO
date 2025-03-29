# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


class ConversionFailure(Exception):
    pass


class OptimizationFailure(Exception):
    pass


class QuantizationFailure(Exception):
    pass


class GraphPreparationFailure(Exception):
    pass


class ExecutionFailure(Exception):
    pass
