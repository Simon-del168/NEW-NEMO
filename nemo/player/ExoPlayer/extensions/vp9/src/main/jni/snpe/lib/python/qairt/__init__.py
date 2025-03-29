# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

__version__ = '0.1.0b1'

from .api import Model, compile, import_model
from .enums import BackendType, DevicePlatformType
from .params import InputTensorConfig, OutputTensorConfig, ConversionParams, QuantizationParams, CompilationParams, ExecutionParams, Target

import logging

qairt_logger = logging.getLogger('qairt_api')
