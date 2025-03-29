# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from enum import Enum
from qti.aisw.tools.core.modules.api.definitions.common import BackendType


# TODO: Switch to using Enum from DeviceAPI once all platforms are supported
class DevicePlatformType(str, Enum):
    """
    Enum representing known device platforms.
    """

    ANDROID = "android"
    X86_64_LINUX = "x86_64-linux_clang"
