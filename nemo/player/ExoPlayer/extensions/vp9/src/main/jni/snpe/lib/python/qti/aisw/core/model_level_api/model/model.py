# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self, name):
        self.name = name
