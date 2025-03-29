# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from abc import ABC, abstractmethod


class Config(ABC):
    @abstractmethod
    def as_command_line_args(self):
        pass


class RunConfig(Config, ABC):
    def as_command_line_args(self):
        return ''


class GenerateConfig(Config, ABC):
    def as_command_line_args(self):
        return ''
