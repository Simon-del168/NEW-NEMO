# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from dataclasses import dataclass
from typing import Optional, List
from qti.aisw.core.model_level_api.config.sdk_config import Config, RunConfig, GenerateConfig


@dataclass
class QNNCommonConfig(Config):
    log_level: Optional[str] = None

    def as_command_line_args(self):
        return f'--log_level {self.log_level}' if self.log_level else ''


@dataclass
class QNNRunConfig(QNNCommonConfig, RunConfig):
    profiling_level: Optional[str] = None
    batch_multiplier: Optional[int] = None
    use_native_output_data: Optional[bool] = None
    use_native_input_data: Optional[bool] = None
    native_input_tensor_names: Optional[List[str]] = None
    synchronous: Optional[bool] = None

    def as_command_line_args(self):
        qnn_common_config_args = super().as_command_line_args()
        profile_arg = f'--profiling_level {self.profiling_level}' if self.profiling_level else ''
        batch_arg = f'--batch_multiplier {self.batch_multiplier}' if self.batch_multiplier else ''
        native_output_data_arg = '--use_native_output_files' if self.use_native_output_data else ''
        native_input_data_arg = '--use_native_input_files' if self.use_native_input_data else ''
        native_input_tensor_names_arg = '--native_input_tensor_names ' \
            f"{','.join(self.native_input_tensor_names)}" if self.native_input_tensor_names else ''
        synchronous_arg = '--synchronous' if self.synchronous else ''

        return f'{qnn_common_config_args} {profile_arg} {batch_arg} {native_output_data_arg} ' \
               f'{native_input_data_arg} {native_input_tensor_names_arg} {synchronous_arg}'


@dataclass
class QNNGenerateConfig(QNNCommonConfig, GenerateConfig):
    def as_command_line_args(self):
        return super().as_command_line_args()
