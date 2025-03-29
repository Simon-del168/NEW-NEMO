# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
from pathlib import Path

from qti.aisw.core.model_level_api.backend.backend import Backend
from qti.aisw.core.model_level_api.workflow.workflow import WorkflowMode
from qti.aisw.core.model_level_api.target.android import AndroidTarget
from qti.aisw.core.model_level_api.target.x86 import X86Target


class HtpBackend(Backend):
    def __init__(self, target=None, config_file=None, config_dict=None):
        super().__init__(target)
        if config_file:
            with open(config_file, 'r') as f:
                self._config = json.load(f)

        if config_dict:
            self._config.update(config_dict)

    def _workflow_mode_setter_hook(self, mode):
        if self._default_target:
            if mode == WorkflowMode.INFERENCE:
                self.target = AndroidTarget()
            elif mode == WorkflowMode.CONTEXT_BINARY_GENERATION:
                self.target = X86Target()
            else:
                raise ValueError('Invalid workflow_mode: ', mode)

    @property
    def backend_library(self):
        return "libQnnHtp.so"

    @property
    def backend_extensions_library(self):
        return "libQnnHtpNetRunExtensions.so"

    def get_required_artifacts(self, sdk_root):
        artifacts = []
        if self.target.target_name == 'aarch64-android':
            android_lib_dir = sdk_root + '/lib/' + self.target.target_name + '/'
            android_libs = [android_lib_dir + 'libQnnHtpPrepare.so']
            hexagon_libs = []

            # push all known stub/skel combinations to device
            # todo: add user-configuration for hexagon arch or do platform detection to avoid
            # pushing all stubs and skels
            htp_archs = ['v68', 'v69', 'v73', 'v75', 'v79']
            for arch in htp_archs:
                htp_arch_stub = android_lib_dir + 'libQnnHtp' + arch.upper() + 'Stub.so'
                if Path(htp_arch_stub).is_file():
                    android_libs.append(htp_arch_stub)

                hexagon_arch_lib_dir = sdk_root + '/lib/hexagon-' + arch + '/unsigned/'
                htp_arch_skel = hexagon_arch_lib_dir + 'libQnnHtp' + arch.upper() + 'Skel.so'
                if Path(htp_arch_stub).is_file():
                    hexagon_libs.append(htp_arch_skel)

            artifacts.extend(android_libs)
            artifacts.extend(hexagon_libs)
        return artifacts
