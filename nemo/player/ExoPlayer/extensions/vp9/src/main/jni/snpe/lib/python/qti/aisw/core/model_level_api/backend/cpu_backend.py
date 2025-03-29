# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.core.model_level_api.backend.backend import Backend
from qti.aisw.core.model_level_api.workflow.workflow import WorkflowMode
from qti.aisw.core.model_level_api.target.x86 import X86Target


class CpuBackend(Backend):
    def __init__(self, target=None):
        super().__init__(target)
        if target is None:
            self.target = X86Target()

    @property
    def backend_library(self):
        return 'libQnnCpu.so'

    @property
    def backend_extensions_library(self):
        # no backend extensions support
        return None

    def get_required_artifacts(self, sdk_root):
        return []

    def _workflow_mode_setter_hook(self, mode):
        if mode == WorkflowMode.CONTEXT_BINARY_GENERATION:
            raise ValueError("CPU backend does not support context binary generation")
