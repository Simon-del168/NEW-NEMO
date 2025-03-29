# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
import logging

from qti.aisw.core.model_level_api.backend.backend import Backend, BackendConfig
from qti.aisw.core.model_level_api.target.android import AndroidTarget

logger = logging.getLogger(__name__)


class GpuBackend(Backend):
    graph_names = BackendConfig()
    precision_mode = BackendConfig()
    disable_memory_optimizations = BackendConfig()
    disable_node_optimizations = BackendConfig()
    kernel_repo_path = BackendConfig()
    invalidate_kernel_repo = BackendConfig()
    disable_queue_recording = BackendConfig()
    perf_hint = BackendConfig()

    def __init__(self, target=None, config_file=None, config_dict=None, **kwargs):
        super().__init__(target)
        if target is None:
            self.target = AndroidTarget()

        if config_file:
            with open(config_file, 'r') as f:
                self._config = json.load(f)

        if config_dict:
            self._config.update(config_dict)

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f'{type(self).__name__} does not have a config for key: {key}')

        logger.debug(f'Config dictionary after processing all provided configs: {self._config}')

    @property
    def backend_library(self):
        return 'libQnnGpu.so'

    @property
    def backend_extensions_library(self):
        return "libQnnGpuNetRunExtensions.so"

    def get_required_artifacts(self, sdk_root):
        return []
