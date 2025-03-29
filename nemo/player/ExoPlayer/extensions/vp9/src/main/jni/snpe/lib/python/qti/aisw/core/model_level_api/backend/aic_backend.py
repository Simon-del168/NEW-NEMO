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
from qti.aisw.core.model_level_api.target.x86 import X86Target

logger = logging.getLogger(__name__)


class AicBackend(Backend):
    graph_names = BackendConfig()
    compiler_compilation_target = BackendConfig()
    compiler_hardware_version = BackendConfig()
    compiler_num_of_cores = BackendConfig()
    compiler_convert_to_FP16 = BackendConfig()
    compiler_do_host_preproc = BackendConfig()
    compiler_stat_level = BackendConfig()
    compiler_printDDRStats = BackendConfig()
    compiler_printPerfMetrics = BackendConfig()
    compiler_perfWarnings = BackendConfig()
    compiler_compilationOutputDir = BackendConfig()
    compiler_enableDebug = BackendConfig()
    compiler_buffer_dealloc_delay = BackendConfig()
    compiler_genCRC = BackendConfig()
    compiler_stats_batch_size = BackendConfig()
    compiler_crc_stride = BackendConfig()
    compiler_enable_depth_first = BackendConfig()
    compiler_overlap_split_factor = BackendConfig()
    compiler_depth_first_mem = BackendConfig()
    compiler_VTCM_working_set_limit_ratio = BackendConfig()
    compiler_size_split_granularity = BackendConfig()
    compiler_compileThreads = BackendConfig()
    compiler_userDMAProducerDMAEnabled = BackendConfig()
    compiler_do_DDR_to_multicast = BackendConfig()
    compiler_combine_inputs = BackendConfig()
    compiler_combine_outputs = BackendConfig()
    compiler_directApi = BackendConfig()
    compiler_force_VTCM_spill = BackendConfig()
    compiler_PMU_recipe_opt = BackendConfig()
    compiler_PMU_events = BackendConfig()
    compiler_cluster_sizes = BackendConfig()
    compiler_max_out_channel_split = BackendConfig()
    runtime_device_id = BackendConfig()
    runtime_num_activations = BackendConfig()
    runtime_submit_timeout = BackendConfig()
    runtime_submit_num_retries = BackendConfig()
    runtime_threads_per_queue = BackendConfig()
    runtime_process_lock = BackendConfig()

    def __init__(self, target=None, config_file=None, config_dict=None, **kwargs):
        super().__init__(target)
        if target is None:
            self.target = X86Target()

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

        logger.debug(f"Config dictionary after processing all provided configs: {self._config}")

    @property
    def backend_library(self):
        return "libQnnAic.so"

    @property
    def backend_extensions_library(self):
        return "libQnnAicNetRunExtensions.so"

    def get_required_artifacts(self, sdk_root):
        return []
