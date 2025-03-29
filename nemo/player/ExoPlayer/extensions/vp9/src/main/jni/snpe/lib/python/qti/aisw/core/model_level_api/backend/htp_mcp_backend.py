# =============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
from pathlib import Path
from shutil import copyfile

from qti.aisw.core.model_level_api.backend.backend import Backend
from qti.aisw.core.model_level_api.target.x86 import X86Target


class HtpMcpBackend(Backend):
    def __init__(self, target=None, config_file=None, config_dict=None):
        super().__init__(target)
        if target is None:
            self.target = X86Target()

        if config_file:
            with open(config_file, 'r') as f:
                self._config = json.load(f)

        if config_dict:
            self._config.update(config_dict)

    @property
    def backend_library(self):
        return 'libQnnHtpMcp.so'

    @property
    def backend_extensions_library(self):
        return 'libQnnHtpMcpNetRunExtensions.so'

    def get_required_artifacts(self, sdk_path):
        return []

    def before_generate_hook(self, temp_directory, sdk_path):
        mcp_elf_path = Path(sdk_path, 'lib', 'hexagon-v68', 'unsigned', 'libQnnHtpMcpV68.elf')
        if not mcp_elf_path.exists():
            raise FileNotFoundError(f"Could not find HTP MCP elf file {mcp_elf_path}")

        copyfile(mcp_elf_path, Path(temp_directory, 'network.elf'))