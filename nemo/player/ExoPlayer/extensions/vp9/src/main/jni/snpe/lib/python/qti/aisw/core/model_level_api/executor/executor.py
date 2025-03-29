#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from abc import ABC, abstractmethod


class Executor(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def setup(self, workflow_mode, backend, model, sdk_root):
        pass

    @abstractmethod
    def run_inference(self, config, backend, model, sdk_root, input_data):
        pass

    @abstractmethod
    def generate_context_binary(self, config, backend, model, sdk_root, output_path,
                                output_filename):
        pass
