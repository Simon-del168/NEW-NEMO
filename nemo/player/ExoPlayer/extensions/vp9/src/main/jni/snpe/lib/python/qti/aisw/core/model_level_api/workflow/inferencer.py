#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================
from qti.aisw.core.model_level_api.workflow.workflow import Workflow, WorkflowMode


class Inferencer(Workflow):
    def __init__(self, backend, model, executor=None, sdk_path=None):
        super().__init__(backend, model, executor, sdk_path)
        workflow_mode = WorkflowMode.INFERENCE
        self._backend.workflow_mode = workflow_mode

        if self._executor is None:
            target_default_executor_cls = self._backend.target.get_default_executor_cls()
            self._executor = target_default_executor_cls()

        self._executor.setup(workflow_mode,
                             self._backend,
                             self._model,
                             self._sdk_path)

    def run(self, input_data, config=None):
        output_data, profiling_data = self._executor.run_inference(config,
                                                                   self._backend,
                                                                   self._model,
                                                                   self._sdk_path,
                                                                   input_data)
        if profiling_data and len(profiling_data) > 0:
            self._profiling_data.append(profiling_data)
        return output_data
