#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from qti.aisw.core.model_level_api.workflow.workflow import Workflow, WorkflowMode


class ContextBinaryGenerator(Workflow):
    def __init__(self, backend, model, executor=None, sdk_path=None):
        super().__init__(backend, model, executor, sdk_path)
        workflow_mode = WorkflowMode.CONTEXT_BINARY_GENERATION
        self._backend.workflow_mode = workflow_mode

        if self._executor is None:
            target_default_executor_cls = self._backend.target.get_default_executor_cls()
            self._executor = target_default_executor_cls()

        self._executor.setup(workflow_mode,
                             self._backend,
                             self._model,
                             self._sdk_path)


    def generate(self, output_path='./output/', output_filename=None, config=None):
        return self._executor.generate_context_binary(config,
                                                      self._backend,
                                                      self._model,
                                                      self._sdk_path,
                                                      output_path,
                                                      output_filename)
