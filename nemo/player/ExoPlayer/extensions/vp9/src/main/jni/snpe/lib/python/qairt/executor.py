# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import numpy as np
from typing import NamedTuple
from pathlib import Path

from .enums import BackendType, DevicePlatformType
from .params import ExecutionParams, Target
from .exceptions import ExecutionFailure

qairt_logger = logging.getLogger('qairt_api')

from qti.aisw.tools.core.modules.api.definitions.common import Model, Target as module_target
from qti.aisw.tools.core.modules.net_runner import net_runner_module
from qti.aisw.tools.core.utilities.devices.api.device_definitions import RemoteDeviceIdentifier


class InferenceResults(NamedTuple):
    output_data: list[dict[str, np.ndarray]]
    profiling_data: list = None


class Executor():

    def __init__(self, compiled_model: str, backend: BackendType, target: Target,
                 execution_params: ExecutionParams | None = None) -> None:
        """
        Executor constructor

        Args:
            compiled_model (str): The path to DLC or binary file to execute.
            backend (BackendType): The backend to use for execution (e.g., CPU, GPU, etc.).
            target (Target): The target platform for execution (e.g.,android).
            execution_params (ExecutionParams, optional): Additional execution parameters.
                                                          Defaults to None.

        Raises:
            TypeError: When given compiled_model file is neither .dlc nor .bin
        """
        self.params = execution_params
        self.backend_extensions = execution_params.backend_extensions if execution_params else None

        if backend is BackendType.GPU and target.type is not DevicePlatformType.ANDROID:
            raise Exception(
                f'GPU backend is only supported on Android targets. Given target = {self.target}')

        self.backend = backend
        self.target = self._get_module_target(target)

        file_type = Path(compiled_model).suffix
        if file_type == ".bin":
            self.compiled_model = Model(context_binary_path=str(compiled_model))
        elif file_type == ".dlc":
            self.compiled_model = Model(dlc_path=str(compiled_model))
        else:
            raise ValueError(f"Expected a .bin or .dlc file. Received:'{compiled_model}'")

    def run(self, inputs: str | np.ndarray | dict[str, np.ndarray]) -> InferenceResults:
        """
        Perform model inference.

        :param inputs: Inputs to be used for inference. This can be a path to input list file or a
                       numpy array or dictionary of node name and numpy array.

        :return: The inference outputs are formatted as a dictionary with the output node name as key
                 and the numpy array output as value.
                 A list of these dictionaries is formed and returned.

        Args:
            inputs (str | numpy.ndarray | dict[str, numpy.ndarray]): Input data for the model. Can be a
                    string (file path), a numpy array or a dictionary of numpy arrays with named inputs.

        Returns:
            InferenceResults: An object containing the inference outputs and any profiling information.
                              The inference outputs are formatted as list of dictionaries, where each
                              dictionary represents the output data produced by the model. The keys in
                              each dictionary correspond to specific output names, and the associated
                              values are numpy arrays containing the output data.
                              Note: The actual output keys and values will depend on the model.
        """

        qairt_logger.debug('Preparing inference config of net-runner module')
        if self.params:
            inference_config = net_runner_module.InferenceConfig(
                **self.params.model_dump(exclude_unset=True, exclude=['backend_extensions']))
        else:
            inference_config = net_runner_module.InferenceConfig()

        qairt_logger.debug('Preparing arg config of net-runner module')
        net_runner_arg_config = net_runner_module.NetRunnerArgConfig(
            backend=self.backend, backend_config_dict=self.backend_extensions,
            model=self.compiled_model, input_data=inputs, config=inference_config,
            target=self.target)

        qairt_logger.debug('Initializing net-runner module')
        net_runner = net_runner_module.NetRunner(persistent=False)

        qairt_logger.info('Running inference for {} backend on {} target'.format(
            self.backend, self.target.type.value))
        try:
            output_config = net_runner.run(net_runner_arg_config)
            results = InferenceResults(output_config.output_data, output_config.profiling_data)
            qairt_logger.info('Completed inference')

        except Exception as exception:
            raise ExecutionFailure(f"Failed to execute the model!") from exception

        return results

    # TODO: Deprecate once switched over to Module defined Target class
    def _get_module_target(self, target: Target) -> module_target:
        """Get module variant of Target class

        Args:
            target (Target): Target object

        Returns:
            module_target: Module defined Target object
        """
        net_run_target = module_target(type=DevicePlatformType(target.type))
        if target.identifier:
            net_run_target.identifier = RemoteDeviceIdentifier(serial_id=target.identifier)
        return net_run_target
