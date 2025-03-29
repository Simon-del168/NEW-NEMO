# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import shutil
import numpy as np
from typing import NamedTuple
from pathlib import Path

from .enums import BackendType
from .params import ExecutionParams, CompilationParams, Target
from .executor import Executor

qairt_logger = logging.getLogger('qairt_api')


class InferenceResults(NamedTuple):
    """
    Class containing results of inference

    Args:
        outputs (list[dict[str, np.ndarray]]): Outputs of the inference
        profiling_data (list): Profiling data collected during inference
    """
    outputs: list[dict[str, np.ndarray]]
    profiling_data: list = None


class QairtModel():
    """Qairt Model class for the compiled model"""

    def __init__(self, dlc: str | Path = None, binary: str | Path = None,
                 backend: BackendType = None) -> None:
        """
        Initializes the QairtModel instance.

        Args:
            dlc (str | Path): Path to the DLC file.
            binary (str | Path): Path to the binary file generated from offline preparation.
            backend (BackendType, optional): Backend used during model compilation. Defaults to None.
                                             Mandatory if binary is provided.

        Raises:
            Exception: If neither dlc nor binary are not provided.
            FileNotFoundError: If the given file does not exist.
            ValueError: If the backend is not provided along with binary file.
        """

        if not (dlc or binary):
            raise Exception("At least one of 'dlc' or 'binary' arguments must be provided.")

        self.dlc = self.validate_file(dlc, '.dlc') if dlc else None
        self.binary = self.validate_file(binary, '.bin') if binary else None

        if self.binary and not backend:
            raise ValueError('Backend must be provided along with binary file.')

        self.backend = backend

    def __call__(self, inputs: str | np.ndarray | dict[str, np.ndarray], target: Target,
                 execution_params: ExecutionParams = None,
                 backend: BackendType = None) -> InferenceResults:
        """
        Callable to perform model inference.

        Args:
            inputs (str | np.ndarray | dict[str, np.ndarray]): Inputs to be used for inference.
                                                               This can be a path to input list file
                                                               or a numpy array or dictionary of
                                                               node name and numpy array.
            target (Target): Target platform for execution
            execution_params (ExecutionParams, optional): Parameters to be used during execution.
                                                          Defaults to None.
            backend (BackendType, optional): Desired backend for inference. Defaults to None.
                                             Backend is mandatory if not set during compilation or
                                             when loading compiled model from disk.

        Returns:
            InferenceResults: Inference outputs and profiling data, if enabled.
                              The inference outputs are formatted as a dictionary
                              with the output node name as key and the numpy array
                              output as value.

        Raises:
            Exception: If the backend attribute was not set during initialization
                       and is not provided during inference.
            RuntimeError: If instance has binary only and different backend is given.
        """

        compiled_model = self.binary or self.dlc

        if self.backend:
            # Check if inference backend is given and matches compilation backend
            if backend and backend is not self.backend:
                if self.binary and not self.dlc:
                    raise RuntimeError(f'Compiled model "{compiled_model}" is compatible with '
                                       f'{self.backend}. Cannot change backend during inference.')

                # Warn if user have given a different backend for inference
                qairt_logger.warning(f'Model was compiled with {self.backend} backend. Running '
                                     f'with {backend} backend might fail or be suboptimal.')
                compiled_model = self.dlc
        elif not backend:
            raise Exception("No backend info available. Specify backend for inference.")

        backend = backend or self.backend

        executor = Executor(compiled_model=compiled_model, backend=backend, target=target,
                            execution_params=execution_params)
        results = executor.run(inputs=inputs)
        return results

    def export(self, output_directory: str | Path) -> None:
        """
        Exports the compiled model file(s) to the specified directory.

        Args:
            output_directory (str | Path): Directory where the compiled model
                                           file(s) should be saved.
        """
        output_directory = Path(output_directory)
        if not output_directory.exists():
            qairt_logger.warning(
                f'Given path {output_directory} does not exist. Creating the given path.')
            output_directory.mkdir()

        if self.dlc:
            shutil.copy(self.dlc, output_directory / self.dlc.name)
        if self.binary:
            shutil.copy(self.binary, output_directory / self.binary.name)

        return

    def validate_file(self, filepath: str | Path, suffix: str) -> Path:
        """
        Validates that the given file exists and has the expected suffix.

        Args:
            filepath (str | Path): Path to the file to be validated.
            suffix (str): Expected suffix of the file.

        Returns:
            Path: Path object representing the validated file.

        Raises:
            FileNotFoundError: If the given file does not exist.
            ValueError: If the given file does not have the expected suffix.
        """
        given_file = Path(filepath).resolve()
        if not given_file.exists():
            raise FileNotFoundError(f"Given path '{given_file}' does not exist.")

        if given_file.suffix != suffix:
            raise ValueError(f"Expected a {suffix} file. Received a {given_file.suffix} file")

        return given_file
