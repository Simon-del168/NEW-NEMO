# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
from pathlib import Path

from qti.aisw.tools.core.modules.converter.utils import infer_framework

from .enums import BackendType
from .params import CompilationParams
from .compiler import Compiler
from .qairt_model import QairtModel
from .logger import setup_qairt_logger

qairt_logger = logging.getLogger('qairt_api')
setup_qairt_logger()


class Model():
    """Model class"""

    def __init__(self, model_path: str | Path):
        """Constructor of the Model class.

        Args:
            model_path (str | Path): Path to the source model file
        """

        self.model_path = Path(model_path).absolute()
        model_framework = infer_framework(str(self.model_path))
        qairt_logger.info(f'Initialized with {model_framework} model: {self.model_path}')


def compile(model: Model, compilation_params: CompilationParams = None) -> QairtModel:
    """
    Compiles the given model using the specified backend and parameters.

    Args:
        model (Model): The Model object generated from model loading.
        compilation_params (CompilationParams, optional): Parameters for model compilation. Defaults to None.

    Returns:
        QairtModel: The QairtModel object with path to DLC or binary file from model compilation.
    """
    compiler = Compiler()
    qairt_model = compiler.compile(model_file=model.model_path,
                                   compilation_params=compilation_params)
    return qairt_model


def import_model(compiled_model: str | Path, backend: BackendType = None) -> QairtModel:
    """
    Import a pre-compiled model.

    Args:
        compiled_model (str | Path): Path to the compiled model file (.dlc or .bin)
        backend (BackendType, optional): Backend used for compilation. Defaults to None.

    Returns:
        QairtModel: The QairtModel callable object containing information about the imported model

    Raises:
        FileNotFoundError: If the given path does not exist
    """

    compiled_model = Path(compiled_model)

    if not compiled_model.exists():
        raise FileNotFoundError(f'Given path {compiled_model} does not exist.')

    suffix = compiled_model.suffix
    if suffix not in {'.bin', '.dlc'}:
        raise ValueError(
            f"Expected a .bin or .dlc file. Received a {suffix} file '{compiled_model}'")

    qairt_logger.info(f'Importing compiled model: {compiled_model}')

    if suffix == '.dlc':
        return QairtModel(dlc=compiled_model, backend=backend)
    if suffix == '.bin':
        if not backend:
            raise ValueError('Backend must be provided along with binary file.')
        return QairtModel(binary=compiled_model, backend=backend)
