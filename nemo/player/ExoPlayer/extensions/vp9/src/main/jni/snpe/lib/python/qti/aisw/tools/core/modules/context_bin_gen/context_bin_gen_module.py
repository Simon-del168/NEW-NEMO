# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import Union, Literal, List, Dict, Tuple, Optional, Any
from pathlib import Path
from os import PathLike
import logging
from pydantic import model_validator

from qti.aisw.tools.core.modules.api import ModuleSchemaVersion, ModuleSchema, AISWBaseModel, \
                                            BackendType, Target, Model, Module, \
                                            expect_module_compliance
from qti.aisw.tools.core.modules.api.utils.model_level_api import get_supported_backends, \
    create_mlapi_backend, create_mlapi_model, create_mlapi_generate_config
import qti.aisw.core.model_level_api as mlapi


ContextBinaryGeneratorCacheKey = Tuple[BackendType, str, str]


class GenerateConfig(AISWBaseModel):
    """
    Defines supported context binary generation parameters that are implemented by backend-agnostic
    tools, therefore are applicable to all backends.
    """
    log_level: Optional[str] = None


class ContextBinGenArgConfig(AISWBaseModel):
    """
    Defines all possible arguments for context binary generation.

    Backend-specific parameters should be passed via config file or config dict. Any duplicate
    arguments specified in the config dict will override identical arguments in the config file.

    If target is not provided, the backend will choose a sane default based on its typical
    workflows, e.g. QNN HTP will generate on the host by default, but QNN GPU will generate on
    Android since offline preparation is not supported.
    """
    backend: BackendType
    backend_config_file: Optional[Union[str, PathLike]] = None
    backend_config_dict: Optional[Dict[str, Any]] = None
    target: Optional[Target] = None
    model: Model
    output_dir: Optional[Union[str, PathLike]] = "./output/"
    output_filename: Optional[str] = None
    config: Optional[GenerateConfig] = None

    @model_validator(mode="after")
    def validate_model_type(self) -> 'ContextBinGenArgConfig':
        if self.model.context_binary_path:
            raise ValueError('Cannot generate a context binary from an existing context binary')
        return self


class ContextBinGenOutputConfig(AISWBaseModel):
    """
    Defines context binary generation output format
    """
    context_binary: Model

    @model_validator(mode="after")
    def validate_model_type(self) -> 'ContextBinGenOutputConfig':
        if not self.context_binary.context_binary_path:
            raise ValueError('Context binary generation output must be a context binary')
        return self


class ContextBinGenModuleSchema(ModuleSchema):
    _BACKENDS = get_supported_backends()
    _VERSION = ModuleSchemaVersion(major=0, minor=2, patch=0)

    name: Literal["ContextBinGenModule"] = "ContextBinGenModule"
    path: Path = Path(__file__)
    arguments: ContextBinGenArgConfig
    outputs: Optional[ContextBinGenOutputConfig] = None
    backends: List[str] = _BACKENDS


@expect_module_compliance
class ContextBinGen(Module):
    _SCHEMA = ContextBinGenModuleSchema
    _LOGGER = logging.getLogger("ContextBinGenLogger")

    def __init__(self, persistent: bool = False, logger: Any = None):
        """
        Initializes a ContextBinGen module instance

        Args:
            persistent (bool): Indicates that objects initialized for a particular
            (backend, target, model) tuple should persist between calls to generate() so setup steps
            (e.g. pushing backend + model artifacts to a remote device) only need to be performed
            once instead of each time a binary is generated.

            logger (any): A logger instance to be used by the ContextBinGen module
        """
        super().__init__(logger)
        self._persistent: bool = persistent
        self._context_binary_generator_cache: Dict[ContextBinaryGeneratorCacheKey,
                                                   mlapi.ContextBinaryGenerator] = {}

    def properties(self) -> Dict[str, Any]:
        return self._SCHEMA.model_json_schema()

    def get_logger(self) -> Any:
        return self._logger

    def enable_debug(self, debug_level: int, **kwargs) -> Optional[bool]:
        pass

    def generate(self, config: ContextBinGenArgConfig) -> ContextBinGenOutputConfig:
        """
        Generates a context binary from a given model using the specified backend on the provided
        target.

        Args:
            config (ContextBinGenArgConfig): Arguments of context binary generation where a model,
            backend, and target are specified, as well as any other miscellaneous parameters.

        Returns:
            ContextBinGenOutputConfig: A structure containing a path to the generated context binary
            as an instance of a Model with the context_binary_path populated.

        Examples: The example below generates a context binary for HTP on x86 linux
            >>> model = Model(dlc_path="/path/to/dlc")
            >>> generate_config = GenerateConfig()
            >>> context_bin_gen_arg_config = ContextBinGenArgConfig(backend=BackendType.HTP,
            >>>                                                     model=model,
            >>>                                                     output_dir='./htp_output/',
            >>>                                                     config=generate_config)
            >>> context_bin_gen = ContextBinGen()
            >>> output_config = context_bin_gen.generate(context_bin_gen_arg_config)
        """
        self._logger.info(f'Generating context with config: {config}')

        # check if a ContextBinaryGenerator with the same (backend, target, model) tuple is in the
        # cache to avoid redundant setup done during ContextBinaryGenerator initialization
        target_str = config.target.model_dump_json() if config.target else ""
        context_binary_generator_cache_key = (config.backend,
                                              target_str,
                                              config.model.model_dump_json())
        context_bin_gen = \
            self._context_binary_generator_cache.get(context_binary_generator_cache_key)
        if context_bin_gen is None:
            mlapi_backend = create_mlapi_backend(config)
            mlapi_model = create_mlapi_model(config.model)
            context_bin_gen = mlapi.ContextBinaryGenerator(mlapi_backend, mlapi_model)
            if self._persistent:
                # if persistence enabled, store the ContextBinaryGenerator in a dict so it is not
                # garbage collected which would undo the initialization
                self._context_binary_generator_cache[context_binary_generator_cache_key] = \
                    context_bin_gen

        generate_config = create_mlapi_generate_config(config.config) if config.config else None
        output_binary = context_bin_gen.generate(output_path=config.output_dir,
                                                 output_filename=config.output_filename,
                                                 config=generate_config)

        context_binary = Model(context_binary_path=output_binary.binary_path)
        return ContextBinGenOutputConfig(context_binary=context_binary)
