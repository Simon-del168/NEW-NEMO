#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from typing import Union, Literal, List, Dict, Tuple, Optional, Any, Type
from pathlib import Path
from os import PathLike
import numpy as np
import logging
from pydantic.json_schema import SkipJsonSchema

from qti.aisw.tools.core.modules.api import ModuleSchemaVersion, ModuleSchema, AISWBaseModel, \
    Module, BackendType, Target, Model, expect_module_compliance
from qti.aisw.tools.core.modules.api.utils.model_level_api import get_supported_backends, \
    create_mlapi_backend, create_mlapi_model, create_mlapi_run_config
import qti.aisw.core.model_level_api as mlapi

InputListInput = Union[PathLike, str]
NamedTensorMapping = Dict[str, np.ndarray]
NetRunnerInputData = Union[InputListInput,
                           np.ndarray, List[np.ndarray],
                           NamedTensorMapping, List[NamedTensorMapping]]
InferencerCacheKey = Tuple[BackendType, str, str]


class InferenceConfig(AISWBaseModel):
    """
    Defines supported inference parameters that are implemented by backend-agnostic tools, therefore
    are applicable to all backends (subject to feature support as generally these correspond to
    optional API capabilities).
    """
    log_level: Optional[str] = None
    profiling_level: Optional[str] = None
    batch_multiplier: Optional[str] = None
    use_native_output_data: Optional[bool] = None
    use_native_input_data: Optional[bool] = None
    native_input_tensor_names: Optional[List[str]] = None
    synchronous: Optional[bool] = None


class NetRunnerArgConfig(AISWBaseModel, arbitrary_types_allowed=True):
    """
    Defines all possible arguments for an inference.

    Backend-specific parameters should be passed via config file or config dict. Any duplicate
    arguments specified in the config dict will override identical arguments in the config file.

    If target is not provided, the backend will choose a sane default based on its typical
    workflows, e.g. QNN HTP will run on Android by default, but QNN CPU will run on the host.
    """
    backend: BackendType
    backend_config_file: Optional[Union[str, PathLike]] = None
    backend_config_dict: Optional[Dict[str, Any]] = None
    target: Optional[Target] = None
    model: Model
    input_data: SkipJsonSchema[NetRunnerInputData]
    config: Optional[InferenceConfig] = None


class NetRunnerOutputConfig(AISWBaseModel, arbitrary_types_allowed=True):
    """
    Defines inference output data. Output data will be returned as a list of tensor name -> np array
    mappings, once mapping per inference. Profiling data will be returned if it was enabled in the
    InferenceConfig.
    """
    output_data: List[NamedTensorMapping]
    profiling_data: Optional[Dict[Any, Any]] = None


class NetRunnerModuleSchema(ModuleSchema):
    _BACKENDS = get_supported_backends()
    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)

    name: Literal["NetRunnerModule"] = "NetRunnerModule"
    path: Path = Path(__file__)
    arguments: NetRunnerArgConfig
    outputs: SkipJsonSchema[Optional[NetRunnerOutputConfig]] = None
    backends: List[str] = _BACKENDS


@expect_module_compliance
class NetRunner(Module):
    _SCHEMA = NetRunnerModuleSchema
    _LOGGER = logging.getLogger("NetRunnerLogger")

    def __init__(self, persistent: bool = False, logger: Any = None):
        """
        Initializes a NetRunner module instance

        Args:
            persistent (bool): Indicates that objects initialized for a particular
            (backend, target, model) tuple should persist between calls to run() so setup steps
            (e.g. pushing backend + model artifacts to a remote device) only need to be performed
            once instead of per-inference.

            logger (any): A logger instance to be used by the NetRunner module
        """
        super().__init__(logger)
        self._persistent: bool = persistent
        self._inferencer_cache: Dict[InferencerCacheKey, mlapi.Inferencer] = {}

    def properties(self) -> Dict[str, Any]:
        return self._SCHEMA.model_json_schema()

    def get_logger(self) -> Any:
        return self._logger

    def enable_debug(self, debug_level: int, **kwargs) -> Optional[bool]:
        pass

    def run(self, config: NetRunnerArgConfig) -> NetRunnerOutputConfig:
        """
        Runs inferences of a given model using the specified backend on the provided target.

        Args:
            config (NetRunnerArgConfig): Arguments of the inference where a model, backend,
            and target are specified, as well as any other miscellaneous parameters.

        Returns:
            NetRunnerOutputConfig: The output data as a list of tensor name -> np array mappings, as
            well as profiling data if profiling was enabled.

        Examples: The example below shows how to run a dlc on an X86 linux target
            >>> model = Model(dlc_path="/path/to/dlc")
            >>> input_data = np.fromfile("/path/to/numpy_raw_data").astype(np.float32)
            >>> inf_config = InferenceConfig()
            >>> run_arg_config = NetRunnerArgConfig(backend=BackendType.HTP,
            >>>                                     model=model,
            >>>                                     output_dir='./htp_output/',
            >>>                                     config=inf_config)
            >>> net_runner = NetRunner()
            >>> output_config = net_runner.run(run_arg_config)
        """
        self._logger.info(f'Running inference with config: {config}')

        # check if an Inferencer with the same (backend, target, model) tuple is in the cache
        # to avoid redundant setup done during Inferencer initialization
        target_str = config.target.model_dump_json() if config.target else ""
        inferencer_cache_key = (config.backend, target_str, config.model.model_dump_json())
        inferencer = self._inferencer_cache.get(inferencer_cache_key)
        if inferencer is None:
            mlapi_backend = create_mlapi_backend(config)
            mlapi_model = create_mlapi_model(config.model)
            inferencer = mlapi.Inferencer(mlapi_backend, mlapi_model)
            if self._persistent:
                # if persistence is enabled, store the Inferencer in a dict so it is not garbage
                # collected which would undo the initialization
                self._inferencer_cache[inferencer_cache_key] = inferencer

        run_config = create_mlapi_run_config(config.config) if config.config else None
        output_data = inferencer.run(config.input_data, run_config)

        # profiling data is a list of dicts since it can be accumulated, need to extract the first
        # element, or return None if the list is empty (i.e. if profiling is disabled)
        profiling_data = inferencer.get_profiling_data()
        profiling_data = profiling_data[0] if profiling_data else None
        return NetRunnerOutputConfig(output_data=output_data, profiling_data=profiling_data)
