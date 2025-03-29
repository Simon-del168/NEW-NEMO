# =============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from __future__ import annotations
from pathlib import Path
from typing import List, Union, TYPE_CHECKING

import qti.aisw.core.model_level_api as mlapi
from qti.aisw.tools.core.modules.api.definitions.common import BackendType, Target, Model
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType

if TYPE_CHECKING:
    from qti.aisw.tools.core.modules.net_runner.net_runner_module import NetRunnerArgConfig, \
        InferenceConfig
    from qti.aisw.tools.core.modules.context_bin_gen.context_bin_gen_module import \
        ContextBinGenArgConfig, GenerateConfig

_backend_type_to_mlapi_backend = {
    BackendType.CPU: mlapi.CpuBackend,
    BackendType.GPU: mlapi.GpuBackend,
    BackendType.HTP: mlapi.HtpBackend,
    BackendType.HTP_MCP: mlapi.HtpMcpBackend,
    BackendType.AIC: mlapi.AicBackend
}


def get_supported_backends() -> List[str]:
    # creating a list from a dict returns only the keys
    return list(_backend_type_to_mlapi_backend)


def create_mlapi_target(target: Target) -> mlapi.Target:
    # handle hostname/port when support is added to model-level API
    if target.type == DevicePlatformType.ANDROID:
        device_id = target.identifier.serial_id if target.identifier else None
        return mlapi.AndroidTarget(device_id=device_id)
    elif target.type == DevicePlatformType.X86_64_LINUX:
        return mlapi.X86Target()
    else:
        raise ValueError(f'Unknown target type: {target.type}')


def create_mlapi_backend(config: Union['NetRunnerArgConfig', 'ContextBinGenArgConfig']) -> \
        mlapi.Backend:
    mlapi_target = create_mlapi_target(config.target) if config.target else None

    mlapi_backend_type = _backend_type_to_mlapi_backend.get(config.backend)
    if not mlapi_backend_type:
        raise ValueError(f'Unknown backend type: {config.backend}')

    if mlapi_backend_type is mlapi.CpuBackend:
        # CPU does not support backend specific configs, so skip passing config file/dict
        return mlapi_backend_type(target=mlapi_target)
    else:
        return mlapi_backend_type(target=mlapi_target,
                                  config_file=config.backend_config_file,
                                  config_dict=config.backend_config_dict)


def create_mlapi_model(model: Model) -> mlapi.Model:
    if model.qnn_model_library_path:
        model_type = mlapi.QnnModelLibrary
        model_path = Path(str(model.qnn_model_library_path))
    elif model.context_binary_path:
        model_type = mlapi.QnnContextBinary
        model_path = Path(str(model.context_binary_path))
    elif model.dlc_path:
        model_type = mlapi.DLC
        model_path = Path(str(model.dlc_path))
    else:
        raise ValueError(f'Unknown model type {model}')

    return model_type(name=model_path.stem, path=str(model_path))


def create_mlapi_run_config(config: 'InferenceConfig') -> mlapi.QNNRunConfig:
    return mlapi.QNNRunConfig(**config.dict())


def create_mlapi_generate_config(config: 'GenerateConfig') -> mlapi.QNNGenerateConfig:
    return mlapi.QNNGenerateConfig(**config.dict())
