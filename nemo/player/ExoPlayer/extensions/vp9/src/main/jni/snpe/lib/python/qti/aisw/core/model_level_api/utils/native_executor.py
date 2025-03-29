# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import numpy as np
from pathlib import Path
from contextlib import contextmanager
import os

from . import libPythonNetRun as py_net_run
from qti.aisw.core.model_level_api.utils.subprocess_executor import \
    get_name_input_pairs_from_input_list, get_np_dtype_from_qnn_dtype

logger = logging.getLogger(__name__)


def create_profile_level_argument(config):
    profile_level_str_to_enum = {
        'off': py_net_run.ProfilingLevel.OFF,
        'basic': py_net_run.ProfilingLevel.BASIC,
        'detailed': py_net_run.ProfilingLevel.DETAILED,
        'backend': py_net_run.ProfilingLevel.BACKEND_CUSTOM
    }

    if config and config.profiling_level:
        if config.profiling_level not in profile_level_str_to_enum:
            raise KeyError(f'profiling level {config.profiling_level} not recognized')
        return profile_level_str_to_enum[config.profiling_level]
    return py_net_run.ProfilingLevel.OFF


def create_log_level_argument(config):
    log_level_str_to_enum = {
        'error': py_net_run.QnnLogLevel.ERROR,
        'warn': py_net_run.QnnLogLevel.WARN,
        'info': py_net_run.QnnLogLevel.INFO,
        'verbose': py_net_run.QnnLogLevel.VERBOSE,
        'debug': py_net_run.QnnLogLevel.DEBUG
    }

    if config and config.log_level:
        if config.log_level not in log_level_str_to_enum:
            raise KeyError(f'log level {config.log_level} not recognized')
        return log_level_str_to_enum[config.log_level]
    return py_net_run.QnnLogLevel.ERROR


def create_op_package_argument(backend):
    op_packages = backend.get_registered_op_packages()
    if not op_packages:
        return ''

    op_package_strings = []
    for pkg_path, pkg_provider, pkg_target in op_packages:
        target_str = f':{pkg_target}' if pkg_target else ''
        op_package_str = f'{pkg_path}:{pkg_provider}{target_str}'
        op_package_strings.append(op_package_str)

    return ','.join(op_package_strings)


def create_backend_extension_argument(backend, temp_directory, sdk_root):
    backend_extension_lib_name = backend.backend_extensions_library
    backend_extension_lib_path = ''
    if backend_extension_lib_name:
        backend_extension_lib_path = Path(sdk_root, 'lib', backend.target.target_name,
                                           backend_extension_lib_name)

    backend_config_str = backend.get_config_json()
    if not backend_config_str:
        return [f'{backend_extension_lib_path}', '']

    backend_json_path = Path(temp_directory, 'backend_json.txt')
    with backend_json_path.open('w') as file:
        file.write(backend_config_str)

    return [f'{backend_extension_lib_path}', f'{backend_json_path}']


def create_batch_multiplier_argument(config):
    return config.batch_multiplier if config and config.batch_multiplier else 1


def create_output_datatype_argument(config):
    return 'native_only' if config and config.use_native_output_data else 'float_only'


def input_list_to_in_memory_input(input_list_path,
                                  native_inputs,
                                  native_input_tensor_names,
                                  input_name_dtype_pairs):
    input_data_list = []

    name_input_pairs = get_name_input_pairs_from_input_list(input_list_path)
    input_name_dtype_dict = None
    if input_name_dtype_pairs:
        input_name_dtype_dict = dict(input_name_dtype_pairs)

    for inference_input in name_input_pairs:
        inference_input_dict = {}
        for input_idx, (input_name, input_path) in enumerate(inference_input):
            np_dtype = np.float32
            if not input_name:
                input_name = f'placeholder_input_{input_idx}'
                if native_inputs:
                    # if input name is not present in the input list and all inputs are requested
                    # as native, look up the datatype based on index
                    input_dtype = input_name_dtype_pairs[input_idx][1]
                    np_dtype = get_np_dtype_from_qnn_dtype(input_dtype)
                elif native_input_tensor_names:
                    # if only certain inputs are requested, look up the input name by index and see
                    # if it is one of the requested inputs, if so look up the datatype by index
                    input_name = input_name_dtype_pairs[input_idx][0]
                    if input_name in native_input_tensor_names:
                        input_dtype = input_name_dtype_pairs[input_idx][1]
                        np_dtype = get_np_dtype_from_qnn_dtype(input_dtype)
            else:
                # if input name is present, look up the datatype based on name
                if native_inputs or (native_input_tensor_names and
                                     input_name in native_input_tensor_names):
                    input_dtype = input_name_dtype_dict[input_name]
                    np_dtype = get_np_dtype_from_qnn_dtype(input_dtype)

            inference_input_dict[input_name] = np.fromfile(input_path, dtype=np_dtype)
        input_data_list.append(inference_input_dict)

    return input_data_list

# a utility to temporarily change into a working directory, reverting back to the previous working
# directory when the object goes out of scope
@contextmanager
def temporaryDirectoryChange(temp_dir):
    old_cwd = os.getcwd()
    os.chdir(Path(temp_dir).resolve())

    try:
        yield
    finally:
        os.chdir(old_cwd)