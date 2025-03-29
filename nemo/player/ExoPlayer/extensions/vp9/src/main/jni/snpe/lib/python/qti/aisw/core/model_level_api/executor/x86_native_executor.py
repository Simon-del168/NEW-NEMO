#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from tempfile import TemporaryDirectory
from pathlib import Path
import logging
import numpy as np
import os

from qti.aisw.core.model_level_api.executor.executor import Executor
from qti.aisw.core.model_level_api.model.context_binary import QnnContextBinary
from qti.aisw.core.model_level_api.utils.native_executor import create_log_level_argument, \
    create_op_package_argument, create_profile_level_argument, create_batch_multiplier_argument, \
    create_backend_extension_argument, create_output_datatype_argument, \
    input_list_to_in_memory_input, py_net_run, temporaryDirectoryChange
from qti.aisw.core.model_level_api.utils.qnn_profiling import default_profiling_log_name, \
    output_dir_to_profiling_data
from qti.aisw.core.model_level_api.utils.exceptions import InferenceError, \
    ContextBinaryGenerationError, NetRunErrorCode

logger = logging.getLogger(__name__)


def _check_pnr_error(err, msg, exception_type, netrun_error_enum):
    if err != py_net_run.StatusCode.SUCCESS:
        raise exception_type(netrun_error_enum, msg)


class X86NativeExecutor(Executor):
    _model_lib_key = "model_lib_key"
    _backend_lib_key = "backend_lib_key"
    _backend_key = "backend_key"
    _logger_key = "logger_key"
    _device_key = "device_key"
    _context_key = "context_key"

    def __init__(self):
        super().__init__()

    def run_inference(self, config, backend, model, sdk_root, input_data):
        temp_directory = TemporaryDirectory()
        logger.debug(f'created temp directory: {temp_directory.name}')

        backend.before_run_hook(temp_directory.name, sdk_root)

        log_level_arg = create_log_level_argument(config)
        op_package_arg = create_op_package_argument(backend)
        profile_level_arg = create_profile_level_argument(config)

        # if the model is a context binary, its path must be provided during PythonBackendManager
        # construction
        try:
            binary_path_arg = model.binary_path
        except AttributeError:
            binary_path_arg = ''

        if not binary_path_arg:
            try:
                dlc_path = str(Path(model.dlc_path).resolve())
            except AttributeError:
                dlc_path = ''

            if dlc_path:
                model_path = Path(sdk_root, 'lib', 'x86_64-linux-clang', 'libQnnModelDlc.so')
            else:
                model_path = Path(model.model_path)
            model_path = model_path.resolve()
            if not model_path.is_file():
                raise FileNotFoundError(f"Could not locate {model_path}")

        # if input data is an input list (Path or str), resolve the path before entering the temp
        # directory in case a relative path was provided
        if isinstance(input_data, Path) or isinstance(input_data, str):
            input_data = Path(input_data).resolve()

        with temporaryDirectoryChange(temp_directory.name):
            pbm = py_net_run.PythonBackendManager(logLevel=log_level_arg,
                                                  opPackagePaths=op_package_arg,
                                                  cachedBinaryPath=binary_path_arg,
                                                  profilingLevel=profile_level_arg)

            if not binary_path_arg:
                err = pbm.loadModelLib(str(model_path), self._model_lib_key)
                _check_pnr_error(err,
                                 f'Failed to load model library: {model_path}',
                                 InferenceError,
                                 NetRunErrorCode.INITIALIZE)

            err = pbm.loadBackendLib(self._backend_lib_key, backend.backend_library)
            _check_pnr_error(err,
                             f'Failed to load backend library: {backend.backend_library}',
                             InferenceError,
                             NetRunErrorCode.INITIALIZE)

            err = pbm.createLogHandle(self._backend_lib_key, self._logger_key, log_level_arg)
            _check_pnr_error(err,
                             'Failed to initialize logging in the backend',
                             InferenceError,
                             NetRunErrorCode.INITIALIZE)

            [extension_lib_path, json_path] = create_backend_extension_argument(backend,
                                                                                temp_directory.name,
                                                                                sdk_root)
            if extension_lib_path:
                err = pbm.initializeBackendExtension(self._backend_lib_key,
                                                     extension_lib_path,
                                                     json_path,
                                                     py_net_run.AppType.QNN_APP_NETRUN)
                _check_pnr_error(err,
                                 'Failed to initialize backend extensions',
                                 InferenceError,
                                 NetRunErrorCode.INITIALIZE)

            if profile_level_arg != py_net_run.ProfilingLevel.OFF:
                err = pbm.initializeProfileLogger(self._backend_lib_key,
                                                  temp_directory.name,
                                                  default_profiling_log_name)
                _check_pnr_error(err,
                                 'Failed to initialize the profile logger',
                                 InferenceError,
                                 NetRunErrorCode.INITIALIZE)

            err = pbm.createBackendHandle(self._backend_lib_key, self._logger_key, self._backend_key)
            _check_pnr_error(err,
                             'Failed to create a backend handle',
                             InferenceError,
                             NetRunErrorCode.CREATE_BACKEND)

            err = pbm.createDeviceHandle(self._backend_lib_key, self._logger_key, self._device_key)
            _check_pnr_error(err,
                             'Failed to create a device handle',
                             InferenceError,
                             NetRunErrorCode.CREATE_DEVICE)

            err = pbm.registerOpPackage(self._backend_lib_key, self._backend_key)
            _check_pnr_error(err,
                             'Failed to register op packages',
                             InferenceError,
                             NetRunErrorCode.REGISTER_OPPACKAGE)

            if binary_path_arg:
                err = pbm.createContextFromBinaryFile(self._backend_lib_key,
                                                      self._backend_key,
                                                      self._device_key,
                                                      self._context_key)
                _check_pnr_error(err,
                                 f'Failed to create a context from the provided binary file: '
                                 f'{binary_path_arg}',
                                 InferenceError,
                                 NetRunErrorCode.CREATE_FROM_BINARY)
            else:
                err = pbm.createContext(self._backend_lib_key,
                                        self._backend_key,
                                        self._device_key,
                                        self._context_key)
                _check_pnr_error(err,
                                 'Failed to create a context',
                                 InferenceError,
                                 NetRunErrorCode.CREATE_CONTEXT)

                if dlc_path:
                    err = pbm.composeGraphsFromDlc(self._backend_lib_key,
                                                   self._backend_key,
                                                   self._context_key,
                                                   self._model_lib_key,
                                                   dlc_path)
                else:
                    err = pbm.composeGraphs(self._backend_lib_key,
                                            self._backend_key,
                                            self._context_key,
                                            self._model_lib_key)
                _check_pnr_error(err,
                                 'Failed to compose graphs',
                                 InferenceError,
                                 NetRunErrorCode.COMPOSE_GRAPHS)

                err = pbm.finalizeGraphs(self._backend_lib_key,
                                         self._backend_key,
                                         self._context_key,
                                         profilingLevel=profile_level_arg)
                _check_pnr_error(err,
                                 'Failed to finalize graphs',
                                 InferenceError,
                                 NetRunErrorCode.FINALIZE_GRAPHS)

            # if input_data is an input list, read it into memory so it can be passed via pybind
            if isinstance(input_data, Path):
                native_inputs = None
                native_input_tensor_names = None
                graph_input_name_dtype_pairs = None
                if config:
                    native_inputs = config.use_native_input_data
                    native_input_tensor_names = config.native_input_tensor_names
                    if native_inputs or native_input_tensor_names:
                        graph_input_name_dtype_pairs = pbm.getGraphInputNameDtypePairs()
                input_data = input_list_to_in_memory_input(input_data,
                                                           native_inputs,
                                                           native_input_tensor_names,
                                                           graph_input_name_dtype_pairs)

            # the pybind layer supports 2 forms of input data:
            # - list[list[np.ndarray]], where the length of the inner list must be the # of inputs to
            #   the network, and the length of the outer list is the # of inferences to run
            # - list[dict[str, np.ndarray]], where the inner list provides name -> input mappings for
            #   all network inputs, and the outer list is the # of inferences to run

            # if input_data is a single np array, wrap it in in a 2d list as it can be assumed the user
            # is running one (possibly batched) inference of a single input network
            elif isinstance(input_data, np.ndarray):
                input_data = [[input_data]]

            # if input data is a list of numpy arrays, assume the user is running multiple inferences of
            # a single input network and wrap each np array in its own list
            elif isinstance(input_data, list) and isinstance(input_data[0], np.ndarray):
                input_data = [[input_arr] for input_arr in input_data]

            # if input data is a dict of name -> np.array mappings, the user is running a single
            # inference of a multi-input network so wrap the dict in a list
            elif isinstance(input_data, dict):
                input_data = [input_data]

            batch_multiplier_arg = create_batch_multiplier_argument(config)
            output_datatype_arg = create_output_datatype_argument(config)
            synchronous_arg = config and config.synchronous
            try:
                output_data = pbm.executeGraphs(self._backend_lib_key,
                                                self._backend_key,
                                                self._context_key,
                                                '',
                                                input_data,
                                                profilingLevel=profile_level_arg,
                                                batchMultiplier=batch_multiplier_arg,
                                                outputDataType=output_datatype_arg,
                                                synchronous=synchronous_arg)
            except Exception as e:
                error_str = f'({NetRunErrorCode.EXECUTE_GRAPHS}) Exception occurred during graph ' \
                            f'execution: {e}'
                logger.error(error_str)
                raise InferenceError(NetRunErrorCode.EXECUTE_GRAPHS, error_str)

            err = pbm.freeGraphsInfo(self._backend_lib_key, self._backend_key, self._context_key)
            _check_pnr_error(err,
                             'Failed to free graph info(s)',
                             InferenceError,
                             NetRunErrorCode.TERMINATE)

            err = pbm.freeContext(self._backend_lib_key,
                                  self._backend_key,
                                  self._context_key,
                                  profilingLevel=profile_level_arg)
            _check_pnr_error(err,
                             'Failed to free context',
                             InferenceError,
                             NetRunErrorCode.FREE_CONTEXT)

            err = pbm.freeDeviceHandle(self._backend_lib_key, self._device_key)
            _check_pnr_error(err,
                             'Failed to free device',
                             InferenceError,
                             NetRunErrorCode.FREE_DEVICE)

            if profile_level_arg != py_net_run.ProfilingLevel.OFF:
                err = pbm.disposeProfileLogger(self._backend_lib_key)
                _check_pnr_error(err,
                                 'Failed to dispose profile logger',
                                 InferenceError,
                                 NetRunErrorCode.TERMINATE)

            err = pbm.freeBackendHandle(self._backend_lib_key, self._backend_key)
            _check_pnr_error(err,
                             'Failed to free backend handle',
                             InferenceError,
                             NetRunErrorCode.FREE_BACKEND)

            err = pbm.freeLogHandle(self._backend_lib_key, self._logger_key)
            _check_pnr_error(err,
                             'Failed to free log handle',
                             InferenceError,
                             NetRunErrorCode.TERMINATE)

            err = pbm.unloadBackendLib(self._backend_lib_key)
            _check_pnr_error(err,
                             'Failed to unload backend library',
                             InferenceError,
                             NetRunErrorCode.TERMINATE)

            if not binary_path_arg:
                err = pbm.unloadModelLib(self._model_lib_key)
                _check_pnr_error(err,
                                 'Failed to unload model library',
                                 InferenceError,
                                 NetRunErrorCode.TERMINATE)

            profiling_data = None
            if profile_level_arg != py_net_run.ProfilingLevel.OFF:
                profiling_data = output_dir_to_profiling_data(temp_directory.name, sdk_root)

                backend.after_run_hook(temp_directory.name, sdk_root)

            return output_data, profiling_data

    def generate_context_binary(self, config, backend, model, sdk_root, output_path,
                                output_filename):
        temp_directory = TemporaryDirectory()
        logger.debug(f'created temp directory: {temp_directory.name}')

        backend.before_generate_hook(temp_directory.name, sdk_root)

        log_level_arg = create_log_level_argument(config)
        op_package_arg = create_op_package_argument(backend)

        try:
            dlc_path = str(Path(model.dlc_path).resolve())
        except AttributeError:
            dlc_path = ''

        if dlc_path:
            model_path = Path(sdk_root, 'lib', 'x86_64-linux-clang', 'libQnnModelDlc.so')
        else:
            model_path = Path(model.model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Could not locate {model_path}")

        if output_filename:
            output_name = output_filename
        elif model.name:
            output_name = model.name
        else:
            output_name = 'context'
        output_filepath = Path(output_path, output_name + '.bin').resolve()

        with temporaryDirectoryChange(temp_directory.name):
            pbm = py_net_run.PythonBackendManager(logLevel=log_level_arg, opPackagePaths=op_package_arg)

            err = pbm.loadModelLib(str(model_path), self._model_lib_key)
            _check_pnr_error(err,
                             f'Failed to load model library: {model_path}',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.INITIALIZE)

            err = pbm.loadBackendLib(self._backend_lib_key, backend.backend_library)
            _check_pnr_error(err,
                             f'Failed to load backend library: {backend.backend_library}',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.INITIALIZE)

            err = pbm.createLogHandle(self._backend_lib_key, self._logger_key, log_level_arg)
            _check_pnr_error(err,
                             'Failed to initialize logging in the backend',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.INITIALIZE)

            [extension_lib_path, json_path] = create_backend_extension_argument(backend,
                                                                                temp_directory.name,
                                                                                sdk_root)
            if extension_lib_path:
                err = pbm.initializeBackendExtension(self._backend_lib_key,
                                                     extension_lib_path,
                                                     json_path,
                                                     py_net_run.AppType.QNN_APP_CONTEXT_BINARY_GENERATOR)
                _check_pnr_error(err,
                                 'Failed to initialize backend extensions',
                                 ContextBinaryGenerationError,
                                 NetRunErrorCode.INITIALIZE)

            err = pbm.createBackendHandle(self._backend_lib_key, self._logger_key, self._backend_key)
            _check_pnr_error(err,
                             'Failed to create a backend handle',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.CREATE_BACKEND)

            err = pbm.createDeviceHandle(self._backend_lib_key, self._logger_key, self._device_key)
            _check_pnr_error(err,
                             'Failed to create a device handle',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.CREATE_DEVICE)

            err = pbm.registerOpPackage(self._backend_lib_key, self._backend_key)
            _check_pnr_error(err,
                             'Failed to register op packages',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.REGISTER_OPPACKAGE)

            err = pbm.createContext(self._backend_lib_key,
                                    self._backend_key,
                                    self._device_key,
                                    self._context_key)
            _check_pnr_error(err,
                             'Failed to create a context',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.CREATE_CONTEXT)

            if dlc_path:
                err = pbm.composeGraphsFromDlc(self._backend_lib_key,
                                               self._backend_key,
                                               self._context_key,
                                               self._model_lib_key,
                                               dlc_path)
            else:
                err = pbm.composeGraphs(self._backend_lib_key,
                                        self._backend_key,
                                        self._context_key,
                                        self._model_lib_key)
            _check_pnr_error(err,
                             'Failed to compose graphs',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.COMPOSE_GRAPHS)

            err = pbm.finalizeGraphs(self._backend_lib_key,
                                     self._backend_key,
                                     self._context_key)
            _check_pnr_error(err,
                             'Failed to finalize graphs',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.FINALIZE_GRAPHS)

            logger.info(f'Writing context binary: {output_filepath}')
            err = pbm.saveContextToBinaryFile(self._backend_lib_key,
                                              self._backend_key,
                                              self._context_key,
                                              str(Path(output_filepath.parent, output_filepath.stem)),
                                              '')
            _check_pnr_error(err,
                             'Failed to write context binary to a file',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.FINALIZE_GRAPHS)
            if not output_filepath.is_file():
                raise RuntimeError(f'Failed to generate binary file: {output_filepath}')

            err = pbm.freeGraphsInfo(self._backend_lib_key, self._backend_key, self._context_key)
            _check_pnr_error(err,
                             'Failed to free graph info(s)',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.TERMINATE)

            err = pbm.freeContext(self._backend_lib_key,
                                  self._backend_key,
                                  self._context_key, )
            _check_pnr_error(err,
                             'Failed to free context',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.FREE_CONTEXT)

            err = pbm.freeDeviceHandle(self._backend_lib_key, self._device_key)
            _check_pnr_error(err,
                             'Failed to free device',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.FREE_DEVICE)

            err = pbm.freeBackendHandle(self._backend_lib_key, self._backend_key)
            _check_pnr_error(err,
                             'Failed to free backend handle',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.FREE_BACKEND)

            err = pbm.freeLogHandle(self._backend_lib_key, self._logger_key)
            _check_pnr_error(err,
                             'Failed to free log handle',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.TERMINATE)

            err = pbm.unloadBackendLib(self._backend_lib_key)
            _check_pnr_error(err,
                             'Failed to unload backend library',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.TERMINATE)

            err = pbm.unloadModelLib(self._model_lib_key)
            _check_pnr_error(err,
                             'Failed to unload model library',
                             ContextBinaryGenerationError,
                             NetRunErrorCode.TERMINATE)

            backend.after_generate_hook(temp_directory.name, sdk_root)

            return QnnContextBinary(output_filepath.stem, str(output_filepath))
