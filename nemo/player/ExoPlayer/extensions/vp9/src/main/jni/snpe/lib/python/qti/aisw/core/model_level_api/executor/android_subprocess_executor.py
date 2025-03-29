#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from tempfile import TemporaryDirectory
from pathlib import Path, PurePosixPath
import logging
from uuid import uuid4
import weakref

from qti.aisw.core.model_level_api.executor.executor import Executor
from qti.aisw.core.model_level_api.model.context_binary import QnnContextBinary
from qti.aisw.core.model_level_api.workflow.workflow import WorkflowMode
from qti.aisw.core.model_level_api.utils.subprocess_executor import generate_input_list, \
    generate_config_file, output_dir_to_np_array, create_op_package_argument, \
    update_run_config_native_inputs
from qti.aisw.core.model_level_api.utils.qnn_profiling import output_dir_to_profiling_data
from qti.aisw.core.model_level_api.utils.exceptions import InferenceError, \
    ContextBinaryGenerationError, return_code_to_netrun_error_enum
from qti.aisw.core.model_level_api.config.qnn_config import QNNRunConfig

logger = logging.getLogger(__name__)


class AndroidTempDirectory:
    def __init__(self, directory_name, target):
        self.name = '/data/local/tmp/' + directory_name
        self._target = target

        logger.debug(f"Creating temp dir {self.name}")
        self._target.make_directory(self.name)

        # create a callable that will be called upon garbage collection if it is not used as a
        # context manager. This finalizer can only be called once, so if it is used as a context
        # manager, it is called during __exit__ and will be a no-op when it is subsequently
        # garbage collected
        self._finalizer = weakref.finalize(self, self._cleanup, self.name, self._target)

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        # mark the finalizer as dead so _cleanup is not redundantly called during garbage collection
        self._finalizer.detach()
        self._cleanup(self.name, self._target)

    @classmethod
    def _cleanup(cls, name, target):
        logger.debug(f"Deleting temp dir {name}")
        target.remove(name)

class AndroidSubprocessExecutor(Executor):
    def __init__(self):
        super().__init__()
        # a directory which will be used to store backend, model, and executables that are relevant
        # for the entire lifetime of the Executor. Must be created during setup() below since the
        # remote device has not been selected when the Executor is created.
        self._artifact_directory = None


    def setup(self, workflow_mode, backend, model, sdk_root):
        artifacts_to_push = []
        sdk_root = Path(sdk_root)

        if workflow_mode == WorkflowMode.INFERENCE:
            executable_name = 'qnn-net-run'
        elif workflow_mode == WorkflowMode.CONTEXT_BINARY_GENERATION:
            executable_name = 'qnn-context-binary-generator'
        else:
            raise RuntimeError(f'Unknown WorkflowMode {workflow_mode}')

        executable_path = sdk_root / 'bin' / 'aarch64-android' / executable_name
        if not executable_path.is_file():
            raise FileNotFoundError(f'Could not find executable {executable_path}')
        artifacts_to_push.append(executable_path)

        backend_lib_path = sdk_root / 'lib' / 'aarch64-android' / backend.backend_library
        if not backend_lib_path.is_file():
            raise FileNotFoundError(f'Could not find backend library: {backend_lib_path}')
        artifacts_to_push.append(backend_lib_path)

        backend_required_artifacts = [Path(artifact)
                                      for artifact in backend.get_required_artifacts(str(sdk_root))]
        artifacts_to_push.extend(backend_required_artifacts)

        artifacts_to_push.extend(self._get_model_artifacts(model, sdk_root))

        # generate a random uuid which will be used as a directory name on device
        random_uuid_str = str(uuid4())
        self._artifact_directory = AndroidTempDirectory(random_uuid_str, backend.target)
        for artifact in artifacts_to_push:
            backend.target.push(str(artifact), self._artifact_directory.name)


    def run_inference(self, config, backend, model, sdk_root, input_data):
        temp_directory = TemporaryDirectory()
        logger.debug(f"created temp dir: {temp_directory.name}")

        # create temp directory on device based on host temp directory name to make the directory
        # name on device unique
        temp_dir_name = Path(temp_directory.name).name
        with AndroidTempDirectory(temp_dir_name, backend.target) as android_temp_directory:
            backend.before_run_hook(android_temp_directory, sdk_root)

            input_list_filename, input_files = generate_input_list(input_data,
                                                                   temp_directory.name,
                                                                   android_temp_directory)
            device_input_list_filename = PurePosixPath(android_temp_directory ,'input_list.txt')

            config_file_arg, config_file_artifacts = generate_config_file(backend,
                                                                          temp_directory.name,
                                                                          sdk_root,
                                                                          android_temp_directory)
            op_package_arg, op_package_artifacts = create_op_package_argument(
                backend,
                android_temp_directory)

            device_backend_lib = PurePosixPath(self._artifact_directory.name,
                                               backend.backend_library)

            device_netrun = PurePosixPath(self._artifact_directory.name, 'qnn-net-run')
            model_arg = self._create_inference_model_argument(model,
                                                              self._artifact_directory.name)
            output_dir = PurePosixPath(android_temp_directory, 'output')

            # push artifacts to device
            backend.target.push(input_list_filename, device_input_list_filename)
            for input_file in input_files:
                # if the input file needs to be renamed, it will be colon separated like
                # <host_filesystem_path>:<renamed_filename>, extract both of these strings
                input_file, _, input_file_rename = input_file.partition(':')
                # input_file_rename is an empty string if the file does not need to be renamed (i.e.
                # if the separator is not found), so it can be unconditionally appended to the dest
                # filepath because an empty string will result in the file being pushed without
                # being renamed because the dst path ends in '/'
                dest_filepath = PurePosixPath(android_temp_directory, input_file_rename)
                backend.target.push(input_file, str(dest_filepath))
            for artifact in config_file_artifacts:
                backend.target.push(artifact, android_temp_directory)
            for artifact in op_package_artifacts:
                backend.target.push(artifact, android_temp_directory)

            netrun_command = f'{device_netrun} --backend {device_backend_lib} ' \
                             f'{model_arg} --input_list {device_input_list_filename} ' \
                             f'--output_dir {output_dir} {config_file_arg} {op_package_arg} '

            if not config:
                config = QNNRunConfig()

            update_run_config_native_inputs(config, input_data)
            netrun_command += config.as_command_line_args()

            command_env = {'LD_LIBRARY_PATH': f'$(pwd):{self._artifact_directory.name}',
                           'ADSP_LIBRARY_PATH': f'$(pwd);{self._artifact_directory.name}'}
            logger.debug("running command " + netrun_command)
            return_code, stdout, stderr = backend.target.run_command(netrun_command,
                                                                     cwd=android_temp_directory,
                                                                     env=command_env)
            if return_code != 0:
                err_str = f"qnn-net-run execution failed, stdout: {stdout}, stderr: {stderr}, " \
                          f"check adb logcat for additional logs"
                netrun_error_enum = return_code_to_netrun_error_enum(return_code)
                if netrun_error_enum:
                    raise InferenceError(netrun_error_enum, err_str)
                raise RuntimeError(err_str)

            backend.target.pull(str(output_dir),
                                temp_directory.name)
            host_output_dir = Path(temp_directory.name, 'output')
            if not Path(host_output_dir).is_dir():
                raise RuntimeError("Failed to pull outputs from device")

            profiling_data = None
            if config and config.profiling_level:
                profiling_data = output_dir_to_profiling_data(host_output_dir, sdk_root)

            backend.after_run_hook(android_temp_directory, sdk_root)

            native_outputs = config and config.use_native_output_data
            return output_dir_to_np_array(host_output_dir, native_outputs), profiling_data

    def generate_context_binary(self, config, backend, model, sdk_root, output_path,
                                output_filename):
        temp_directory = TemporaryDirectory()
        logger.debug(f"created temp dir: {temp_directory.name}")

        # create temp directory on device based on host temp directory name to make the directory
        # name on device unique
        temp_dir_name = Path(temp_directory.name).name
        with AndroidTempDirectory(temp_dir_name, backend.target) as android_temp_directory:
            backend.before_generate_hook(android_temp_directory, sdk_root)

            device_backend_lib = PurePosixPath(self._artifact_directory.name,
                                               backend.backend_library)

            device_context_bin_generator = PurePosixPath(self._artifact_directory.name,
                                                         'qnn-context-binary-generator')

            config_file_arg, config_file_artifacts = generate_config_file(backend,
                                                                          temp_directory.name,
                                                                          sdk_root,
                                                                          android_temp_directory)
            op_package_arg, op_package_artifacts = create_op_package_argument(
                backend,
                android_temp_directory)

            model_arg = self._create_context_binary_generator_model_argument(
                model,
                self._artifact_directory.name)

            if output_filename:
                binary_file = output_filename
            elif model.name:
                binary_file = model.name
            else:
                binary_file = 'context'

            # push artifacts to device
            for artifact in config_file_artifacts:
                backend.target.push(artifact, android_temp_directory)
            for artifact in op_package_artifacts:
                backend.target.push(artifact, android_temp_directory)

            context_bin_command = f'{device_context_bin_generator} ' \
                                  f'--backend {device_backend_lib} ' \
                                  f'{model_arg} --binary_file {binary_file} ' \
                                  f'{config_file_arg} {op_package_arg} '

            if config:
                context_bin_command += config.as_command_line_args()

            command_env = {'LD_LIBRARY_PATH': f'$(pwd):{self._artifact_directory.name}',
                           'ADSP_LIBRARY_PATH': f'$(pwd);{self._artifact_directory.name}'}
            logger.debug("running command " + context_bin_command)
            return_code, stdout, stderr = backend.target.run_command(context_bin_command,
                                                                     cwd=android_temp_directory,
                                                                     env=command_env)
            if return_code != 0:
                err_str = f"qnn-context-binary-generator failed, stdout: {stdout}, stderr: " \
                          f"{stderr}, check adb logcat for additional logs"
                netrun_error_enum = return_code_to_netrun_error_enum(return_code)
                if netrun_error_enum:
                    raise ContextBinaryGenerationError(netrun_error_enum, err_str)
                raise RuntimeError(err_str)

            device_bin_path = PurePosixPath(android_temp_directory, 'output', binary_file + '.bin')
            backend.target.pull(str(device_bin_path),
                                output_path)
            output_bin = Path(output_path, binary_file + '.bin').absolute()
            if not output_bin.is_file():
                raise RuntimeError("Failed to pull context binary from device")

            backend.after_generate_hook(android_temp_directory, sdk_root)

            return QnnContextBinary(output_bin.stem, str(output_bin))


    @staticmethod
    def _create_context_binary_argument(model, temp_directory):
        binary_path = Path(model.binary_path)
        binary_device_path = PurePosixPath(temp_directory, binary_path.name)
        return f'--retrieve_context {binary_device_path}'

    @staticmethod
    def _create_model_lib_argument(model, temp_directory):
        model_path = Path(model.model_path)
        model_device_path = PurePosixPath(temp_directory, model_path.name)
        return f'--model {model_device_path}'

    @staticmethod
    def _create_dlc_argument(model, temp_directory):
        dlc_path = Path(model.dlc_path)
        qnn_model_dlc_device_path = PurePosixPath(temp_directory, 'libQnnModelDlc.so')
        dlc_device_path = PurePosixPath(temp_directory, dlc_path.name)
        return f'--dlc_path {dlc_device_path} --model {qnn_model_dlc_device_path}'

    @staticmethod
    def _create_inference_model_argument(model, temp_directory):
        try:
            return AndroidSubprocessExecutor._create_model_lib_argument(model,
                                                                        temp_directory)
        except AttributeError:
            pass

        try:
            return AndroidSubprocessExecutor._create_dlc_argument(model,
                                                                  temp_directory)
        except AttributeError:
            pass

        try:
            return AndroidSubprocessExecutor._create_context_binary_argument(model,
                                                                             temp_directory)
        except AttributeError:
            raise ValueError("Could not retrieve path to provided model")

    @staticmethod
    def _create_context_binary_generator_model_argument(model, temp_directory):
        try:
            return AndroidSubprocessExecutor._create_model_lib_argument(model,
                                                                        temp_directory)
        except AttributeError:
            pass

        try:
            return AndroidSubprocessExecutor._create_dlc_argument(model,
                                                                  temp_directory)
        except AttributeError:
            raise ValueError("Could not retrieve path to provided model")

    @staticmethod
    def _get_model_artifacts(model, sdk_root):
        try:
            binary_path = Path(model.binary_path)
            if not binary_path.is_file():
                raise FileNotFoundError(f"Could not find context binary: {binary_path}")
            return [binary_path]
        except AttributeError:
            pass

        try:
            model_path = Path(model.model_path)
            if not model_path.is_file():
                raise FileNotFoundError(f"Could not find model library: {model_path}")
            return [model_path]
        except AttributeError:
            pass

        try:
            dlc_path = Path(model.dlc_path)
            if not dlc_path.is_file():
                raise FileNotFoundError(f"Could not find DLC: {dlc_path}")
            qnn_model_dlc_path = Path(sdk_root, 'lib', 'aarch64-android', 'libQnnModelDlc.so')
            if not qnn_model_dlc_path.is_file():
                raise FileNotFoundError(f"Could not find {qnn_model_dlc_path}")
            return [dlc_path, qnn_model_dlc_path]
        except AttributeError:
            raise ValueError("Could not retrieve path to provided model")
