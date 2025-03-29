# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from tempfile import TemporaryDirectory
from pathlib import Path
import logging

from qti.aisw.core.model_level_api.executor.executor import Executor
from qti.aisw.core.model_level_api.model.context_binary import QnnContextBinary
from qti.aisw.core.model_level_api.utils.subprocess_executor import generate_input_list, \
    generate_config_file, output_dir_to_np_array, create_op_package_argument, \
    update_run_config_native_inputs
from qti.aisw.core.model_level_api.utils.qnn_profiling import output_dir_to_profiling_data
from qti.aisw.core.model_level_api.utils.exceptions import InferenceError, \
    ContextBinaryGenerationError, return_code_to_netrun_error_enum
from qti.aisw.core.model_level_api.config.qnn_config import QNNRunConfig

logger = logging.getLogger(__name__)


class X86SubprocessExecutor(Executor):
    def __init__(self):
        super().__init__()

    def run_inference(self, config, backend, model, sdk_root, input_data):
        temp_directory = TemporaryDirectory()
        logger.debug(f"created dir: {temp_directory.name}")
        backend.before_run_hook(temp_directory.name, sdk_root)

        input_list_filename, _ = generate_input_list(input_data, temp_directory.name)
        config_file_arg, _ = generate_config_file(backend, temp_directory.name, sdk_root)
        op_package_arg, _ = create_op_package_argument(backend)

        backend_lib = sdk_root + '/lib/x86_64-linux-clang/' + backend.backend_library
        if not Path(backend_lib).is_file():
            raise FileNotFoundError(f"Could not find backend library: {backend_lib}")

        netrun = sdk_root + '/bin/x86_64-linux-clang/qnn-net-run'
        model_arg = self._create_inference_model_argument(model, sdk_root)
        output_dir = temp_directory.name + '/output'

        netrun_command = f'{netrun} --backend {backend_lib} --input_list {input_list_filename} ' \
                         f'{model_arg} --output_dir {output_dir} {config_file_arg} ' \
                         f'{op_package_arg} '

        if not config:
            config = QNNRunConfig()
        update_run_config_native_inputs(config, input_data)
        netrun_command += config.as_command_line_args()

        logger.debug(f'Running command: {netrun_command}')
        return_code, stdout, stderr = backend.target.run_command(netrun_command,
                                                                 cwd=temp_directory.name)
        if return_code != 0:
            err_str = f"qnn-net-run execution failed, stdout: {stdout}, stderr: {stderr}"
            netrun_error_enum = return_code_to_netrun_error_enum(return_code)
            if netrun_error_enum:
                raise InferenceError(netrun_error_enum, err_str)
            raise RuntimeError(err_str)

        if config and config.log_level:
            print(f'stdout: ', *stdout, sep='\n')

        profiling_data = None
        if config and config.profiling_level:
            profiling_data = output_dir_to_profiling_data(output_dir, sdk_root)

        backend.after_run_hook(temp_directory.name, sdk_root)

        native_outputs = config and config.use_native_output_data
        return output_dir_to_np_array(output_dir, native_outputs), profiling_data

    def generate_context_binary(self, config, backend, model, sdk_root, output_path,
                                output_filename):
        temp_directory = TemporaryDirectory()
        logger.debug(f"created dir: {temp_directory.name}")
        backend.before_generate_hook(temp_directory.name, sdk_root)

        backend_lib = sdk_root + '/lib/x86_64-linux-clang/' + backend.backend_library
        if not Path(backend_lib).is_file():
            raise FileNotFoundError(f"Could not find backend library: {backend_lib}")

        context_binary_generator = sdk_root + '/bin/x86_64-linux-clang/qnn-context-binary-generator'
        model_arg = self._create_context_binary_generator_model_argument(model, sdk_root)

        config_file_arg, _ = generate_config_file(backend, temp_directory.name, sdk_root)
        op_package_arg, _ = create_op_package_argument(backend)

        if output_filename:
            output_filename = output_filename + '.bin'
        elif model.name:
            output_filename = model.name + '.bin'
        else:
            output_filename = 'context.bin'
        abs_output_filepath = Path(output_path).absolute().joinpath(output_filename)

        context_bin_command = f'{context_binary_generator} --backend {backend_lib} ' \
                              f'{model_arg} ' \
                              f'--binary_file {abs_output_filepath.stem} ' \
                              f'--output_dir {abs_output_filepath.parent} {config_file_arg} ' \
                              f'{op_package_arg} '

        if config:
            context_bin_command += config.as_command_line_args()

        logger.debug(f'Running command: {context_bin_command}')
        return_code, stdout, stderr = backend.target.run_command(context_bin_command,
                                                                 cwd=temp_directory.name)
        if return_code != 0:
            err_str = f"qnn-context-binary-generator execution failed, stdout: {stdout}, stderr: " \
                      f"{stderr}"
            netrun_error_enum = return_code_to_netrun_error_enum(return_code)
            if netrun_error_enum:
                raise ContextBinaryGenerationError(netrun_error_enum, err_str)
            raise RuntimeError(err_str)

        if config and config.log_level:
            print(f'stdout: ', *stdout, sep='\n')

        backend.after_generate_hook(temp_directory.name, sdk_root)

        return QnnContextBinary(abs_output_filepath.stem, str(abs_output_filepath))

    @staticmethod
    def _create_context_binary_argument(model):
        binary_path = Path(model.binary_path)
        if not binary_path.is_file():
            raise FileNotFoundError(f"Could not find context binary: {binary_path}")
        return f'--retrieve_context {binary_path.resolve()}'

    @staticmethod
    def _create_model_lib_argument(model):
        model_path = Path(model.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Could not find model library: {model_path}")
        return f'--model {model_path.resolve()}'

    @staticmethod
    def _create_dlc_argument(model, sdk_root):
        dlc_path = Path(model.dlc_path)
        if not dlc_path.is_file():
            raise FileNotFoundError(f"Could not find DLC: {dlc_path}")
        qnn_model_dlc_path = Path(sdk_root, 'lib', 'x86_64-linux-clang', 'libQnnModelDlc.so')
        if not qnn_model_dlc_path.is_file():
            raise FileNotFoundError(f"Could not locate {qnn_model_dlc_path}")
        return f'--dlc_path {dlc_path.resolve()} --model {qnn_model_dlc_path}'

    @staticmethod
    def _create_inference_model_argument(model, sdk_root):
        try:
            return X86SubprocessExecutor._create_model_lib_argument(model)
        except AttributeError:
            pass

        try:
            return X86SubprocessExecutor._create_dlc_argument(model, sdk_root)
        except AttributeError:
            pass

        try:
            return X86SubprocessExecutor._create_context_binary_argument(model)
        except AttributeError:
            raise ValueError("Could not retrieve path from provided model")

    @staticmethod
    def _create_context_binary_generator_model_argument(model, sdk_root):
        try:
            return X86SubprocessExecutor._create_model_lib_argument(model)
        except AttributeError:
            pass

        try:
            return X86SubprocessExecutor._create_dlc_argument(model, sdk_root)
        except AttributeError:
            raise ValueError("Could not retrieve path from provided model")
