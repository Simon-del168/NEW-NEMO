# =============================================================================
#
#  Copyright (c) 2019-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import importlib
import os
import subprocess
import sys
import zipfile
from collections import OrderedDict

from qti.aisw.accuracy_debugger.lib.inference_engine import inference_engine_repository
from qti.aisw.accuracy_debugger.lib.inference_engine.inference_engines.nd_inference_engine import InferenceEngine
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType, Engine, host_device_to_arch
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path, sanitize_output_tensor_files
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.utils.nd_verifier_utility import get_tensor_names_from_dlc


@inference_engine_repository.register(cls_type=ComponentType.inference_engine, framework=None,
                                      engine=Engine.SNPE, engine_version="1.22.2.233")
class SNPEInferenceEngine(InferenceEngine):

    class DLCData:

        def __init__(self, output_layers, host_dlc_path, host_model_inputs, target_model_inputs):
            self.output_layers = output_layers
            # dict of input tensor names to input data locations on host device
            self.host_model_inputs = host_model_inputs
            # dict of input tensor names to input data locations on target device
            self.target_model_inputs = target_model_inputs

            # note that the dlc paths are the host and target base dlc in coarse grained mode
            self.host_dlc_path = host_dlc_path
            self.target_dlc_path = None

            self.host_input_list_path = None
            self.target_input_list_path = None

    def __init__(self, context, converter, executor):
        super().__init__(context, converter, executor)
        # Fields from context
        self.engine_type = context.engine
        self.stage = context.stage
        self.fine_grained_mode = context.fine_grain_mode
        self.engine_path = context.engine_path
        self.host_device = context.host_device
        self.target_device = context.target_device
        self.model_inputs = context.input_tensor
        self.model_outputs = context.output_tensor
        self.model_path = context.model_path
        self.quantization_overrides = context.quantization_overrides
        self.extra_converter_args = context.extra_converter_args
        self.extra_runtime_args = context.extra_runtime_args
        self.precision = context.precision
        self.userlogs = context.userlogs
        self.extra_quantizer_args = context.extra_quantizer_args
        self.snpe_lib_python = context.snpe_lib_python
        self.snpe_dlc_utils_package = context.snpe_dlc_utils_package
        self.host_output_dir = context.output_dir
        self.target_arch = self.executor.target_arch
        self.target_path = self.executor.target_path
        if self.target_arch in ["wos-remote", "wos"]:
            self.backend_paths = context.backend_locations["wos"]
        elif self.target_arch == "x86_64-windows-msvc":
            self.backend_paths = context.backend_locations["x86_64-windows-msvc"]
        elif self.target_arch == "aarch64-android":
            self.backend_paths = context.backend_locations["android"]
        self.static_model = context.static_model
        self.runtime = context.runtime
        self.input_list = context.input_list
        self.weights_bitwidth = context.weights_bitwidth
        self.bias_bitwidth = context.bias_bitwidth
        self.act_bitwidth = context.act_bitwidth
        self.no_weight_quantization = context.no_weight_quantization
        self.use_symmetric_quantize_weights = context.use_symmetric_quantize_weights
        self.use_adjusted_weights_quantizer = context.use_adjusted_weights_quantizer
        self.use_enhanced_quantizer = context.use_enhanced_quantizer
        self.override_params = context.override_params
        self.offline_prepare = context.offline_prepare
        self.htp_socs = context.htp_socs
        self.debug_mode = context.debug_mode
        self.snpe_quantizer_config = context.snpe_quantizer_config
        self.extra_quantizer_args = context.extra_quantizer_args
        if self.target_arch in ["x86_64-windows-msvc", "wos"]:
            self.executor.executable = self.executor.executable + '.exe'
            self.snpe_quantizer_config['executable'] = self.snpe_quantizer_config[
                'windows_executable']
        if self.engine_path.endswith('.zip'):
            self.engine_zip_path = self.engine_path
            with zipfile.ZipFile(self.engine_zip_path, 'r') as zip_file:
                zip_file.extractall(self.host_output_dir)
                for file in zip_file.namelist():
                    os.chmod(os.path.join(self.host_output_dir, file), 0o755)
            if './' in zip_file.namelist()[0]:
                self.engine_path = os.path.join(self.host_output_dir,
                                                os.path.dirname(zip_file.namelist()[1]))
            else:
                self.engine_path = os.path.join(self.host_output_dir,
                                                os.path.dirname(zip_file.namelist()[0]))
        elif os.path.isdir(self.engine_path):
            self.engine_zip_path = None
        else:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_PATH_INVALID")(self.engine_path))

        # Working directory and file location constants
        self._INPUT_LIST_DIR = os.path.join(self.host_output_dir, 'input_list_files')
        self._CONVERTER_LOCATION = context.converter_location.format(
            engine_path=self.engine_path, host_arch=host_device_to_arch[self.host_device.device])
        self._BASE_DLC_PATH = os.path.join(self.host_output_dir, 'base.dlc')

        # base dlc and associated info
        self.base_dlc_data = None
        self.base_dlc_info = None
        # list of sub dlcs and their associated information
        self.sub_dlc_data_list = []

        self.logger = context.logger

        # both _original_input_paths and _full_path_input_paths could contain tensor specific
        # inputs like input:=datapath.raw or just normal data path like path/to/data1.raw
        # _full_path_input_paths are basically the same as _original_input_paths except all
        # the path inside are absolute path
        # paths in _original_input_paths could be absolute or relative.
        # We should use _full_path_input_paths to refer to data path
        # from the input_list for most of the time
        self._original_input_paths = None
        self._full_path_input_paths = None

    # -------------------------------------------------- COMMON ------------------------------------------------------------

    def _common_setup(self):
        """Set the base dlc, and create directories for input lists on host and
        target."""
        if self.target_device.device in ["x86", "x86_64-windows-msvc", "wos"]:
            self._TARGET_INPUT_LIST_DIR = self._INPUT_LIST_DIR
            self._TARGET_EXCUTE_DIR = self.host_output_dir
        else:
            self._TARGET_INPUT_LIST_DIR = os.path.join(self.executor.target_path, 'data')
            self._TARGET_EXCUTE_DIR = self.executor.target_path

        self._set_original_input_paths()
        self._set_input_list()

        self._set_host_environment()
        # Create the base dlc and info
        self._set_base_dlc()

        # Create directory for input list files on host
        os.makedirs(self._INPUT_LIST_DIR, exist_ok=True)
        # Create directory for input list files on target
        code, _, err = self.target_device.make_directory(self._TARGET_INPUT_LIST_DIR)
        if code != 0:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_MKDIR_FAILED")(err))

        if self.target_device.device == "wos" and "dsp" in self.runtime:
            binary_paths = self.get_binaries_to_push()
            destination_dir = os.path.join(self.engine_path, "lib", "aarch64-windows-msvc")
            for source, _ in binary_paths:
                if "hexagon" in source:
                    destination = os.path.join(destination_dir, os.path.basename(source))
                    code, _, err = self.target_device.push(source, destination)
                    if code != 0:
                        raise InferenceEngineError(
                            get_message("ERROR_INFERENCE_ENGINE_BINARIES_FAILED_DEVICE"))

    def _set_original_input_paths(self):
        """This function should be called before any modification to the
        original input_list."""
        if self.input_list is not None:
            # get original input list paths
            # _original_input_paths stores all rel input paths in form of list of lists;
            # ie. if a input list has 2 batch and each batch require 3 inputs then the _original_input_paths would look like:
            # [[batch1_input1,batch1_input2,batch1_input3],[batch2_input1,batch2_input2,batch2_input3]]
            with open(self.input_list, "r") as input_list:
                self._original_input_paths = [
                    line.strip(' \n').split(' ') for line in input_list.readlines()
                ]
            if self._original_input_paths == []: self.input_list = None

    def _set_host_environment(self):
        self.host_env = {}
        self.host_env["PATH"] = self._CONVERTER_LOCATION + os.pathsep \
                                + ('' if self.host_device.device in ["x86_64-windows-msvc", "wos"] else '$PATH')
        self.host_env['PYTHONPATH'] = os.path.join(
            self.engine_path, self.snpe_lib_python) + os.pathsep + (
                '' if self.host_device.device in ["x86_64-windows-msvc", "wos"] else '$PYTHONPATH')

        print(self.host_env)
        sys.path.insert(0, os.path.join(self.engine_path, self.snpe_lib_python))

    def _set_base_dlc(self):
        """Convert the entire model to set the base dlc and extract model
        info."""
        base_model_inputs = None
        if self.stage == 'source':
            # Execute the conversion
            conversion_inputs = {input_name: dim for input_name, dim, data in self.model_inputs}
            self._execute_conversion(self._BASE_DLC_PATH, conversion_inputs, self.model_outputs)
            base_model_inputs = {input_name: data for input_name, dim, data in self.model_inputs}
        elif self.stage == 'converted' or self.stage == 'compiled':
            self._BASE_DLC_PATH = self.static_model
        else:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_SNPE_STAGE_ERROR"))

        # quantize the model if it's float dlc ie. from stage source or converted.
        if (self.stage == 'source' or self.stage == 'converted') and "dsp" in self.runtime:
            try:
                self._BASE_DLC_PATH = self._execute_dlc_quantization(self._BASE_DLC_PATH)
            except Exception as e:
                self.logger.info("DLC Quantized failed, please check if the model is already quantized.")
        # Set base dlc data object
        self.base_dlc_data = self.DLCData(output_layers=self.model_outputs,
                                          host_dlc_path=self._BASE_DLC_PATH,
                                          host_model_inputs=base_model_inputs,
                                          target_model_inputs=None)
        # Extract info from initial dlc
        sys.path.insert(0, os.path.join(self.engine_path, self.snpe_dlc_utils_package))
        self.snpe_dlc = importlib.import_module('snpe_dlc_utils')
        m = self.snpe_dlc.ModelInfo()
        m.load(self.base_dlc_data.host_dlc_path)
        if len(m.graph_names) > 1:
            # If incase in future multiple graphs are created then raise an exception to catch this scenario
            raise InferenceEngineError(
                f"Multiple graphs found: {m.graph_names}, this scenario is not handled yet.")
        graph_names = [item for item in m.graph_names]
        rows, total_params, total_macs, total_memory_cpu, total_memory_fxp, total_op_types, is_stripped = m.extract_model_info(
            graph_names[0])
        self.base_dlc_info = rows

    def _set_input_list(self):
        """This function prepends input list's current directory to each of the
        relative paths in input list, resulting in an input list with absolute
        paths.

        """

        if not self.input_list: return

        curr_dir = os.path.dirname(self.input_list)

        # this here basically means for each item in each line of _original_input_paths, make it an absolute path
        self._full_path_input_paths = [
            [get_absolute_path(rel_path, checkExist=True, pathPrepend=curr_dir) if ":=" not in rel_path \
                 else rel_path.split(":=")[0] + ":=" + get_absolute_path(rel_path.split(":=")[1], checkExist=True,
                                                                         pathPrepend=curr_dir) for rel_path in
             per_batch] \
            for per_batch in self._original_input_paths]
        # create a new input_list_file in the output_dir and use that
        self.input_list = os.path.join(self.host_output_dir, os.path.basename(self.input_list))
        with open(self.input_list, "w") as input_list:
            input_list.write('\n'.join(
                [' '.join(per_batch) for per_batch in self._full_path_input_paths]))

    # -------------------------------------------------- COMMON HELPERS ----------------------------------------------------

    @staticmethod
    def _write_input_list(dlc_data, input_list_dir):
        """Create an input list on the host device.

        :param dlc_data: DLC data object
        :param input_list_dir: Directory to place input list files
        """

        # Set input list file name
        layers = ['-'.join(layer_name.split('/')) for layer_name in dlc_data.output_layers]
        layer_file_name = '&'.join(layers) + '.txt'
        file_path = os.path.join(input_list_dir, layer_file_name)
        dlc_data.host_input_list_path = file_path
        print("####file_path####:{}".format(file_path))
        string_to_write = "#"
        for output in dlc_data.output_layers:
            string_to_write += output + ' '
        string_to_write += '\n'

        for batch in dlc_data.target_model_inputs:
            for items in batch:
                print("item in write_input_list: {}".format(items))
                string_to_write += items + ' '
            string_to_write += '\n'

        SNPEInferenceEngine._write(dlc_data, string_to_write)

    @staticmethod
    def _write(dlc_data, string_to_write):
        with open(dlc_data.host_input_list_path, 'w+') as f:
            print("####stringtowrite#####: {}".format(string_to_write))
            f.write(string_to_write)

    def _execute_dlc_quantization(self, dlc_path):
        """Execute DLC quantization.

        param dlc_path: Path to the converter dlc result
        """
        file_name = '.'.join(os.path.basename(dlc_path).split('.')[:-1])
        snpe_quantize_outputs = os.path.join(self.host_output_dir, file_name + "_quantized.dlc")
        try:
            convert_command = [
                self.snpe_quantizer_config["executable"],
                self.snpe_quantizer_config["arguments"]["dlc_path"], dlc_path
            ]

            if self.input_list and self.precision not in [None, 'fp16', 'fp32']:
                convert_command += [
                    self.snpe_quantizer_config["arguments"]["input_list"], self.input_list
                ]

            convert_command += [
                self.snpe_quantizer_config["arguments"]["act_bitwidth"] + "=" +
                str(self.act_bitwidth)
            ]
            convert_command += [
                self.snpe_quantizer_config["arguments"]["bias_bitwidth"] + "=" +
                str(self.bias_bitwidth)
            ]
            convert_command += [
                self.snpe_quantizer_config["arguments"]["output_path"], snpe_quantize_outputs
            ]
            if not self.no_weight_quantization:
                if self.use_symmetric_quantize_weights:
                    convert_command += [
                        self.snpe_quantizer_config["arguments"]["use_symmetric_quantize_weights"]
                    ]
                else:
                    convert_command += [
                        self.snpe_quantizer_config["arguments"]["weights_bitwidth"] + "=" +
                        str(self.weights_bitwidth)
                    ]
            else:
                convert_command += [
                    self.snpe_quantizer_config["arguments"]["no_weight_quantization"]
                ]

            if self.use_adjusted_weights_quantizer:
                convert_command += [
                    self.snpe_quantizer_config["arguments"]["use_adjusted_weights_quantizer"]
                ]
            if self.use_enhanced_quantizer:
                convert_command += [
                    self.snpe_quantizer_config["arguments"]["use_enhanced_quantizer"]
                ]
            if self.override_params:
                convert_command += [self.snpe_quantizer_config["arguments"]["override_params"]]

            if self.snpe_quantizer_config["executable"] != "snpe-dlc-quant.exe":
                if self.offline_prepare:
                    convert_command += [self.snpe_quantizer_config["arguments"]["enable_htp"]]
                if self.htp_socs:
                    convert_command += [
                        self.snpe_quantizer_config["arguments"]["htp_socs"] + "=" + self.htp_socs
                    ]
            if self.extra_quantizer_args:
                convert_command += [self.extra_quantizer_args]

            convert_command += self.snpe_quantizer_config["arguments"]["flags"]
            convert_command_str = ' '.join(convert_command)

            log_string = 'Running DLC quantize with: ' + \
                         'Inputs: ' + str(dlc_path) + ' ' + \
                         'Outputs: ' + str(snpe_quantize_outputs)
            self.logger.info(log_string)
            code, _, err = self.host_device.execute(commands=[convert_command_str],
                                                    cwd=self.engine_path, env=self.host_env)
            if code != 0 or not os.path.exists(snpe_quantize_outputs):
                raise InferenceEngineError(
                    get_message('"ERROR_INFERENCE_ENGINE_SNPE_DLC_QUANTIZED_FAILED"'))
            self.logger.info('DLC model quantized successfully')
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(
                get_message('"ERROR_INFERENCE_ENGINE_SNPE_DLC_QUANTIZED_FAILED"'))

        if os.path.isfile(snpe_quantize_outputs):
            return snpe_quantize_outputs

    def _execute_conversion(self, dlc_path, inputs, outputs):
        """Convert a model into a dlc.

        :param dlc_path: Path to save the new dlc
        :param inputs: Input names and dimensions to the model
        :param outputs: Output names of the model
        """

        convert_command = self.converter.build_convert_command(self.model_path, inputs, outputs,
                                                               dlc_path,
                                                               self.quantization_overrides,
                                                               self.extra_converter_args)
        log_string = 'Starting conversion with: ' + \
                     'Inputs: ' + str(list(inputs.keys())) + ' ' + \
                     'Outputs: ' + str(outputs)
        self.logger.info(log_string)

        print(self.host_env)

        try:
            code, _, err = self.host_device.execute(commands=[convert_command], env=self.host_env)
            if code != 0:
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_BASE_CONVERSION_FAILED'))

            self.logger.info('Model converted successfully')
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_CONVERSION_FAILED'))

    def _execute_inference(self, dlc_data):
        """Run inference.

        :param dlc_data: DLC data object
        """

        # TODO: Fix the target arch name to arm64x-windows once libs and bins are shipped in this arch
        arch = 'aarch64-windows-msvc' if self.target_arch in ['wos-remote', 'wos'
                                                              ] else self.target_arch
        if self.target_device.device in ["x86", "x86_64-windows-msvc", "wos"]:
            self.host_env['LD_LIBRARY_PATH'] = os.path.join(self.engine_path, "lib", arch)
        else:
            for library_path_name, path in self.executor.get_execute_environment_variables():
                self.host_env[library_path_name] = path

        if self.target_arch == "wos-remote":
            snpe_net_runner = self.executor.executable
            self.executor.executable = './' + self.executor.executable + '.exe'

        execute_command = self.executor.build_execute_command(dlc_data.target_dlc_path,
                                                              dlc_data.target_input_list_path,
                                                              self.userlogs)

        # Revert back to original name
        if self.target_arch == "wos-remote":
            self.executor.executable = snpe_net_runner

        log_string = 'Running inference with: ' + \
                     'Inputs: ' + str(dlc_data.target_model_inputs) + ' ' + \
                     'Outputs: ' + str(dlc_data.output_layers)
        self.logger.info(log_string)

        try:
            code, _, err = self.target_device.execute(
                commands=[execute_command], cwd=self._TARGET_EXCUTE_DIR,
                env=None if self.target_arch == "wos-remote" else self.host_env)
            if code != 0:
                raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED'))
            self.logger.info('Inference executed successfully')
        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_INFERENCE_FAILED'))

    def _pull_inference_results(self):
        """Pull inference results from target device to host."""
        code, _, err = self.target_device.pull(os.path.join(self._TARGET_EXCUTE_DIR, "output"),
                                               self.host_output_dir)
        if code != 0:
            raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_PULL_RESULTS_FAILED"))

        code, out, err = self.target_device.remove(target_path=self._TARGET_EXCUTE_DIR)
        if code != 0:
            err_msg = str(err) if err else str(out)
            raise InferenceEngineError(
                get_message('ERROR_INFERENCE_ENGINE_REMOVE_RESULTS_FAILED')(err_msg))

        self.logger.info('Removed inference results from device successfully')

    def get_binaries_to_push(self):
        """Get SNPE binaries used to convert and run a DLC.

        :return: List of binary paths
        """
        if self.runtime == "dspv69":
            htp_runtime = "V69"
            htp_runtime_lower = "v69"
        elif self.runtime == "dspv68":
            htp_runtime = "V68"
            htp_runtime_lower = "v68"
        elif self.runtime == "dspv73":
            htp_runtime = "V73"
            htp_runtime_lower = "v73"
        elif self.runtime == "dspv75":
            htp_runtime = "V75"
            htp_runtime_lower = "v75"
        elif self.runtime == "dspv79":
            htp_runtime = "V79"
            htp_runtime_lower = "v79"
        elif self.runtime.lower() in ['cpu', 'gpu']:
            htp_runtime = self.runtime
            htp_runtime_lower = self.runtime.lower()
        else:
            raise InferenceEngineError(
                get_message("{} runtime is not supported.".format(self.runtime))
            )

        binaries_to_push = []
        runtime = 'dsp' if 'dsp' in self.runtime else self.runtime
        for src_path in self.backend_paths[runtime]:
            binaries_to_push.append(
                (src_path.format(
                    engine_path=self.engine_path, target_arch='aarch64-windows-msvc'
                    if self.target_arch in ["wos-remote", "wos"] else self.target_arch,
                    htp_runtime=htp_runtime, htp_runtime_lower=htp_runtime_lower),
                 os.path.join(self.executor.target_path,
                              os.path.basename(src_path).format(htp_runtime=htp_runtime))))

        return binaries_to_push

    # -------------------------------------------------- COARSE GRAINED ----------------------------------------------------

    def _push_libs(self):
        """Push binaries and base dlc to target device."""
        # Push binaries to target device
        binary_paths = self.get_binaries_to_push()
        self.logger.info('Pushing binaries')
        for source, target in binary_paths:
            code, _, err = self.target_device.push(source, target)
            if code != 0:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_BINARIES_FAILED_DEVICE"))

        # Push base dlc to target device
        try:
            self.logger.info('Pushing base dlc to target')
            code, _, err = self.target_device.push(self.base_dlc_data.host_dlc_path,
                                                   self.executor.target_path)
            self.base_dlc_data.target_dlc_path = os.path.join(
                self.executor.target_path, os.path.basename(self.base_dlc_data.host_dlc_path))
            if code != 0:
                raise InferenceEngineError(get_message("ERROR_INFERENCE_ENGINE_DLC_FAILED_DEVICE"))

        except subprocess.CalledProcessError as exc:
            self.logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_DLC_FAILED_DEVICE'))

    def _push_input_data(self, target_device):
        """Push user specified input data to target device.

        if target and host device are same then does not push, just need to set the correct path
        """
        target_model_inputs = []
        target_is_host = target_device in ['x86', 'x86_64-windows-msvc', 'wos']
        # Push input data to target device
        self.logger.info('Pushing input data to target')
        if self.model_inputs is not None:
            per_batch_inputs = []
            for input_name, dim, data_path in self.model_inputs:

                if target_is_host:
                    per_batch_inputs.append(input_name + ":=" + data_path)
                else:
                    device_model_input_path = os.path.join(self.executor.target_path,
                                                           os.path.basename(data_path))
                    code, _, err = self.target_device.push(data_path, device_model_input_path)
                    if code != 0:
                        raise InferenceEngineError(
                            get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
                    per_batch_inputs.append(input_name + ":=" + device_model_input_path)
            target_model_inputs = [per_batch_inputs]
        elif self.input_list:
            target_model_inputs = []
            # get from input_list.
            # assumes the path is already full correct
            for batch in self._full_path_input_paths:
                per_batch_inputs = []
                for inputs in batch:
                    if target_is_host:
                        device_model_input_path = inputs
                    else:
                        if ":=" in inputs:
                            splited_inputs = inputs.split(":=")
                            tensor = splited_inputs[0]
                            data_path = splited_inputs[1]
                        else:
                            data_path = inputs

                        device_model_input_path = os.path.join(self.executor.target_path,
                                                               os.path.basename(data_path))
                        code, _, err = self.target_device.push(data_path, device_model_input_path)
                        if code != 0:
                            raise InferenceEngineError(
                                get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
                        # Update the path after push to become tensor specific path
                        if ":=" in inputs:
                            device_model_input_path = tensor + ":=" + os.path.join(
                                self.executor.target_path, os.path.basename(data_path))

                    per_batch_inputs.append(device_model_input_path)
                target_model_inputs.append(per_batch_inputs)
        return target_model_inputs

    def _sanitize_result_folder(self):
        """
        Iteratively go through all result folders in output directory and sanitize
        output raw file names.
        """
        dlc_tensor_names = get_tensor_names_from_dlc(self.base_dlc_data.host_dlc_path, sanitize_names=False)

        output_folder = os.path.join(self.host_output_dir, "output")
        for item in os.listdir(output_folder):
            item_full_path = os.path.join(output_folder, item)
            if os.path.isdir(item_full_path) and item.startswith("Result_"):
                status = sanitize_output_tensor_files(item_full_path, dlc_tensor_names)
                if status != 0:
                    raise InferenceEngineError(
                        "Failed to sanitize output folder {}".format(item_full_path))

    def _coarse_grained_run(self):
        """Convert and run in coarse grained mode.

        Creates an input list for every layer in the base dlc that uses,
        each using user specified model inputs for the input and the
        layer output for the output, then pushes the file to target,
        runs inference, and pulls the result.
        """
        # if target device is same as host, no need to push
        if self.target_device.device in ["x86", "x86_64-windows-msvc", "wos"]:
            self.base_dlc_data.target_dlc_path = self.base_dlc_data.host_dlc_path
        else:
            # Push binaries and base dlc
            self._push_libs()
        target_model_inputs = self._push_input_data(self.target_device.device)
        # Target model inputs are same for every inference in coarse grained mode

        # Set inputs to the model on the target device
        self.base_dlc_data.target_model_inputs = target_model_inputs

        # Create dlc data objects and add to the engine's list of dlc data
        def dlcdata_inference(layer_row):
            # always use the base dlc and base inputs in coarse grained mode
            dlc_data = self.DLCData(output_layers=[layer_row.name],
                                    host_dlc_path=self.base_dlc_data.host_dlc_path,
                                    host_model_inputs=self.base_dlc_data.host_model_inputs,
                                    target_model_inputs=self.base_dlc_data.target_model_inputs)
            dlc_data.target_dlc_path = self.base_dlc_data.target_dlc_path
            self.sub_dlc_data_list.append(dlc_data)

            # Create input list
            SNPEInferenceEngine._write_input_list(dlc_data, self._INPUT_LIST_DIR)

            # Push input list to target and set dlc_data's target input list
            self.logger.info('Pushing input list to target: ' + dlc_data.host_input_list_path)
            input_list_file_dest = os.path.join(self._TARGET_INPUT_LIST_DIR,
                                                os.path.basename(dlc_data.host_input_list_path))
            dlc_data.target_input_list_path = input_list_file_dest
            if not self.target_device.device in ["x86", "x86_64-windows-msvc", "wos"]:
                try:
                    code, _, err = self.target_device.push(dlc_data.host_input_list_path,
                                                           input_list_file_dest)
                    if code != 0:
                        raise InferenceEngineError(
                            get_message("ERROR_INFERENCE_ENGINE_TARGET_PUSH_FAILED")(err))
                    dlc_data.target_input_list_path = input_list_file_dest
                except subprocess.CalledProcessError as exc:
                    self.logger.error(
                        'Error pushing input list to target. Continuing with inference on next layer.'
                    )
                    self.logger.error(str(exc))
                    return

            # Execute inference
            try:
                self._execute_inference(dlc_data)
            except InferenceEngineError as exc:
                self.logger.error(
                    'Error executing inference. Continuing with inference on next layer.')
                self.logger.error(str(exc))
                return

        if not self.offline_prepare and self.debug_mode:
            for layer_row in self.base_dlc_info:
                dlcdata_inference(layer_row)
        else:
            dlcdata_inference(self.base_dlc_info[-1])

        if not self.target_device.device in ["x86", "x86_64-windows-msvc", "wos"]:
            # Pull results
            self._pull_inference_results()
        self._sanitize_result_folder()

    # -------------------------------------------------- FINE GRAINED ------------------------------------------------------

    def _fine_grained_run(self):
        raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_FINE_GRAINED_NOT_SUPPORTED'))

    # ----------------------------------------------------------------------------------------------------------------------

    def run(self):
        self.logger.info(f"Running on {self.runtime} runtime with {self.precision} precision")
        self._common_setup()
        if self.fine_grained_mode:
            self._fine_grained_run()
        else:
            self._coarse_grained_run()

    def get_graph_structure(self):
        if self.base_dlc_info is None:
            self.logger.info('Converting model to obtain graph structure.')
            self._set_base_dlc()
        graph_list_structure = [(layer.name, [
            layer.type,
            dict(
                zip([santize_node_name(input_name)
                     for input_name in layer.input_names], layer.input_dims)),
            dict(
                zip([santize_node_name(output_name)
                     for output_name in layer.output_names], layer.output_dims_list))
        ]) for layer in self.base_dlc_info]
        return OrderedDict(graph_list_structure)
