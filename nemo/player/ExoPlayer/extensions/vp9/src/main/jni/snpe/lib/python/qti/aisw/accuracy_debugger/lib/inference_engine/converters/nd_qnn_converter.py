# =============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import abc

from qti.aisw.accuracy_debugger.lib.inference_engine.converters.nd_converter import Converter
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message


class QNNConverter(Converter):

    def __init__(self, context):
        super(QNNConverter, self).__init__(context)
        # Instantiate lib generator fields from context
        self.executable = context.executable
        self.model_path_flag = context.arguments["model_path_flag"]
        self.output_path_flag = context.arguments["output_path_flag"]

        quantization_flag = context.arguments["quantization_flag"]
        self.input_list_flag = quantization_flag["input_list_flag"]
        self.quantization_overrides_flag = quantization_flag["quantization_overrides_flag"]
        self.param_quantizer_flag = quantization_flag["param_quantizer_flag"]
        self.act_quantizer_flag = quantization_flag["act_quantizer_flag"]
        self.weight_bw_flag = quantization_flag["weight_bw_flag"]
        self.bias_bw_flag = quantization_flag["bias_bw_flag"]
        self.act_bw_flag = quantization_flag["act_bw_flag"]
        self.float_bias_bw_flag = quantization_flag["float_bias_bw_flag"]
        self.restrict_quantization_steps_flag = quantization_flag[
            "restrict_quantization_steps_flag"]
        self.algorithms_flag = quantization_flag["algorithms_flag"]
        self.ignore_encodings_flag = quantization_flag["ignore_encodings_flag"]
        self.use_per_channel_quantization_flag = quantization_flag[
            "use_per_channel_quantization_flag"]
        self.act_quantizer_calibration_flag = quantization_flag["act_quantizer_calibration_flag"]
        self.param_quantizer_calibration_flag = quantization_flag[
            "param_quantizer_calibration_flag"]
        self.act_quantizer_schema_flag = quantization_flag["act_quantizer_schema_flag"]
        self.param_quantizer_schema_flag = quantization_flag["param_quantizer_schema_flag"]
        self.percentile_calibration_value_flag = quantization_flag[
            "percentile_calibration_value_flag"]
        self.float_fallback_flag = quantization_flag["float_fallback_flag"]

    def quantization_command(self, input_list_txt, quantization_overrides, param_quantizer,
                             act_quantizer, weight_bw, bias_bw, act_bw, float_bias_bw,
                             restrict_quantization_steps, algorithms, ignore_encodings,
                             per_channel_quantization, act_quantizer_calibration,
                             param_quantizer_calibration, act_quantizer_schema,
                             param_quantizer_schema, percentile_calibration_value,
                             float_fallback):
        convert_command = []
        if quantization_overrides and ignore_encodings:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_QNN_QUANTIZATION_FLAG_INPUTS"))

        if input_list_txt:
            convert_command += [self.input_list_flag, input_list_txt]
            if param_quantizer:
                convert_command += [self.param_quantizer_flag, param_quantizer]
            if act_quantizer:
                convert_command += [self.act_quantizer_flag, act_quantizer]
            if act_quantizer_calibration:
                convert_command += [self.act_quantizer_calibration_flag, act_quantizer_calibration]
                convert_command += [self.act_quantizer_schema_flag, act_quantizer_schema]
            if param_quantizer_calibration:
                convert_command += [self.param_quantizer_calibration_flag, param_quantizer_calibration]
                convert_command += [self.param_quantizer_schema_flag, param_quantizer_schema]
            if param_quantizer_calibration == "percentile" or act_quantizer_calibration == "percentile":
                convert_command += [
                    self.percentile_calibration_value_flag, percentile_calibration_value
                ]
            if weight_bw:
                convert_command += [self.weight_bw_flag, str(weight_bw)]
            if bias_bw:
                convert_command += [self.bias_bw_flag, str(bias_bw)]
            if act_bw:
                convert_command += [self.act_bw_flag, str(act_bw)]
            if restrict_quantization_steps:
                convert_command += [self.restrict_quantization_steps_flag, restrict_quantization_steps]
            if algorithms:
                convert_command += [self.algorithms_flag, algorithms]
            if ignore_encodings:
                convert_command += [self.ignore_encodings_flag]
            if per_channel_quantization:
                convert_command += [self.use_per_channel_quantization_flag]
        elif float_fallback:
            convert_command += [self.float_fallback_flag]

        if quantization_overrides:
            convert_command += [self.quantization_overrides_flag, quantization_overrides]

        if float_bias_bw:
            convert_command += [self.float_bias_bw_flag, str(float_bias_bw)]

        # Enable --preserve_io layout always since debugger receives inputs layouts as per framework model
        convert_command += ['--preserve_io', 'layout']

        return convert_command

    @abc.abstractmethod
    def build_convert_command(self, model_path, input_tensors, output_tensors, output_path,
                              input_list_txt, quantization_overrides, param_quantizer,
                              act_quantizer, weight_bw, bias_bw, act_bw, float_bias_bw,
                              restrict_quantization_steps, algorithms, ignore_encodings,
                              per_channel_quantization):
        """Build command (using converter tools) to convert model to QNN Graph.

        model_path: Path to model file
        input_tensors: Names and dimensions of input tensors
        output_tensors: Names of output tensors for network
        output_path: Output directory for QNN .cpp and .bin

        return value: String command using converter tool (ie. tensorflow-to-qnn)
        that is used to generate QNN .cpp and .bin files
        """
        pass
