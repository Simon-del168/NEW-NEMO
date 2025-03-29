# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Annotated, Optional
from pydantic import Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from qti.aisw.tools.core.modules.api.definitions.common import AISWBaseModel
from qti.aisw.tools.core.modules.converter import (InputTensorConfig, OutputTensorConfig,
                                                   ConverterInputConfig, QuantizerInputConfig,
                                                   BackendInfoConfig)
from qti.aisw.tools.core.modules.net_runner.net_runner_module import InferenceConfig

from .enums import BackendType, DevicePlatformType


class ConversionParams(ConverterInputConfig):
    """
    Pydantic class for parameters related to source graph to use for model loading.

    Parameters:
        input_tensors (list[:class:`InputTensorConfig`]): List of :class:`InputTensorConfig` for details of input tensors of the model.
        output_tensors (list[:class:`OutputTensorConfig`]): List of :class:`OutputTensorConfig` for details of output tensors of the model. Required to Tensorflow models.
        float_bitwidth (Literal[32, 16]): Convert the graph to specified float bitwidth, either 32 or 16 (default: 32)
        float_bias_bitwidth (Literal[32, 16]): Option to select the bitwidth to use for float bias tensor.
        quantization_overrides (str): Path to JSON file with parameters to use for quantization.
                                      These will override any quantization data carried from
                                      conversion (e.g., TF fake quantization) or calculated during
                                      normal quantization process. Format defined as per AIMET specification.
        copyright_file (str): Path to copyright file whose contents are to be added to the output model.
        model_version (str): ASCII string to identify the model. only first 64 bytes will be stored.
        onnx_simplification (bool): Perform model simplification for ONNX models. (default: True)
        onnx_batch (int): The batch dimension override. This will take the first dimension of all
                          inputs and treat it as a batch dim, overriding it with the given value.
        onnx_define_symbol (list[tuple[str, int]]): Set given input dimension symbols to given value.
        enable_framework_trace (bool): Enable tracing of output tensor change information for ONNX models. (default: False)
        tf_show_unconsumed_nodes (bool): Display a list of unconsumed nodes, if there any are found.
        tf_saved_model_tag (str): Tag to select a MetaGraph from savedmodel.
        tf_saved_model_signature_key (str): Signature key to select input and output of the model.
        tf_no_optimization (bool): Do not attempt to optimize the model automatically. (default: False)
        tf_validate_models (bool): Validate the original TF model against optimized TF model. (default: False)
        tflite_signature_name (str): Specify specific Subgraph signature name to convert.

        # Custom op related arguments
        op_package_config (str | list[str]): Absolute paths to a Qnn Op Package XML configuration
                                             file that contains user defined custom operations.
        converter_op_package_lib (str | list[str]): Absolute paths to converter op package library
                                                    compiled by the OpPackage generator.
        package_name (str): A global package name to be used for each node in the model file.
                            Defaults to Qnn header defined package name.
    """
    _model_framework: Annotated[str, Field(exclude=True, init=False)]
    input_network: SkipJsonSchema[str] = Field(default='', init=False, exclude=True)
    dry_run: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    output_path: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)

    @field_validator("input_network")
    @classmethod
    def validate_framework(cls, v):
        pass

    @model_validator(mode="after")
    def validate_input_arguments(self):
        pass


class QuantizationParams(QuantizerInputConfig):
    """Pydantic class of parameters for model quantization

    Parameters:
        input_list (str): Path to a file containing input files to be used for quantization.
        float_fallback (bool): Option to enable fallback to floating point (FP) instead of fixed point. (default: False)
        algorithms (str): Desired optimization algorithm.
        bias_bitwidth (int): Bitwidth for quantizing the biases, either 8 or 32. (default: 8)
        act_bitwidth (int): Bitwidth for quantizing the activations, either 8 or 16. (default: 8)
        weights_bitwidth (int): Bitwidth for quantizing the weights, either 8 or 4. (default: 8)
        float_bitwidth (int): Convert the graph to specified float bitwidth, either 32 or 16
        float_bias_bitwidth (int): Bitwidth to use for float bias tensor, either 32 or 16
        keep_weights_quantized (bool): Keep the weights quantized even when the output of the operator is in floating point. (default: False)
        use_aimet_quantizer (bool): Use AIMET for Quantization instead of QNN IR quantizer. (default: False)
        config_file (str): Configuration YAML file with quantizer options for AIMET quantizer. **Required** for AIMET quantizer.
        ignore_encodings (bool): Ignore encodings from model or user. Use only quantizer generated encodings. (default: False)
        use_per_channel_quantization (bool): Enable per-channel quantization for convolution-based op weights. (default: False)
        use_per_row_quantization (bool): Enable row-wise quantization of MatMul and FullyConnected ops. (default: False)
        use_native_input_files (bool): Boolean flag to indicate how to read input files.
                                       False (default): Reads inputs as floats and quantizes, if necessary based on quantization parameters in the model.
                                       True: Reads inputs assuming the data type to be native to the model.
        use_native_output_files (bool): Boolean flag to indicate the data type of the output files.
                                        False(default): Output the file as float datatype.
                                        True: Outputs the file that is native to the model.
        restrict_quantization_steps (list[str]): Specify the number of steps to use for computing quantization encodings
        act_quantizer_calibration (str): Specify quantization calibration method to use for activations.
                                         Options: min-max (default), sqnr, entropy, mse, percentile.
        param_quantizer_calibration (str): Specify quantization calibration method to use for parameters.
                                           Options: min-max (default), sqnr, entropy, mse, percentile.

        act_quantizer_schema (str): Specify quantization schema to use for activations.
                                           Options: asymmetric (default), symmetric.
        param_quantizer_schema (str): Specify quantization schema to use for parameters.
                                           Options: asymmetric (default), symmetric.
        percentile_calibration_value (float): Percentile value, between 90 and 100, to be used with Percentile calibration method. (default: 99.99)
        op_package_lib (list[str]): List of op package library for quantization.

    """
    input_dlc: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    output_dlc: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    backend_info: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)


class CompilationParams(AISWBaseModel):
    """Pydantic class of parameters for model compilation

    Parameters:
        conversion_params (ConversionParams): Conversion params object
        quantization_params (QuantizationParams): Quantization params object
        backend (BackendType): Desired backend. Providing this option will generate a graph optimized for the given backend.
        soc_model (str): Desired SOC model.
        backend_extensions (dict): Dictionary of backend extensions to apply during offline preparation
                                   (only applicable for certain backends).
        offline_prepare (bool): Enable offline preparation.
        """
    conversion_params: Optional[ConversionParams] = None
    quantization_params: Optional[QuantizationParams] = None
    backend: Optional[BackendType] = None
    soc_model: str = ''
    backend_extensions: Optional[dict] = None
    offline_prepare: Optional[bool] = None

    @model_validator(mode='after')
    def validate(self):
        if self.offline_prepare:
            if self.backend is None:
                raise Exception('Backend parameter must be set when enabling offline_prepare')
            offline_prep_backends = BackendType.offline_preparable_backends()
            if self.backend not in offline_prep_backends:
                raise ValueError(
                    'Offline preparation is not supported for {} backend. Valid backends={}'.format(
                        self.backend, list(map(str, offline_prep_backends))))

        if self.quantization_params and self.backend:
            quant_backends = BackendType.quantizable_backends()
            if self.backend not in quant_backends:
                raise ValueError(
                    'Quantization is not supported for {} backend. Valid backends={}'.format(
                        self.backend, list(map(str, quant_backends))))

        if self.soc_model and not self.backend:
            raise Exception('backend is mandatory when setting soc_model parameter')


class ExecutionParams(InferenceConfig):
    """Pydantic class of parameters for model execution

    Parameters:
        profiling_level (str): Enable profiling with given level. Options: basic, detailed, backend
        batch_multiplier (int): Specifies the value with which the batch value in input and output
                                tensors dimensions will be multiplied. The modified tensors will be
                                used only during the execute graphs. Composed graphs will still use
                                the tensor dimensions from model.
        use_native_input_data (bool): Specifies that the input files will be parsed in the data
                                      type native to the graph. If False (default), input files will
                                      be parsed in floating point.
        use_native_output_data (bool): Specifies that the output files will be generated in the data
                                       type native to the graph. If False (default), output files will
                                       be generated in floating point.
        native_input_tensor_names (list[str]): Provide a comma-separated list of input tensor names,
                                               for which the input files would be read/parsed in native format.
                                               Note that options use_native_input_files and native_input_tensor_names
                                               are mutually exclusive. Only one of the options can be specified at a time.
        backend_extensions (dict): Dictionary of desired backend extensions.
    """
    log_level: SkipJsonSchema[str] = Field(default='', init=False, exclude=True)
    backend_extensions: Optional[dict] = None


# TODO: Switch to Target class of Modules once all platforms and connection types are supported
class Target(AISWBaseModel):
    """
    Type of device to be used for execution, optionally including device identifiers (only for Android platforms currently).

    Args:
        type (DevicePlatformType): The type of device platform to be used
        identifier(str, optional): Serial ID of Android device
    """
    type: DevicePlatformType
    identifier: str = None

    @model_validator(mode='after')
    def check_identifier(self):
        if self.identifier and self.type is not DevicePlatformType.ANDROID:
            raise ValueError('Identifier should be provided only for Android platforms')
