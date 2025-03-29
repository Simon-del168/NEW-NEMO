# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import sys
from pathlib import Path
from typing import Literal, Any, Dict, Tuple

if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated

from pydantic import model_validator, Field, FilePath, field_validator
from pydantic.functional_validators import AfterValidator

# Module Imports
from qti.aisw.tools.core.modules.api import ModuleSchema, Module, expect_module_compliance, \
    AISWBaseModel, ModuleSchemaVersion
from qti.aisw.tools.core.modules.converter.utils import *
from qti.aisw.tools.core.modules.converter.constants import *

# Converter Imports
from qti.aisw.converters.common.converter_ir.op_graph import InputLayout, InputEncodings
from qti.aisw.converters.common.utils import converter_utils


def check_layout(layout_str: str) -> str:
    if not InputLayout.is_valid_type(layout_str):
        raise AssertionError(f"Input layout: {layout_str} is not valid")
    return layout_str


def check_encoding(encoding_str: str) -> str:
    if not InputEncodings.is_valid_encoding(encoding_str):
        raise AssertionError(f"Input Encoding: {encoding_str} is not valid")
    return encoding_str


# Define string types for encoding and layouts
InputLayoutStr = Annotated[str, AfterValidator(check_layout)]
InputEncodingStr = Annotated[str, AfterValidator(check_encoding)]


class InputTensorConfig(AISWBaseModel):
    name: str = Field(description="Name of tensor.")
    source_model_input_datatype: str = Field(default="float32",
                                             description="Data type of input tensor.")
    source_model_input_layout: Optional[InputLayoutStr] = Field(default=None,
                                                                description="Layout of each "
                                                                            "input tensor.")
    desired_input_shape: Union[str, List[str]] = Field(default="",
                                                       description="Shape of input tensor.")
    desired_input_color_encoding: Optional[InputEncodingStr] = (
        Field(default="bgr", description="Input encoding of the network inputs. Default is bgr."))


class OutputTensorConfig(AISWBaseModel):
    name: str


class ConverterInputConfig(AISWBaseModel):
    _model_framework: str
    input_network: FilePath = Field(description="Path to the source framework model.")
    input_tensors: Optional[List[InputTensorConfig]] = Field(default=None,
                                                             description="Input tensors config.")
    output_tensors: Optional[List[OutputTensorConfig]] = Field(default=None,
                                                               description="Output tensors config.")
    float_bitwidth: Optional[Literal[32, 16]] = Field(default=32,
                                                      description="Convert the graph to specified "
                                                                  "float bitwidth.")
    float_bias_bitwidth: Optional[Literal[16, 32]] = Field(default=None,
                                                           description="Option to select the "
                                                                       "bit-width to use for float "
                                                                       "bias tensor.")
    quantization_overrides: Optional[FilePath] = Field(default=None,
                                                       description="Option to specify a json file"
                                                                   " with parameters to use for "
                                                                   "quantization.")
    dry_run: bool = Field(default=False,
                          description="Evaluates the model without actually converting any ops"
                                      ", and return unsupported ops/attributes as well as"
                                      " unused inputs and/or outputs if any.")
    copyright_file: Optional[FilePath] = Field(default=None,
                                               description="Path to copyright file. If provided, "
                                                           "the content of the file will be added "
                                                           "to the output model")
    model_version: str = Field(default="",
                               description="User-defined ASCII string to identify the model, "
                                           "only first 64 bytes will be stored.")
    onnx_simplification: bool = Field(default=True,
                                      description="Do not attempt to simplify the model "
                                                  "automatically.This may prevent some models from "
                                                  "converting when sequences of unsupported static "
                                                  "operations are present.")
    onnx_batch: Optional[int] = Field(default=None,
                                      description="The batch dimension override.This will take the "
                                                  "first dimension of all inputs and treat it as "
                                                  "a batch dim, overriding it with the value "
                                                  "provided here.")
    onnx_define_symbol: Optional[List[Tuple[str, int]]] = (
        Field(default=None, description="Option to override specific input dimension symbols."))
    tf_show_unconsumed_nodes: bool = Field(default=False,
                                           description="Displays a list of unconsumed nodes, if "
                                                       "there any are found.")
    tf_saved_model_tag: Optional[str] = Field(default="serve",
                                              description="Specify the tag to select a MetaGraph "
                                                          "from saved model.")
    tf_saved_model_signature_key: str = Field(default="serving_default",
                                              description="Specify signature key to select input "
                                                          "and output of the model.")
    tf_no_optimization: bool = Field(default=False,
                                     description="Do not attempt to optimize the model "
                                                 "automatically.")
    tf_validate_models: bool = Field(default=False,
                                     description="Validate the original TF model against optimized "
                                                 "TF model.")
    tflite_signature_name: str = Field(default="",
                                       description="Use this option to specify a specific Subgraph "
                                                   "signature to convert.")
    enable_framework_trace: bool = Field(default=False,
                                         description="Use this option to enable converter to trace "
                                                     "the o/p tensor change"
                                                     " information.")
    output_path: str = Field(default="",
                             description="Path where the converted output model should be saved.If"
                                         " not specified, the converter model will be written to a "
                                         "file with same name as the input model")
    # Custom op related arguments
    op_package_config: Optional[List[FilePath]] = (
        Field(default=None,
              description="List of absolute paths to a Qnn Op Package XML configuration file that "
                          "contains user defined custom operations."))
    converter_op_package_lib: Optional[List[FilePath]] = (
        Field(default=None,
              description="List of absolute paths to converter op package library compiled by "
                          "the OpPackage generator."))
    package_name: str = Field(default="",
                              description="A global package name to be used for each node in "
                                          "the Model.cpp file. Defaults to Qnn header defined "
                                          "package name.")

    @field_validator("input_network")
    @classmethod
    def validate_framework(cls, v):
        cls._model_framework = infer_framework(input_network=v)
        return v

    @model_validator(mode="after")
    def validate_input_arguments(self):
        if (self._model_framework == TensorflowFrameworkInfo.name
                or self._model_framework == PytorchFrameworkInfo.name):
            if not self.input_tensors:
                raise ValueError("Input tensor name(s) and dimension(s) details "
                                 "required for '{}' model."
                                 .format(self._model_framework))
            for input_tensor in self.input_tensors:
                if not input_tensor.desired_input_shape:
                    raise ValueError("Dimension is needed for input tensor {}."
                                     .format(input_tensor.name))
        if self._model_framework == TensorflowFrameworkInfo.name:
            if not self.output_tensors:
                raise ValueError("Output tensor name(s) and dimension(s) details"
                                 " required for '{}' model."
                                 .format(self._model_framework))

        # Only one of the 'package_name', 'op_package_config' can be specified
        if self.op_package_config and self.package_name:
            raise ValueError("'package_name' and 'op_package_config' are "
                             "mutually exclusive options."
                             "Only one of them can be specified.")

        # Check if quantization override file is loadable.
        if self.quantization_overrides:
            import json
            try:
                with open(self.quantization_overrides) as override_file:
                    _ = json.load(override_file)
            except Exception as e:
                self._logger.error("Failed to load quantization override file "
                                   "with error: {}".format(str(e)))
                raise
        return self

    def get_model_framework(self):
        return self._model_framework


class ConverterOutputConfig(AISWBaseModel):
    ir_graph: Any
    framework: str


class ConverterModuleSchemaV1(ModuleSchema):
    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)
    _BACKENDS = None
    name: Literal["ConverterModule"] = "ConverterModule"
    path: Path = Path(__file__)
    arguments: ConverterInputConfig
    outputs: Optional[ConverterOutputConfig] = None
    backends: Optional[List[str]] = _BACKENDS
    version: ModuleSchemaVersion = _VERSION


@expect_module_compliance
class QAIRTConverter(Module):
    """
    User interface class for model conversion API.
    """

    _SCHEMA = ConverterModuleSchemaV1
    _PREVIOUS_SCHEMAS = []

    def __init__(self, logger: Any = None) -> None:
        """
        Args:
            logger:
        """
        if not logger:
            logger = logging.getLogger("ConverterLogger")
        converter_utils.LOGGER = logger
        super().__init__(logger)
        self._debug_level = LOGLEVEL.INFO

    @staticmethod
    def _get_args(config: ConverterInputConfig) -> Namespace:
        """
        This method accepts converter input config and return arguments in namespace object.
        1. Converts arguments names to converter internal names.
        2. Sets default values for suppressed arguments.
        3. Converts v1 args to v2. See Converter API for more details.
        Args:
            config: Converter input arguments config.

        Returns:
            Return namespace object containing arguments.
        """
        option_dict = config.model_dump()
        option_dict["input_network"] = str(option_dict["input_network"])

        option_dict["custom_op_config_paths"] = option_dict.pop("op_package_config")
        if option_dict["custom_op_config_paths"]:
            option_dict["custom_op_config_paths"] = [str(item) for item in
                                                     option_dict["custom_op_config_paths"]]

        if option_dict["converter_op_package_lib"]:
            option_dict["converter_op_package_lib"] = ",".join([str(item) for item in
                                                                option_dict[
                                                                    "converter_op_package_lib"]])
        else:
            option_dict["converter_op_package_lib"] = ""

        # Unpack input tensors
        input_tensors = option_dict.pop("input_tensors")
        input_dim = []
        input_layout = []
        input_dtype = []
        input_encoding = []

        if input_tensors:
            for input_tensor in input_tensors:
                if input_tensor["desired_input_shape"]:
                    input_dim.append([input_tensor["name"],
                                      input_tensor["desired_input_shape"]])
                if input_tensor["source_model_input_datatype"]:
                    input_dtype.append([input_tensor["name"],
                                        input_tensor["source_model_input_datatype"]])
                if input_tensor["source_model_input_layout"]:
                    input_layout.append([input_tensor["name"],
                                         input_tensor["source_model_input_layout"]])
                if input_tensor["desired_input_color_encoding"]:
                    input_encoding.append([input_tensor["name"],
                                           input_tensor["desired_input_color_encoding"]])

        option_dict["input_dim"] = input_dim
        option_dict["input_dtype"] = input_dtype
        option_dict["input_layout"] = input_layout
        option_dict["input_encoding"] = input_encoding

        # unpack output tensors.
        output_tensors = option_dict.pop("output_tensors")
        out_names = []
        if output_tensors:
            for output_tensor in output_tensors:
                out_names.append(output_tensor["name"])
        option_dict["out_names"] = out_names

        if option_dict["float_bias_bitwidth"] is None:
            option_dict["float_bias_bitwidth"] = 0

        # Map user input arguments name to converter internal names.
        option_dict["no_simplification"] = not option_dict.pop("onnx_simplification")
        option_dict["batch"] = option_dict.pop("onnx_batch")
        option_dict["validate_models"] = option_dict.pop("tf_validate_models")

        option_dict["define_symbol"] = option_dict.pop("onnx_define_symbol")
        if option_dict["define_symbol"]:
            option_dict["define_symbol"] = ["{} {}".format(item[0], item[1])
                                            for item in option_dict.pop("define_symbol")]

        option_dict["no_optimization"] = option_dict.pop("tf_no_optimization")
        option_dict["show_unconsumed_nodes"] = option_dict.pop("tf_show_unconsumed_nodes")
        option_dict["saved_model_tag"] = option_dict.pop("tf_saved_model_tag")
        option_dict["saved_model_signature_key"] = option_dict.pop("tf_saved_model_signature_key")
        option_dict["signature_name"] = option_dict.pop("tflite_signature_name")

        # TODO: expose input config option when network specialization support is added.
        option_dict["io_config"] = ""

        # Suppressed arguments
        option_dict["dump_inferred_model"] = False
        option_dict["dump_value_info"] = False
        option_dict["disable_preserve_io"] = False
        option_dict["keep_int64_inputs"] = False
        option_dict["keep_quant_nodes"] = False
        option_dict["use_onnx_relay"] = False
        option_dict["disable_match_lstms"] = False
        option_dict["expand_lstm_op_structure"] = False
        option_dict["pytorch_custom_op_lib"] = ""
        option_dict["dump_relay"] = ""

        args = Namespace(**option_dict)
        args = convert_args_v2_to_v1(args)

        return args

    def convert(self, config: ConverterInputConfig) -> ConverterOutputConfig:
        """
        This method accepts a converter input config containing a framework model and produces a
        converter output config containing an IRGraph.

        Args:
            config: "ConverterInputConfig" object containing converter module input arguments.

        Returns:
            "ConverterOutputConfig" object containing IRgraph and framework.

        Examples:
            >>> input_network = "path/to/model"
            >>> converter_module  = QAIRTConverter()
            >>> out_config = converter_module.convert(ConverterInputConfig(input_network=input_network))
        """

        args = self._get_args(config)

        # Add log level to args so that internal libs use same log level as API.
        args.debug = self._debug_level

        try:
            num_graph_configs = get_num_graph_configs(args)
            if num_graph_configs != 1:
                raise NotImplementedError("Multi configuration is not supported in API version {}".
                                          format(self._schema.get_version()))

            framework = config.get_model_framework()
            converter = get_frontend_converter(framework, args)
            ir_graph = converter.convert()
            # TODO: Generate dlc here when AISW-91995: Enable serialization of IRgraph is enabled.
            output_config = ConverterOutputConfig(ir_graph=ir_graph,
                                                  framework=framework)

            # validate original model vs framework optimized model. See Converter
            # help for more details.
            self._validate_framework_optimizations(converter)

            return output_config

        except Exception as e:
            self._logger.error("Model conversion failed.")
            raise e

    def _validate_framework_optimizations(self, converter: ConverterFrontend) -> None:
        """
        Validates the original framework model against a model that has undergone
        frontend optimizations.

        #TODO: Need a better description of this from the converter team

        Args:
            converter: Frontend converter instance of

        Returns: None
        """
        # Validate output of original model with frontend optimized model
        if not hasattr(converter, "validator"):
            self._logger.warning(f"Validation is not supported for this frontend: "
                                 f"{converter.__name__}")
            return

        if validator := getattr(converter, "validator", None):
            try:
                results = validator.validate()
                for result in results:
                    self._logger.info(result)
            except Exception as e:
                self._logger.warning(
                    "Model conversion is completed but error "
                    "encountered during validation : {}".format(str(e)))

    def enable_debug(self, debug_level: int, **kwargs) -> Optional[bool]:
        """
        Sets converter log level.
        Args:
            debug_level: LOGLEVEL.VERBOSE enables VERBOSE and higher level messages.
               LOGLEVEL.DEBUG enables DEBUG and higher level messages.
               LOGLEVEL.DEBUG_3 enables DEBUG_3 and higher level messages.
               LOGLEVEL.DEBUG_2 enables DEBUG_2 and higher level messages.
               LOGLEVEL.DEBUG_1 enables DEBUG_1 and higher level messages.
               LOGLEVEL.INFO enables INFO and higher level messages.
            **kwargs:

        Returns:
            bool: 'True' if debugging is enabled else return 'False'.
        """

        if debug_level < LOGLEVEL.INFO or debug_level > LOGLEVEL.VERBOSE:
            return False
        self._debug_level = debug_level
        converter_utils.setup_logging(self._debug_level)
        return True

    @property
    def _schema(self):
        return self._SCHEMA

    def get_logger(self) -> Any:
        return self._logger

    def properties(self) -> Dict[str, Any]:
        return self._schema.model_json_schema()
