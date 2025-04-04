# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import tempfile
import logging
from pathlib import Path
from argparse import Namespace
from typing import Literal, Any, Dict, Optional
from pydantic import Field

# Module Imports
from qti.aisw.tools.core.modules.api import Module, ModuleSchema, AISWBaseModel, \
    ModuleSchemaVersion, \
    expect_module_compliance
from qti.aisw.tools.core.modules.converter.common import BackendInfoConfig
from qti.aisw.tools.core.modules.converter.utils import get_framework_extension
from qti.aisw.tools.core.modules.converter.constants import *

# Converter Imports
from qti.aisw.converters.common.utils import converter_utils
from qti.aisw.converters.common.converter_ir.op_graph_optimizations import IROptimizations
from qti.aisw.converters.common.backend_awareness import BackendInfo
from qti.aisw.converters.qnn_backend.ir_to_dlc import DLCBackend


class OptimizerInputConfig(AISWBaseModel):
    # TODO: Accept DLC as input once, DLC serialization enabled for IRgraph. make Ir_graph as
    # optional.
    ir_graph: Any = Field(description="IRgraph generated by model conversion.")
    framework: str = Field(description="Source framework from which IRgraph generated.")
    output_dlc: str = Field(default=None,
                            description="Path to DLC container containing optimized model.")
    backend_info: Optional[BackendInfoConfig] = Field(default=None,
                                                      description="Backend information.")


class OptimizerOutputConfig(AISWBaseModel):
    dlc_path: str = Field(description="Path to output DLC file ")


class OptimizerModuleSchemaV1(ModuleSchema):
    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)
    _BACKENDS = None
    name: Literal["OptimizerModule"] = "OptimizerModule"
    path: Path = Path(__file__)
    arguments: OptimizerInputConfig
    outputs: Optional[OptimizerOutputConfig] = None
    backends: Optional[List[str]] = _BACKENDS
    version: ModuleSchemaVersion = _VERSION


@expect_module_compliance
class QAIRTOptimizer(Module):
    """
    User interface class for optimizer API
    """
    _SCHEMA = OptimizerModuleSchemaV1
    _PREVIOUS_SCHEMAS = []

    def __init__(self, logger=None):
        if not logger:
            logger = logging.getLogger("OptimizerLogger")
        converter_utils.LOGGER = logger
        super().__init__(logger)
        self._debug_level = LOGLEVEL.INFO

    def optimize(self, config: OptimizerInputConfig) -> OptimizerOutputConfig:
        """
        Performs optimizations on the IRGraph contained in the config. Optimizations
        are intended to increase performance while maintaining mathematical equivalence.

        Args:
            config (OptimizerInputConfig): contains optimizer module input arguments

        Returns:
            config (OptimizerOutputConfig): contains optimizer module output arguments

        Examples:
            >>> from qti.aisw.tools.core.modules.converter import QAIRTConverter, \
            >>> ConverterInputConfig
            >>> converter_config = QAIRTConverter().convert(ConverterInputConfig("/path/to/model"))
            >>> ir_graph = converter_config.ir_graph
            >>> framework = converter_config.framework
            >>> optimizer = QAIRTOptimizer()
            >>> out_config = optimizer.optimize(OptimizerInputConfig(ir_graph=ir_graph,
            >>>                                                      framework=framework))
        """

        # transform args to converter namespace
        args = self._get_args(config)

        try:
            backend_info_obj = BackendInfo.get_instance(args.backend, args.soc_model)
            optimizer = IROptimizations(args)
            optimized_graph = optimizer.optimize(config.ir_graph, backend_info_obj)

            # TODO: May be better to make this optional for when we need to debug
            # serialization issues
            backend = DLCBackend(args)
            backend.save(optimized_graph)

        except Exception as e:
            self._logger.error("IRgraph optimization failed.")
            raise e
        return OptimizerOutputConfig(dlc_path=args.output_path)

    def enable_debug(self, debug_level: int, **kwargs) -> Optional[bool]:
        """
        Sets optimizer log level.
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

    def _get_args(self, config: OptimizerInputConfig):
        """
        This method accepts converter input config and return arguments in namespace object.
        1. Converts arguments names to optimizer internal names.
        2. Sets default values for suppressed arguments.
        Args:
            config: Optimizer input arguments config.

        Returns:
            Return namespace containing arguments.
        """

        option_dict = config.model_dump()
        _ = option_dict.pop("ir_graph")
        framework = option_dict.pop("framework")
        output_dlc = option_dict.pop("output_dlc")
        # Unpack backend options

        backend_info = option_dict.pop("backend_info")
        if backend_info:
            option_dict["backend"] = backend_info["backend"]
            option_dict["soc_model"] = backend_info["soc_model"]
        else:
            option_dict["backend"] = None
            option_dict["soc_model"] = None

        args = Namespace(**option_dict)
        args.dumpIR = False
        args.disable_batchnorm_folding = False
        args.disable_match_lstms = False
        args.squash_box_decoder = False
        args.match_caffe_ssd_to_tf = False
        args.adjust_nms_features_dims = False
        args.extract_color_transform = False
        args.perform_axes_to_spatial_first_order = False
        args.preprocess_roi_pool_inputs = False
        args.multi_time_steps_lstm = False
        args.unroll_lstm_time_steps = False
        args.expand_lstm_op_structure = False
        args.multi_time_steps_gru = False
        args.unroll_gru_time_steps = False
        args.expand_gru_op_structure = False
        args.force_prune_cast_ops = False
        args.inject_cast_for_gather = False
        args.use_convert_quantization_nodes = False
        args.align_matmul_ranks = False
        args.prepare_inputs_as_params = False
        args.handle_gather_negative_indices = False
        args.enable_match_gathernd = False
        args.expand_sparse_op_structure = False
        args.keep_disconnected_nodes = False
        args.align_matmul_ranks = False
        args.prepare_inputs_as_params = False
        args.handle_gather_negative_indices = False
        args.enable_match_gathernd = False
        args.expand_sparse_op_structure = False
        args.apply_masked_softmax = False
        args.packed_masked_softmax_inputs = "uncompressed"
        args.packed_max_seq = 1
        args.op_package_lib = None
        args.keep_int64_inputs = False
        args.copyright_file = None
        args.quantization_overrides = None
        args.float_bitwidth = None
        args.float_bias_bitwidth = None
        args.model_version = None
        args.float_bitwidth = 32
        args.perform_layout_transformation = False

        # Add log level to args to that internal libs use same log level as API.
        args.debug = self._debug_level

        if not output_dlc:
            # TODO: Once optimize API accepts dlc as input, store output file in same location as
            # input dlc.
            output_folder = tempfile.gettempdir()
            args.output_path = os.path.join(output_folder, "optimized_model.dlc")
        else:
            args.output_path = output_dlc
            output_folder, file_name = os.path.split(output_dlc)

        ext = get_framework_extension(framework)

        # TODO: Clean up once dependency on input network is removed.
        args.input_network = os.path.join(output_folder, "input_network" + ext)

        # set the optimization args in place
        set_optimization_args(args, config.framework)

        return args


def set_optimization_args(args: Namespace, framework: str) -> None:
    """"

    """
    # TODO: Align optimizations for all frameworks
    if framework == OnnxFrameworkInfo.name:
        args.expand_gru_op_structure = True
        args.unroll_gru_time_steps = True
        args.expand_sparse_op_structure = True

    if framework == OnnxFrameworkInfo.name or framework == PytorchFrameworkInfo.name:
        args.perform_axes_to_spatial_first_order = True
        args.preprocess_roi_pool_inputs = True

    if framework == OnnxFrameworkInfo.name or framework == TensorflowFrameworkInfo.name:
        args.unroll_lstm_time_steps = True
        args.align_matmul_ranks = True
        args.handle_gather_negative_indices = True

    if framework == TensorflowFrameworkInfo.name or framework == PytorchFrameworkInfo.name:
        args.match_caffe_ssd_to_tf = True

    # Enable/Disable following optimizations for onnx, tf, pytorch
    if framework != TFLiteFrameworkInfo.name:
        args.squash_box_decoder = True
        args.adjust_nms_features_dims = True
        args.extract_color_transform = True
        args.inject_cast_for_gather = True
        args.force_prune_cast_ops = False
