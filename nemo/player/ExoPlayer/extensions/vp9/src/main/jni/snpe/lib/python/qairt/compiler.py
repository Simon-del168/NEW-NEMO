# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
from pathlib import Path
import tempfile

from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.tools.core.modules.converter import converter_module, optimizer_module, quantizer_module
from qti.aisw.tools.core.modules.context_bin_gen import context_bin_gen_module
from qti.aisw.tools.core.modules.api.definitions.common import Model

from .params import CompilationParams, BackendInfoConfig
from .exceptions import ConversionFailure, OptimizationFailure, QuantizationFailure, GraphPreparationFailure
from .qairt_model import QairtModel

qairt_logger = logging.getLogger('qairt_api')


class Compiler():
    """
    Compiler class for performing model compilation
    """

    def __init__(self) -> None:
        pass

    def compile(self, model_file: str | Path, compilation_params: CompilationParams = None,
                out_dir: str = None) -> str:
        """
        Compile the given source model using any given compilation parameters.

        Args:
            model_file (str | Path): Path to the model file
            compilation_params (CompilationParams, optional): Parameters for compiling the model. Default is None.
            out_dir (str, optional): Path to a directory for storing compiled artifacts. Default is None.

        Returns:
            str: Path to DLC file or binary file

        Raises:
            TypeError: If compilation_params is not of type CompilationParams
        """

        self.model_file = model_file
        model_file_name = Path(self.model_file).stem
        binary = None

        if not compilation_params:
            compilation_params = CompilationParams()

        if not isinstance(compilation_params, CompilationParams):
            raise TypeError('compilation_params must be of type CompilationParams')

        self.backend = compilation_params.backend
        self.soc_model = compilation_params.soc_model
        self.conversion_params = compilation_params.conversion_params
        self.quant_params = compilation_params.quantization_params
        self.backend_extensions = compilation_params.backend_extensions
        self.offline_prepare = compilation_params.offline_prepare

        if self.backend:
            qairt_logger.info('Performing model compilation for given backend = {}'.format(
                self.backend))
            self.backend_info = BackendInfoConfig(backend=self.backend.value,
                                                  soc_model=self.soc_model)
            compiled_model_suffix = f'_{self.backend}'
        else:
            qairt_logger.info('Performing generic model compilation')
            self.backend_info = None
            compiled_model_suffix = ''

        if out_dir:
            out_dir = Path(out_dir)
            if not out_dir.exists():
                raise FileNotFoundError(f"Given path '{out_dir}' does not exist.")
        else:
            out_dir = Path(tempfile.mkdtemp(prefix="qairt_compile_"))

        ir_graph, framework = self.convert()

        dlc_path = out_dir / f'{model_file_name}{compiled_model_suffix}.dlc'
        dlc = self.optimize(ir_graph=ir_graph, framework=framework, output_path=dlc_path)

        if self.quant_params:
            compiled_model_suffix = f'_quantized{compiled_model_suffix}'
            quantized_dlc_path = out_dir / f'{model_file_name}{compiled_model_suffix}.dlc'
            dlc = self.quantize(input_dlc=dlc, output_path=quantized_dlc_path)

        if self.offline_prepare:
            binary = self.generate_binary(dlc)

        qairt_logger.info('Completed model compilation.')

        qairt_model = QairtModel(dlc=dlc, binary=binary, backend=self.backend)
        return qairt_model

    def convert(self) -> tuple[IROpGraph, str]:
        """
        Convert source model to IR graph using QAIRT converter

        Returns:
            IROpGraph: The converted IR graph object
        """

        qairt_logger.debug('Preparing converter module config')
        if self.conversion_params is None:
            converter_args = converter_module.ConverterInputConfig(
                input_network=str(self.model_file))
        else:
            qairt_logger.debug(
                f'Conversion parameters: {self.conversion_params.model_dump(exclude_unset=True)}')
            converter_args = converter_module.ConverterInputConfig(
                input_network=str(self.model_file),
                **self.conversion_params.model_dump(exclude_unset=True))

        qairt_logger.debug('Initializing Converter module')
        converter = converter_module.QAIRTConverter()
        qairt_logger.info('Converting source model to IR')

        try:
            converter_output = converter.convert(converter_args)
            ir_graph = converter_output.ir_graph
            framework = converter_output.framework
        except Exception as exception:
            raise ConversionFailure('Failed to convert the model!') from exception
        qairt_logger.info('Completed converting to IR')

        return ir_graph, framework

    def optimize(self, ir_graph: IROpGraph, framework: str, output_path: str | Path) -> str:
        """
        Optimize converted IR graph using IROptimizer

        Args:
            ir_graph (IROpGraph): IR graph to be optimized
            framework (str): Framework of the source model
            output_path (str | Path): Output path where the optimized DLC will be stored

        Returns:
            str: Path to DLC generated by serializing the optimized IR graph
        """
        qairt_logger.debug('Preparing optimizer module config')
        optimizer_args = optimizer_module.OptimizerInputConfig(ir_graph=ir_graph,
                                                               framework=framework,
                                                               output_dlc=str(output_path),
                                                               backend_info=self.backend_info)
        qairt_logger.debug('Initializing Optimizer module')
        optimizer = optimizer_module.QAIRTOptimizer()
        qairt_logger.info("Optimizing IR graph")
        try:
            optimizer_output = optimizer.optimize(optimizer_args)
            optimized_dlc = optimizer_output.dlc_path
        except Exception as exception:
            raise OptimizationFailure('Failed to optimize the model!') from exception
        qairt_logger.info("Completed optimization of IR graph.")

        return optimized_dlc

    def quantize(self, input_dlc: str, output_path: str) -> str:
        """
        Quantize the given DLC

        Args:
            input_dlc (str): Path to the DLC file that needs to be quantized.
            output_path (str): File path to be used for saving the Quantized DLC.

        Returns:
            str: Path to quantized DLC
        """

        qairt_logger.debug('Preparing quantizer module config')

        if self.quant_params is None:
            quant_args = quantizer_module.QuantizerInputConfig(input_dlc=str(input_dlc))
        else:
            qairt_logger.debug(
                f'Quantization parameters: {self.quant_params.model_dump(exclude_unset=True)}')
            quant_args = quantizer_module.QuantizerInputConfig(
                input_dlc=str(input_dlc), **self.quant_params.model_dump(exclude_unset=True))

        quant_args.output_dlc = str(output_path)

        if self.backend_info:
            quant_args.backend_info = self.backend_info

        qairt_logger.debug('Initializing Quantizer module')
        quantizer = quantizer_module.QAIRTQuantizer()
        qairt_logger.info('Performing quantization')
        try:
            quantizer_output = quantizer.quantize(quant_args)
            quantized_dlc = quantizer_output.dlc_output
        except Exception as exception:
            raise QuantizationFailure('Failed to quantize the model!') from exception

        qairt_logger.info('Completed quantization')
        qairt_logger.debug(f'Quantized model saved at: {quantized_dlc}')
        return quantized_dlc

    def generate_binary(self, dlc_file: str | Path) -> str:
        """
        Generate binary from DLC using context-binary-generator module.

        Args:
            dlc_file (str | Path): Path to the DLC file

        Returns
            str: Path to the generated binary
        """

        out_dir = Path(dlc_file).parent
        name = Path(dlc_file).stem

        qairt_logger.debug('Preparing arg config of context-bin-gen module')
        context_bin_gen_config = context_bin_gen_module.ContextBinGenArgConfig(
            backend=self.backend, backend_config_dict=self.backend_extensions,
            model=Model(dlc_path=dlc_file), output_dir=out_dir, output_filename=name, config=None)

        qairt_logger.debug('Initializing context-bin-gen module')
        context_bin_gen = context_bin_gen_module.ContextBinGen()

        qairt_logger.info('Starting offline graph preparation')
        try:
            output_config = context_bin_gen.generate(context_bin_gen_config)
            binary_path = output_config.context_binary.context_binary_path
            qairt_logger.info('Completed offline graph preparation')
            qairt_logger.debug(f'Binary file saved at {binary_path}')
        except Exception as exception:
            raise GraphPreparationFailure(f'Failed to prepare graph!') from exception

        return binary_path
