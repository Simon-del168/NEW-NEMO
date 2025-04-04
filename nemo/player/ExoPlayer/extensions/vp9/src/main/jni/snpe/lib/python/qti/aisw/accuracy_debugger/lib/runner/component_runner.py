# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import traceback

import pandas as pd
import numpy as np
import csv

from qti.aisw.accuracy_debugger.lib.framework_diagnosis.nd_framework_runner import FrameworkRunner
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_get_tensor_mapping import TensorMapper
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_inference_engine_manager import InferenceEngineManager
from qti.aisw.accuracy_debugger.lib.options.acc_debugger_cmd_options import AccDebuggerCmdOptions
from qti.aisw.accuracy_debugger.lib.options.framework_diagnosis_cmd_options import FrameworkDiagnosisCmdOptions
from qti.aisw.accuracy_debugger.lib.options.inference_engine_cmd_options import InferenceEngineCmdOptions
from qti.aisw.accuracy_debugger.lib.options.qairt_inference_engine_cmd_options import QAIRTInferenceEngineCmdOptions
from qti.aisw.accuracy_debugger.lib.options.verification_cmd_options import VerificationCmdOptions
from qti.aisw.accuracy_debugger.lib.options.compare_encodings_cmd_options import CompareEncodingsCmdOptions
from qti.aisw.accuracy_debugger.lib.options.quant_checker_cmd_options import QuantCheckerCmdOptions

from qti.aisw.accuracy_debugger.lib.options.tensor_inspection_cmd_options import TensorInspectionCmdOptions

from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine, DebuggingAlgorithm, FrameworkExtension, Framework, ComponentLogCodes
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_progress_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import FrameworkError, InferenceEngineError, VerifierError
from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace, remove_layer_options
from qti.aisw.accuracy_debugger.lib.utils.nd_symlink import symlink
from qti.aisw.accuracy_debugger.lib.utils.nd_verifier_utility import save_to_file
from qti.aisw.accuracy_debugger.lib.verifier.nd_verification import Verification
from qti.aisw.accuracy_debugger.lib.compare_encodings.compare_encodings_runner import CompareEncodingsRunner
from qti.aisw.accuracy_debugger.lib.tensor_inspection.tensor_inspection_runner import TensorInspectionRunner
from qti.aisw.accuracy_debugger.lib.verifier.nd_tensor_inspector import TensorInspector
from qti.aisw.accuracy_debugger.lib.quant_checker import get_generator_cls
from qti.aisw.accuracy_debugger.lib.quant_checker.nd_quant_checker import QuantChecker


def exec_framework_diagnosis(args, logger=None, validate_args=True):
    framework_args = FrameworkDiagnosisCmdOptions(args, validate_args).parse()
    if (logger is None):
        logger = setup_logger(framework_args.verbose, framework_args.output_dir,
                              component=ComponentLogCodes.framework_diagnosis.value)

    logger.info(get_progress_message('PROGRESS_FRAMEWORK_STARTING'))

    symlink('latest', framework_args.output_dir, logger)

    try:
        framework_runner = FrameworkRunner(logger, framework_args)
        framework_runner.run()
        logger.info(get_progress_message('PROGRESS_FRAMEWORK_FINISHED'))
    except FrameworkError as e:
        raise FrameworkError("Conversion failed: {}".format(str(e)))
    except Exception as e:
        traceback.print_exc()
        raise Exception("Encountered Error: {}".format(str(e)))


def exec_inference_engine(args, engine_type, logger=None, validate_args=True):
    if engine_type in [Engine.QNN.value, Engine.SNPE.value]:
        inference_engine_args = InferenceEngineCmdOptions(engine_type, args, validate_args).parse()
    elif engine_type == Engine.QAIRT.value:
        inference_engine_args = QAIRTInferenceEngineCmdOptions(engine_type, args,
                                                               validate_args).parse()
    else:
        raise InferenceEngineError(
            get_message("ERROR_INFERENCE_ENGINE_ENGINE_NOT_FOUND")(engine_type))

    if (logger is None):
        logger = setup_logger(inference_engine_args.verbose, inference_engine_args.output_dir,
                              component=ComponentLogCodes.inference_engine.value)

    logger.info(get_progress_message('PROGRESS_INFERENCE_ENGINE_STARTING'))

    symlink('latest', inference_engine_args.output_dir, logger)

    try:
        inference_engine_manager = InferenceEngineManager(inference_engine_args, logger=logger)
        inference_engine_manager.run_inference_engine()

        # Run tensor mapping
        get_mapping_arg = Namespace(
            None, framework=inference_engine_args.framework, version=None
            if engine_type == Engine.QAIRT.value else inference_engine_args.framework_version,
            model_path=inference_engine_args.model_path,
            output_dir=inference_engine_args.output_dir, engine=inference_engine_args.executor_type
            if hasattr(inference_engine_args, 'executor_type') else inference_engine_args.engine,
            golden_dir_for_mapping=inference_engine_args.golden_output_reference_directory)
        TensorMapper(get_mapping_arg, logger).run()

        logger.info(get_progress_message('PROGRESS_INFERENCE_ENGINE_FINISHED'))
    except InferenceEngineError as e:
        raise InferenceEngineError("Inference failed: {}".format(str(e)))
    except Exception as e:
        traceback.print_exc()
        raise Exception("Encountered Error: {}".format(str(e)))


def exec_verification(args, logger=None, run_tensor_inspection=False, validate_args=True):
    verification_args = VerificationCmdOptions(args, validate_args).parse()
    if (logger is None):
        logger = setup_logger(verification_args.verbose, verification_args.output_dir,
                              component=ComponentLogCodes.verification.value)

    try:
        logger.info(get_progress_message("PROGRESS_VERIFICATION_STARTING"))
        if not verification_args.tensor_mapping:
            logger.warn(
                "--tensor_mapping is not set, a tensor_mapping will be generated based on user input."
            )
            get_mapping_arg = Namespace(
                None, framework=verification_args.framework,
                version=None if verification_args.engine == Engine.QAIRT.value else
                verification_args.framework_version, model_path=verification_args.model_path,
                output_dir=verification_args.inference_results, engine=verification_args.engine,
                golden_dir_for_mapping=verification_args.golden_output_reference_directory)
            verification_args.tensor_mapping = TensorMapper(get_mapping_arg, logger).run()

        verify_results = []
        for verifier in verification_args.verify_types:
            # Splitting with comma to handle cases where verifiers have parameters(Ex: --default_verifier rtolatol,rtolmargin,0.1,atolmargin,0.2)
            verifier = verifier[0].split(',')
            verify_type = verifier[0]
            verifier_configs = verifier[1:]
            verification = Verification(verify_type, logger, verification_args, verifier_configs)
            if verification.has_specific_verifier() and len(verification_args.verify_types) > 1:
                raise VerifierError(get_message('ERROR_VERIFIER_USE_MULTI_VERIFY_AND_CONFIG'))
            verify_result = verification.verify_tensors()
            verify_result = verify_result.drop(columns=['Units', 'Verifier'])
            verify_result = verify_result.rename(columns={'Metric': verify_type})
            verify_results.append(verify_result)

        if run_tensor_inspection:
            # run tensor inspector which plots analysis graphs between golden and target data
            logger.info(get_progress_message('PROGRESS_TENSOR_INSPECTION_STARTING'))
            inspection_results = TensorInspector(logger, verification_args).run()
            logger.info(get_progress_message('PROGRESS_TENSOR_INSPECTION_FINISHED'))

        # if verification_args.verifier_config is None, all tensors use the same verifer. So we can export Summary
        if verification_args.verifier_config == None:
            summary_df = verify_results[0]
            for verify_result in verify_results[1:]:
                summary_df = pd.merge(summary_df, verify_result,
                                      on=['Name', 'LayerType', 'Size', 'Tensor_dims'])

            if run_tensor_inspection:
                summary_df = pd.merge(summary_df, inspection_results, on=['Name'])
            filename = os.path.join(verification_args.output_dir, Verification.SUMMARY_NAME)
            save_to_file(summary_df, filename)

        symlink('latest', verification_args.output_dir, logger)
        logger.info(get_progress_message("PROGRESS_VERIFICATION_FINISHED"))
    except VerifierError as excinfo:
        raise Exception("Verification failed: {}".format(str(excinfo)))
    except Exception as excinfo:
        traceback.print_exc()
        raise Exception("Encountered error: {}".format(str(excinfo)))


def exec_compare_encodings(args, engine_type, logger=None, validate_args=True):
    compare_encodings_args = CompareEncodingsCmdOptions(args, validate_args).parse()
    if (logger is None):
        logger = setup_logger(compare_encodings_args.verbose, compare_encodings_args.output_dir,
                              component=ComponentLogCodes.compare_encodings.value)

    logger.info(get_progress_message('PROGRESS_COMPARE-ENCODINGS_STARTING'))

    symlink('latest', compare_encodings_args.output_dir, logger)

    try:
        compare_encodings = CompareEncodingsRunner(logger, compare_encodings_args)
        compare_encodings.run(engine_type)
        logger.info(get_progress_message('PROGRESS_COMPARE-ENCODINGS_FINISHED'))
    except Exception as e:
        traceback.print_exc()
        raise Exception("Encountered Error: {}".format(str(e)))


def exec_tensor_inspection(args, logger=None, validate_args=True):
    tensor_inspection_args = TensorInspectionCmdOptions(args, validate_args).parse()
    if (logger is None):
        logger = setup_logger(tensor_inspection_args.verbose, tensor_inspection_args.output_dir,
                              component=ComponentLogCodes.tensor_inspection.value)

    logger.info(get_progress_message('PROGRESS_TENSOR-INSPECTION_STARTING'))

    symlink('latest', tensor_inspection_args.output_dir, logger)

    dtype_map = {
        "int8": np.int8,
        "uint8": np.uint8,
        "int16": np.int16,
        "uint16": np.uint16,
        "float32": np.float32,
    }

    try:
        # initialize TensorInspectionRunner
        tensor_inspector = TensorInspectionRunner(logger)

        summary = []
        golden_files = os.listdir(tensor_inspection_args.golden_data)
        for file in os.listdir(tensor_inspection_args.target_data):

            if file not in golden_files:
                logger.warning(f"{file} present only in target data path, skipping this file.")
                continue

            if not file.endswith('.raw'):
                logger.warning(f"{file} is not a raw file, skipping this file.")
                continue

            golden_path = os.path.join(tensor_inspection_args.golden_data, file)
            target_path = os.path.join(tensor_inspection_args.target_data, file)
            golden = np.fromfile(golden_path, dtype=dtype_map[tensor_inspection_args.data_type])
            target = np.fromfile(target_path, dtype=dtype_map[tensor_inspection_args.data_type])

            # trigger TensorInspectionRunner on current golden and target tensors
            result = tensor_inspector.run(file, golden, target, tensor_inspection_args.output_dir,
                                          target_encodings=tensor_inspection_args.target_encodings,
                                          verifiers=tensor_inspection_args.verifier)
            summary.append(result)

        # dump summary results to csv file
        csv_path = os.path.join(tensor_inspection_args.output_dir, 'summary.csv')
        verifier_names = [verifier[0].split(',')[0] for verifier in tensor_inspection_args.verifier]
        fields = ['Name'
                  ] + verifier_names + ['golden_min', 'golden_max', 'target_min', 'target_max']
        if tensor_inspection_args.target_encodings:
            fields.extend([
                'calibrated_min', 'calibrated_max', '(target_min-calibrated_min)',
                '(target_max-calibrated_max)'
            ])

        with open(csv_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)

        logger.info(get_progress_message('PROGRESS_TENSOR-INSPECTION_FINISHED'))
    except Exception as e:
        traceback.print_exc()
        raise Exception("Encountered Error: {}".format(str(e)))


def exec_wrapper(args, engine_type, logger=None, validate_args=True):
    wrapper_args = AccDebuggerCmdOptions(engine_type, args, validate_args).parse()
    subcomponent = ComponentLogCodes.oneshot_layerwise.value

    if engine_type == Engine.QNN.value:
        if wrapper_args.debugging_algorithm == DebuggingAlgorithm.layerwise.value:
            subcomponent = ComponentLogCodes.layerwise.value
        elif wrapper_args.debugging_algorithm == DebuggingAlgorithm.cumulative_layerwise.value:
            subcomponent = ComponentLogCodes.cumulative_layerwise.value

    if (logger is None):
        logger = setup_logger(wrapper_args.verbose, wrapper_args.output_dir, component=subcomponent)

    # run framework diagnosis if required
    framework_results = None
    if wrapper_args.golden_output_reference_directory:
        # skip framework diagnosis if golden_output_reference_directory is passed
        framework_results = wrapper_args.golden_output_reference_directory
        logger.info("Golden reference directory supplied. Skipping framework diagnosis.")
    elif wrapper_args.framework == Framework.pytorch.value:
        # collect golden reference outputs from inference engine on cpu runtime
        reference_args = list(args)
        # Filter out layer options like '--start_layer', '--end_layer', '--add_layer_outputs', '--add_layer_types',
        # '--skip_layer_outputs', '--skip_layer_types' for framework_diagnosis component
        if engine_type == Engine.QNN.value and \
            (wrapper_args.debugging_algorithm == DebuggingAlgorithm.layerwise.value or \
            wrapper_args.debugging_algorithm == DebuggingAlgorithm.cumulative_layerwise.value):
            reference_args = remove_layer_options(reference_args)
        model_name = "model"
        reference_args.extend(['--model_name', model_name])
        reference_args.extend(['--output_dirname', 'reference_outputs'])
        # replace --runtime arg to avoid ambiguity error
        if '--runtime' in reference_args:
            reference_args[reference_args.index('--runtime')] = '-r'
        if '-r' in reference_args:
            reference_args[reference_args.index('-r') + 1] = 'cpu'
        # replace --engine args to avoid ambiguity error
        if '--engine' in reference_args:
            reference_args[reference_args.index('--engine')] = '-e'
        if '--offline_prepare' in reference_args:
            del reference_args[reference_args.index('--offline_prepare')]
        if '--architecture' in reference_args:
            reference_args[reference_args.index('--architecture') + 1] = "x86_64-linux-clang"
        # runs inference engine
        exec_inference_engine(reference_args, engine_type, logger=logger, validate_args=False)
        framework_results = os.path.join(wrapper_args.working_dir, 'inference_engine',
                                         'reference_outputs', 'output', 'Result_0')
    else:
        framework_args = list(args)
        # Filter out layer options like '--start_layer', '--end_layer', '--add_layer_outputs', '--add_layer_types',
        # '--skip_layer_outputs', '--skip_layer_types' for framework_diagnosis component
        if engine_type == Engine.QNN.value and \
            (wrapper_args.debugging_algorithm == DebuggingAlgorithm.layerwise.value or \
            wrapper_args.debugging_algorithm == DebuggingAlgorithm.cumulative_layerwise.value):
            framework_args = remove_layer_options(framework_args)
        exec_framework_diagnosis(framework_args, logger=logger, validate_args=False)
        framework_results = os.path.join(wrapper_args.working_dir, 'framework_diagnosis', 'latest')

        if "--disable_graph_optimization" not in args and wrapper_args.framework == "onnx":
            optimized_model_path = os.path.join(
                framework_results, "optimized_model" +
                FrameworkExtension.framework_extension_mapping[wrapper_args.framework])
            if os.path.exists(optimized_model_path):
                if '-m' in list(args):
                    args[args.index('-m')] = '--model_path'
                args[args.index('--model_path') + 1] = optimized_model_path

    if engine_type == Engine.QNN.value and \
            (wrapper_args.debugging_algorithm == DebuggingAlgorithm.layerwise.value or \
            wrapper_args.debugging_algorithm == DebuggingAlgorithm.cumulative_layerwise.value):
        layerwise_args = list(args)
        layerwise_args.extend(['--golden_output_reference_directory', framework_results])

        # run layerwise snooping
        if wrapper_args.debugging_algorithm == DebuggingAlgorithm.layerwise.value:
            exec_layerwise_snooping(layerwise_args, logger, validate_args=False)
        # run cumulative layerwise snooping
        elif wrapper_args.debugging_algorithm == DebuggingAlgorithm.cumulative_layerwise.value:
            exec_cumulative_layerwise_snooping(layerwise_args, logger, validate_args=False)
        return

    # inference engine args pre-processing
    inference_args = list(args)
    model_name = "model"
    inference_args.extend(['--model_name', model_name])
    # replace --engine args to avoid ambiguity error
    if '--engine' in inference_args:
        inference_args[inference_args.index('--engine')] = '-e'
    # runs inference engine
    exec_inference_engine(inference_args, engine_type, logger=logger, validate_args=False)

    # verification args pre-processing
    verification_args = list(args)
    graph_structure = model_name + '_graph_struct.json'
    graph_structure_path = os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest',
                                        graph_structure)
    verification_args.extend(['--graph_struct', graph_structure_path])

    verification_args.extend([
        '--inference_results',
        os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest', 'output/Result_0')
    ])

    verification_args.extend([
        '--golden_output_reference_directory', framework_results, '--tensor_mapping',
        os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest', 'tensor_mapping.json')
    ])

    if engine_type == Engine.QNN.value:
        qnn_model_net_json = model_name + '_net.json'
        qnn_model_net_json_path = os.path.join(wrapper_args.working_dir, 'inference_engine',
                                               'latest', qnn_model_net_json)
        verification_args.extend(['--qnn_model_json_path', qnn_model_net_json_path])
    elif engine_type in [Engine.QAIRT.value, Engine.SNPE.value]:
        dlc_path = os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest', 'base.dlc')
        verification_args.extend(['--dlc_path', dlc_path])

    # runs verification
    exec_verification(verification_args, logger=logger,
                      run_tensor_inspection=wrapper_args.enable_tensor_inspection,
                      validate_args=False)

    if engine_type == Engine.QNN.value and wrapper_args.debugging_algorithm == DebuggingAlgorithm.modeldissection.value:
        # deep analyzer args pre-processing
        da_param_index = args.index('--deep_analyzer')
        deep_analyzers = args[da_param_index + 1].split(',')
        del args[da_param_index:da_param_index + 2]
        deep_analyzer_args = list(args)
        deep_analyzer_args.extend([
            '--tensor_mapping',
            os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest',
                         'tensor_mapping.json'), '--inference_results',
            os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest',
                         'output/Result_0'), '--graph_struct', graph_structure_path,
            '--framework_results', framework_results, '--result_csv',
            os.path.join(wrapper_args.working_dir, 'verification', 'latest', 'summary.csv')
        ])
        # runs deep analyzers
        for d_analyzer in deep_analyzers:
            exec_deep_analyzer(deep_analyzer_args + ['--deep_analyzer', d_analyzer], logger=logger,
                               validate_args=False)


def exec_deep_analyzer(args, logger=None, validate_args=True):
    da_args = AccuracyDeepAnalyzerCmdOptions(args, validate_args).parse()
    if not os.path.isdir(da_args.output_dir):
        os.makedirs(da_args.output_dir)
    if not logger:
        logger = setup_logger(da_args.verbose, da_args.output_dir)

    symlink('latest', da_args.output_dir, logger)

    try:
        from qti.aisw.accuracy_debugger.lib.deep_analyzer.nd_deep_analyzer import DeepAnalyzer
        from qti.aisw.accuracy_debugger.lib.options.accuracy_deep_analyzer_cmd_options import AccuracyDeepAnalyzerCmdOptions
        from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import DeepAnalyzerError

        if not da_args.tensor_mapping:
            logger.warn(
                "--tensor_mapping is not set, a tensor_mapping will be generated based on user input."
            )
            get_mapping_arg = Namespace(None, framework=da_args.framework,
                                        version=da_args.framework_version,
                                        model_path=da_args.model_path,
                                        output_dir=da_args.inference_results, engine=da_args.engine,
                                        golden_dir_for_mapping=da_args.framework_results)
            da_args.tensor_mapping = TensorMapper(get_mapping_arg, logger).run()
        deep_analyzer = DeepAnalyzer(da_args, logger)
        deep_analyzer.analyze()
        logger.info("Successfully ran deep_analyzer!")
    except DeepAnalyzerError as excinfo:
        raise DeepAnalyzerError("deep analyzer failed: {}".format(str(excinfo)))
    except Exception as excinfo:
        traceback.print_exc()
        raise Exception("Encountered error: {}".format(str(excinfo)))


def exec_cumulative_layerwise_snooping(args, logger=None, validate_args=True):
    try:
        from qti.aisw.accuracy_debugger.lib.snooping.nd_cumulative_layerwise_snooper import CumulativeLayerwiseSnooping
        from qti.aisw.accuracy_debugger.lib.options.layerwise_snooping_cmd_options import LayerwiseSnoopingCmdOptions
        from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import LayerwiseSnoopingError

        args = LayerwiseSnoopingCmdOptions(args, snooper='cumulative_layerwise',
                                           validate_args=validate_args).parse()
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        if not logger:
            logger = setup_logger(args.verbose, args.output_dir)

        symlink('latest', args.output_dir, logger)

        snooper = CumulativeLayerwiseSnooping(args, logger)
        snooper.run()
        logger.info("Successfully ran cumulative layerwise snooping!")
    except LayerwiseSnoopingError as excinfo:
        raise LayerwiseSnoopingError("Cumulative layerwise snooping failed: {}".format(
            str(excinfo)))
    except Exception as excinfo:
        traceback.print_exc()
        raise Exception("Encountered error: {}".format(str(excinfo)))


def exec_layerwise_snooping(args, logger=None, validate_args=True):
    try:
        from qti.aisw.accuracy_debugger.lib.snooping.nd_layerwise_snooper import LayerwiseSnooping
        from qti.aisw.accuracy_debugger.lib.options.layerwise_snooping_cmd_options import LayerwiseSnoopingCmdOptions
        from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import LayerwiseSnoopingError

        args = LayerwiseSnoopingCmdOptions(args, snooper='layerwise',
                                           validate_args=validate_args).parse()
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        if not logger:
            logger = setup_logger(args.verbose, args.output_dir)

        symlink('latest', args.output_dir, logger)

        snooper = LayerwiseSnooping(args, logger)
        snooper.run()
        logger.info("Successfully ran layerwise snooping!")
    except LayerwiseSnoopingError as excinfo:
        raise LayerwiseSnoopingError("Layerwise snooping failed: {}".format(str(excinfo)))
    except Exception as excinfo:
        traceback.print_exc()
        raise Exception("Encountered error: {}".format(str(excinfo)))


def exec_quant_checker(args, engine, logger=None, validate_args=True):
    # runs the quant checker (parsed_args)
    quant_checker_args = QuantCheckerCmdOptions(args, engine, validate_args).parse()
    if (logger is None):
        logger = setup_logger(quant_checker_args.verbose, quant_checker_args.output_dir,
                              component=ComponentLogCodes.quant_checker.value)
    # Create latest symlink
    symlink('latest', quant_checker_args.output_dir, logger)
    if quant_checker_args.golden_output_reference_directory is None:
        #STEP1: Run and dump the intermediate outputs
        # Since framework diagnosis does not support input list text file and it only
        # takes single input, so we need to iteratively run for all inputs in the text file
        if quant_checker_args.input_list:
            if '--output_dirname' not in args:
                args.extend(['--output_dirname', 'abc'])
                output_dirname_index = -1
                original_dirname = ''
            else:
                output_dirname_index = args.index('--output_dirname') + 1
                original_dirname = args[output_dirname_index]

            input_tensor_flag_indices = [
                idx for idx in range(len(args)) if args[idx] in ('-i', '--input_tensor')
            ]
            input_tensor_file_path_indices = [idx + 3 for idx in input_tensor_flag_indices]
            original_input_tensor_file_paths = [args[idx] for idx in input_tensor_file_path_indices]

            with open(quant_checker_args.input_list) as file:
                for line in file.readlines():
                    filenames = line.rstrip().split('\n')[0]
                    if filenames == "":
                        continue
                    file_name = []
                    for idx, file in enumerate(filenames.split(' ')):
                        #input_list in case of multi input nodes contain ":=" string
                        #while single input model may not contain them
                        file = file.split(":=")[1] if ":=" in file else file
                        base_name = os.path.basename(file)
                        name, _ = os.path.splitext(base_name)
                        file_name.append(name)
                        args[input_tensor_file_path_indices[idx]] = file
                    args[output_dirname_index] = "_".join(file_name)
                    exec_framework_diagnosis(args, logger=logger, validate_args=False)
            #Restore original file path
            for idx, file_path in zip(input_tensor_file_path_indices,
                                      original_input_tensor_file_paths):
                args[idx] = file_path
            #Restore original working dirname
            if original_dirname == '':
                args = args[:-2]
            else:
                args[output_dirname_index] = original_dirname

        #run framework diagnosis
        else:
            exec_framework_diagnosis(args, logger=logger, validate_args=False)

        quant_checker_args.golden_output_reference_directory = os.path.join(
            quant_checker_args.working_dir, 'framework_diagnosis')
        if '--disable_graph_optimization' not in args:
            optimized_model_path = os.path.join(
                quant_checker_args.golden_output_reference_directory, "optimized_model" +
                FrameworkExtension.framework_extension_mapping[quant_checker_args.framework])
            if os.path.exists(optimized_model_path):
                quant_checker_args.model_path = optimized_model_path
    else:
        if '--disable_graph_optimization' not in args:
            optimized_model_path = os.path.join(
                quant_checker_args.golden_output_reference_directory, "optimized_model" +
                FrameworkExtension.framework_extension_mapping[quant_checker_args.framework])
            if os.path.exists(optimized_model_path):
                quant_checker_args.model_path = optimized_model_path
            else:
                logger.info(
                    "Please make sure model passed to QuantChecker is same as on which fp32 outputs are dumped."
                )

    if engine == "SNPE":
        quant_checker_args.input_list = quant_checker_args.snpe_input_list

    #generate model for each quantization scheme, and perform verirfication
    quant_checker = QuantChecker(quant_checker_args, logger)
    quant_checker.run()


def exec_binary_snooping(args, engine, logger=None, validate_args=False):
    from qti.aisw.accuracy_debugger.lib.options.binary_snooping_cmd_options import BinarySnoopingCmdOptions
    from qti.aisw.accuracy_debugger.lib.snooping.nd_binary_snooper import BinarySnooping
    binary_snooping_args = BinarySnoopingCmdOptions(args, engine, validate_args).parse()
    if (logger is None):
        logger = setup_logger(binary_snooping_args.verbose, binary_snooping_args.output_dir,
                              component=ComponentLogCodes.binary_snooping.value)

    if '--output_dirname' not in args:
        args.extend(['--output_dirname', 'abc'])
        output_dirname_index = -1
        original_dirname = ''
    else:
        output_dirname_index = args.index('--output_dirname') + 1
        original_dirname = args[output_dirname_index]

    input_tensor_flag_indices = [
        idx for idx in range(len(args)) if args[idx] in ('-i', '--input_tensor')
    ]
    input_tensor_file_path_indices = [idx + 3 for idx in input_tensor_flag_indices]
    original_input_tensor_file_paths = [args[idx] for idx in input_tensor_file_path_indices]

    with open(binary_snooping_args.input_list) as file:
        for idx, line in enumerate(file.readlines()):
            if line:
                filenames = line.rstrip().split('\n')[0]
                for file_idx, file in enumerate(filenames.split(' ')):
                    #input_list in case of multi input nodes contain ":=" string
                    #while single input model may not contain them
                    file = file.split(":=")[1] if ":=" in file else file
                    args[input_tensor_file_path_indices[file_idx]] = file
                args[output_dirname_index] = "Result_" + str(idx)
                framework_args = [
                    *(args), "--add_layer_outputs", ",".join(binary_snooping_args.output_tensor)
                ]
                exec_framework_diagnosis(framework_args, logger=logger, validate_args=False)
    #Restore original file path
    for idx, file_path in zip(input_tensor_file_path_indices, original_input_tensor_file_paths):
        args[idx] = file_path
    #Restore original working dirname
    if original_dirname == '':
        args = args[:-2]
    else:
        args[output_dirname_index] = original_dirname

    binary_snooper = BinarySnooping(binary_snooping_args, logger)
    binary_snooper.run()

def exec_snooper(args, type, engine, logger=None, validate_args=True):
    from qti.aisw.accuracy_debugger.lib.options.snooping.get_snooping_cmd_option_class import get_snooping_cmd_option_class
    from qti.aisw.accuracy_debugger.lib.snooping.QAIRT.nd_get_snooper_class import get_snooper_class

    try:
        snooper_cmd_class = get_snooping_cmd_option_class(type)
        subcomponent = ComponentLogCodes.binary_snooping.value

        if type == DebuggingAlgorithm.layerwise.value:
            subcomponent = ComponentLogCodes.layerwise.value
        elif type == DebuggingAlgorithm.cumulative_layerwise.value:
            subcomponent = ComponentLogCodes.cumulative_layerwise.value
        elif type == DebuggingAlgorithm.oneshot_layerwise.value:
            subcomponent = ComponentLogCodes.oneshot_layerwise.value

        snooper_args = snooper_cmd_class(args, validate_args=validate_args).parse()
        if (logger is None):
            logger = setup_logger(snooper_args.verbose, snooper_args.output_dir,
                                  component=subcomponent)
        snooper_class = get_snooper_class(type)
        snooper = snooper_class(args=snooper_args, logger=logger)
        snooper.run()
    except Exception as e:
        raise Exception("Encountered Error: {}".format(str(e)))
