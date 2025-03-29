# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from pathlib import Path
from logging import getLogger
from json import load

from qti.aisw.core.model_level_api.utils.subprocess_executor import execute

logger = getLogger(__name__)
default_profiling_log_name = 'qnn-profiling-data_0.log'


def output_dir_to_profiling_data(output_dir, qnn_sdk_root):
    # ensure required artifacts are present
    profile_log = Path(output_dir, default_profiling_log_name)
    if not profile_log.is_file():
        raise FileNotFoundError("Could not locate profile log in output folder")

    qnn_profile_viewer = Path(qnn_sdk_root, 'bin', 'x86_64-linux-clang', 'qnn-profile-viewer')
    if not qnn_profile_viewer.is_file():
        raise FileNotFoundError("Could not locate qnn-profile-viewer in QNN SDK")

    json_reader = Path(qnn_sdk_root, 'lib', 'x86_64-linux-clang', 'libQnnJsonProfilingReader.so')
    if not json_reader.is_file():
        raise FileNotFoundError("Could not locate libQnnJsonProfilingReader.so in QNN SDK")

    output_json = Path(output_dir, 'profiling_output.json')
    profile_viewer_args = f'--input_log {profile_log} --output {output_json} --reader {json_reader}'

    logger.debug(f'Running command: {qnn_profile_viewer} {profile_viewer_args}')
    return_code, stdout, stderr = execute(qnn_profile_viewer, profile_viewer_args.split())
    if return_code != 0:
        raise RuntimeError(f'qnn-profile-viewer execution failed, stdout: {stdout}, stderr: '
                           f'{stderr}')

    with output_json.open() as f:
        output_dict = load(f)

    return output_dict
