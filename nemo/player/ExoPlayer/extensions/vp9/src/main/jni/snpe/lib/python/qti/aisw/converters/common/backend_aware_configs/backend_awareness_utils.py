#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os

def get_path_for_target_config(target_config: str) -> str:
    """
    Returns path for target config such as htp, aic, cpu and lpai

    :return: path for target config file
    """
    if target_config not in {'htp', 'aic', 'cpu', 'lpai'}:
        raise ValueError(f"Backend {target_config} does not have backend aware config.")

    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{target_config}.json')