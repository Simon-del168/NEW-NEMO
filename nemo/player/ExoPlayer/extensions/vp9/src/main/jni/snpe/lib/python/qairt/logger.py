# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import sys


def setup_qairt_logger():
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_level = logging.INFO
    qairt_logger = logging.getLogger('qairt_api')
    qairt_logger.handlers = []

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    ch.setFormatter(formatter)

    qairt_logger.parent = []
    qairt_logger.propagate = False

    qairt_logger.addHandler(ch)
    qairt_logger.setLevel(log_level)
