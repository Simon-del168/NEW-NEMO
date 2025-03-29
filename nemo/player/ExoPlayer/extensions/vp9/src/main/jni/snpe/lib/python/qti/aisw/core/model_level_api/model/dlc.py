# =============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.core.model_level_api.model.model import Model


class DLC(Model):
    def __init__(self, name, path):
        super().__init__(name)
        self.dlc_path = path
