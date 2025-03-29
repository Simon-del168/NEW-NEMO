# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from abc import ABC, abstractmethod
from pathlib import Path
import json


class Backend(ABC):
    @abstractmethod
    def __init__(self, target):
        self.target = target
        self._default_target = target is None
        self._workflow_mode = None
        self._config = {}
        self._op_packages = []

    @property
    def workflow_mode(self):
        return self._workflow_mode

    @workflow_mode.setter
    def workflow_mode(self, mode):
        self._workflow_mode = mode
        self._workflow_mode_setter_hook(mode)

    def _workflow_mode_setter_hook(self, mode):
        """
        This method should be implemented by a backend subclass if:
            - The default target differs based on workflow (e.g. HTP by default should generate
              context binaries on x86 but run inferences on Android), in which case self.target
              should be defaulted to None in the constructor and set in this function if
              self._default_target is True (i.e. if the class was instantiated w/o a target)
            - The backend does not support a workflow (e.g. CPU does not support context binary
              generation), in which case a ValueError should be raised if an unsupported workflow is
              requested
        """
        pass

    @property
    @abstractmethod
    def backend_library(self):
        pass

    @property
    @abstractmethod
    def backend_extensions_library(self):
        pass

    @abstractmethod
    def get_required_artifacts(self, sdk_path):
        pass

    def get_config_json(self):
        return json.dumps(self._config, indent=2) if self._config else None

    def register_op_package(self, path, interface_provider, target=None):
        op_package_path = Path(str(path))
        if not op_package_path.is_file():
            raise FileNotFoundError(f"Could not find op package library: {op_package_path}")
        self._op_packages.append([op_package_path.resolve(), interface_provider, target])

    def get_registered_op_packages(self):
        return self._op_packages

    def before_run_hook(self, temp_directory, sdk_path):
        pass

    def after_run_hook(self, temp_directory, sdk_path):
        pass

    def before_generate_hook(self, temp_directory, sdk_path):
        pass

    def after_generate_hook(self, temp_directory, sdk_path):
        pass


# a descriptor to store backend configs in a dictionary that can be directly serialized to json
# while also providing dot notation access to individual configs, allowing them to be set
# dynamically. This strategy can be used for all backends whose configs are solely key-value pairs
class BackendConfig:
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        return instance._config.get(self._name)

    def __set__(self, instance, value):
        instance._config[self._name] = value
