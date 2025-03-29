# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import logging
import re

from qti.aisw.core.model_level_api.utils.subprocess_executor import execute

logger = logging.getLogger(__name__)
default_adb_shell_timeout = 1800


class Adb:
    def __init__(self, device_id):
        self._device_id = device_id
        devices = self.get_devices()
        if len(devices) == 0:
            raise RuntimeError("No ADB devices detected")
        if self._device_id is None:
            self._device_id = os.getenv('ANDROID_SERIAL')
        if self._device_id is None or self._device_id not in devices:
            self._device_id = devices[0]
            logger.warning(f"Device ID not provided or did not match a connected device, connecting"
                           f" to device ID {self._device_id}")

    def push(self, src, dst):
        dst_dir_exists = False
        return_code, _, _ = self._execute('shell', [f'[ -d {dst} ]'])
        if return_code == 0:
            dst_dir_exists = True

        return_code, stdout, stderr = self._execute('push', [src, dst])
        if return_code != 0:
            raise RuntimeError(f"Failed to push {src} to device, stdout: {stdout} stderr: {stderr}")

        # if destination was a directory that was already present on the device, ensure that the
        # source file/directory was pushed into that directory.
        # If destination was a directory that was not present on device, then it is sufficient to
        # just check that destination exists on the device after push.
        # If destination was a file, it should be checked for existence
        if dst_dir_exists:
            if dst[-1] == '/':
                dst = dst[:-1]
            if src[-1] == '/':
                src = src[:-1]
            src_file_name = os.path.basename(src)
            dst = dst + '/' + src_file_name

        return_code, _, _ = self._execute('shell', [f'[ -e {dst} ]'])
        if return_code != 0:
            raise RuntimeError(f"Failed to push {src} to device")

    def pull(self, src, dst):
        return self._execute('pull', [src, dst])

    def shell(self, command, args=None):
        if args is None:
            args = []

        # print the return code of the command run on device on stdout so it can be parsed on the
        # host. 'adb shell' will return a 0 exit code even if the command fails on device
        shell_args = [f"{command} {' '.join(args)}; echo '\n'$?"]
        logger.debug("Running command on device: " + ' '.join(shell_args))
        return_code, stdout, stderr = self._execute('shell', shell_args)

        if return_code == 0:
            if len(stdout) > 0:
                try:
                    return_code = int(stdout[-1])
                    stdout = stdout[:-1]
                except ValueError as ex:
                    return_code = -1
                    stdout.append(ex)
            else:
                return_code = -1

        return return_code, stdout, stderr

    def get_devices(self):
        return_code, stdout, stderr = self._execute('devices')
        if return_code != 0:
            raise RuntimeError(f"Could not retrieve list of connected adb devices, "
                               f"stdout: {stdout}, stderr: {stderr}")
        devices = []
        for line in stdout:
            match_obj = re.match(r"^([a-zA-Z0-9]+)\s+device", line, re.M)
            if match_obj:
                devices.append(match_obj.group(1))

        return devices

    def _execute(self, command, args=None):
        if args is None:
            args = []
        if self._device_id is None:
            adb_command_args = [command] + args
        else:
            adb_command_args = ['-s', self._device_id, command] + args
        return execute('adb', adb_command_args, timeout=default_adb_shell_timeout)
