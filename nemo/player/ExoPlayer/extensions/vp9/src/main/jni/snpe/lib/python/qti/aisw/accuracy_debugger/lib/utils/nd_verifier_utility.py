# =============================================================================
#
#  Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import os

from qti.aisw.accuracy_debugger.lib.utils.nd_constants import AxisFormat, qnn_datatype_to_size
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json, dump_json
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import DeviceError


def get_ir_graph(dlc_path):
    """
    Returned IRGraph loaded with given dlc
    """
    from qti.aisw.dlc_utils import modeltools

    model_reader = modeltools.IrDlcReader()
    model_reader.open(dlc_path)
    ir_graph = model_reader.get_ir_graph()
    model_reader.close()

    return ir_graph


def get_tensors_axis_from_dlc(dlc_path):
    """
    Returns axis of each tensor in the given dlc
    """

    ir_graph = get_ir_graph(dlc_path)

    irgraph_axis_data = {}
    for name, tensor in ir_graph.get_tensor_map().items():
        sanitized_name = santize_node_name(name)
        irgraph_axis_data[sanitized_name] = {
            'src_axis_format': tensor.src_axis_format().name,
            'axis_format': tensor.axis_format().name,
            'dims': tensor.dims()
        }

    return irgraph_axis_data


def get_irgraph_axis_data(qnn_model_json_path=None, dlc_path=None, output_dir=None):
    """
    Returns axis of each tensor in the given qnn_model_json_path/dlc
    """
    irgraph_axis_data = {}
    if qnn_model_json_path is not None:
        qnn_model_json = read_json(qnn_model_json_path)
        irgraph_axis_data = qnn_model_json['graph']['tensors']
    elif dlc_path is not None:
        dlc_name = os.path.basename(dlc_path)
        axis_info_json_path = os.path.join(output_dir, dlc_name.replace('.dlc', '.json'))
        if os.path.exists(axis_info_json_path):
            irgraph_axis_data = read_json(axis_info_json_path)
        else:
            irgraph_axis_data = get_tensors_axis_from_dlc(dlc_path)
            dump_json(irgraph_axis_data, axis_info_json_path)

    return irgraph_axis_data


def permute_tensor_data_axis_order(src_axis_format, axis_format, tensor_dims, golden_tensor_data):
    """Permutes intermediate tensors goldens to spatial-first axis order for
    verification :param src_axis_format: axis format of source framework tensor
    :param axis_format: axis format of QNN tensor :param tensor_dims: current
    dimensions of QNN tensor :param golden_tensor_data: golden tensor data to
    be permuted :return: np.array of permuted golden tensor data."""

    # base case for same axis format or other invalid cases
    invalid_axis = ['NONTRIVIAL', 'NOT_YET_DEFINED', 'ANY']
    if src_axis_format == axis_format or \
        src_axis_format in invalid_axis or axis_format in invalid_axis:
        return golden_tensor_data, False
    # reshape golden data to spatial-last axis format
    golden_tensor_data = np.reshape(
        golden_tensor_data,
        tuple([
            tensor_dims[i]
            for i in AxisFormat.axis_format_mappings.value[(src_axis_format, axis_format)][0]
        ]))
    # transpose golden data to spatial-first axis format
    golden_tensor_data = np.transpose(
        golden_tensor_data,
        AxisFormat.axis_format_mappings.value[(src_axis_format, axis_format)][1])
    # return flatten golden data
    return golden_tensor_data.flatten(), True


def get_tensor_names_from_dlc(dlc_path, sanitize_names=False):
    """
    Returns tensor names present in the given dlc
    """

    ir_graph = get_ir_graph(dlc_path)

    tensor_names = []
    for name, tensor in ir_graph.get_tensor_map().items():
        if sanitize_names:
            name = santize_node_name(name)
        tensor_names.append(name)

    return tensor_names


def get_intermediate_tensors_size_from_dlc(dlc_path):
    """
    Returns the sizes of intermediate tensors present in the given DLC file.

    Args:
        dlc_path (str): The path to the DLC file.

    Returns:
        dict: A dictionary where keys are tensor names and values are their corresponding sizes in megabytes.
    """
    # Get the IR graph from the DLC file
    ir_graph = get_ir_graph(dlc_path)

    # Initialize an empty dictionary to store intermediate tensor sizes
    intermediate_tensors_size = {}

    # Iterate over tensors in the IR graph
    for name, tensor in ir_graph.get_tensor_map().items():
        # Check if the tensor type is one of ["NATIVE", "APP_WRITE", "APP_READ"]
        if tensor.tensor_type() in ["NATIVE", "APP_WRITE", "APP_READ"]:
            tensor_name = str(tensor.name())
            tensor_dim = tensor.dims()
            data_type_size_bits = int(qnn_datatype_to_size.get(str(tensor.data_type()), 32))
            tensor_size_mbs = (eval('*'.join(map(str, tensor_dim))) *
                               data_type_size_bits) / (8 * 1024 * 1024)
            intermediate_tensors_size[tensor_name] = tensor_size_mbs

    return intermediate_tensors_size


def get_dlc_size(dlc_path):
    """
    Calculates the size of a DLC file in megabytes.

    Args:
        dlc_path (str): The path to the DLC file.

    Returns:
        float: The size of the DLC file in megabytes.
    """
    # Get the size of the file in bytes
    file_size_bytes = os.path.getsize(dlc_path)

    # Convert bytes to megabytes
    file_size_mb = file_size_bytes / (1024 * 1024)

    return file_size_mb


def divide_output_tensors(tensor_size_dict, max_size):
    """
    Divides a dictionary of tensors based on their sizes, ensuring that the total size of each divided list
    does not exceed the specified maximum size.

    Args:
        tensor_size_dict (dict): A dictionary where keys are tensor names and values are their corresponding sizes.
        max_size (int): The maximum size allowed for each divided list.

    Returns:
        list of lists: A list of lists, where each inner list contains tensor names whose total size does not exceed max_size.
    """
    # Initialize lists to store divided tensors
    divided_lists = []
    current_list = []
    current_size = 0

    # Iterate over tensors in the original order
    for tensor_name, tensor_size in tensor_size_dict.items():
        # If adding the current tensor exceeds the max size, start a new list
        if current_size + tensor_size > max_size:
            if current_list: divided_lists.append(current_list)
            current_list = []
            current_size = 0

        # Add the current tensor to the current list
        current_list.append(tensor_name)
        current_size += tensor_size

    # Add the last list to divided_lists
    if current_list:
        divided_lists.append(current_list)

    return divided_lists


def to_csv(data, file_path):
    if isinstance(data, pd.DataFrame):
        data.to_csv(file_path, encoding='utf-8', index=False)


def to_html(data, file_path):
    if isinstance(data, pd.DataFrame):
        data.to_html(file_path, classes='table', index=False)


def to_json(data, file_path):
    if isinstance(data, pd.DataFrame):
        data.to_json(file_path, orient='records', indent=4)


def save_to_file(data, filename) -> None:
    """Save data to file in CSV, HTML and JSON formats :param data: Data to be
    saved to file :param filename: Name of the file."""
    filename = Path(filename)
    to_csv(data, filename.with_suffix(".csv"))
    to_html(data, filename.with_suffix(".html"))
    to_json(data, filename.with_suffix(".json"))


def validate_aic_device_id(device_ids):
    """
    Validate the provided AIC device IDs against the list of connected devices.

    Parameters:
    device_ids (list): List containing the device IDs to be validated.

    Returns:
    bool: True if all device IDs are valid, raises DeviceError otherwise.

    Raises:
    DeviceError: If the device count cannot be retrieved or if any device ID is invalid.
    """
    try:
        # Retrieve the list of valid device IDs by running the qaic-util command
        valid_devices = [
            d.strip() for d in os.popen('/opt/qti-aic/tools/qaic-util -q | grep "QID"').readlines()
        ]
        device_count = len(valid_devices)
    except Exception as e:
        # Raise an error if the device count cannot be retrieved
        raise DeviceError(
            'Failed to get Device Count. Check if devices are connected and Platform SDK is installed.'
        ) from e

    # Validate each provided device ID
    for dev_id in device_ids:
        if f'QID {dev_id}' not in valid_devices:
            # Raise an error if any device ID is invalid
            raise DeviceError(f'Invalid Device ID(s) passed. Device used must be one of '
                              f'{", ".join(valid_devices)}')

    return True
