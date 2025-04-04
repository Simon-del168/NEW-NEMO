# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from enum import Enum

# Dictionary qnn_datatype_to_size
qnn_datatype_to_size = {
    "Qnn_DataType_t.QNN_DATATYPE_INT_8": 8,
    "Qnn_DataType_t.QNN_DATATYPE_INT_16": 16,
    "Qnn_DataType_t.QNN_DATATYPE_INT_32": 32,
    "Qnn_DataType_t.QNN_DATATYPE_INT_64": 64,
    "Qnn_DataType_t.QNN_DATATYPE_UINT_8": 8,
    "Qnn_DataType_t.QNN_DATATYPE_UINT_16": 16,
    "Qnn_DataType_t.QNN_DATATYPE_UINT_32": 32,
    "Qnn_DataType_t.QNN_DATATYPE_UINT_64": 64,
    "Qnn_DataType_t.QNN_DATATYPE_FLOAT_16": 16,
    "Qnn_DataType_t.QNN_DATATYPE_FLOAT_32": 32,
    "Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8": 8,
    "Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_16": 16,
    "Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_32": 32,
    "Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_8": 8,
    "Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_16": 16,
    "Qnn_DataType_t.QNN_DATATYPE_UFIXED_POINT_32": 32,
    "Qnn_DataType_t.QNN_DATATYPE_BOOL_8": 8,
    "Qnn_DataType_t.QNN_DATATYPE_STRING": 32
    # Assuming string as 32 bits for calculation,
    # system does not allocate any fixed memory for the string itself. It points to actual string.
}


class MaxLimits(Enum):
    max_file_name_size = 255
    max_model_size_with_intermediates = 3840  #Model has to fit in one DSP of 3.75 GB = 3840 MB


class Engine(Enum):
    SNPE = 'SNPE'
    ANN = 'ANN'
    QNN = 'QNN'
    QAIRT = 'QAIRT'


class Framework(Enum):
    tensorflow = 'tensorflow'
    tflite = 'tflite'
    onnx = 'onnx'
    pytorch = 'pytorch'


class FrameworkExtension():
    framework_extension_mapping = {
        'tensorflow': '.pb',
        'tflite': '.tflite',
        'onnx': '.onnx',
        'pytorch': '.pt'
    }


class Runtime(Enum):
    cpu = 'cpu'
    gpu = 'gpu'
    dsp = 'dsp'
    dspv68 = 'dspv68'
    dspv69 = 'dspv69'
    dspv73 = 'dspv73'
    dspv75 = 'dspv75'
    dspv79 = 'dspv79'
    aic = 'aic'
    htp = 'htp'


class DebuggingAlgorithm(Enum):
    oneshot_layerwise = 'oneshot-layerwise'
    layerwise = 'layerwise'
    cumulative_layerwise = 'cumulative-layerwise'
    modeldissection = 'modeldissection'


class ComponentLogCodes(Enum):
    accuracy_debugger = "AD"
    framework_diagnosis = "AD-FD"
    inference_engine = "AD-IE"
    verification = "AD-VR"
    compare_encodings = "AD-CE"
    tensor_inspection = "AD-TI"
    quant_checker = "AD-QC"
    binary_snooping = "AD-BS"
    layerwise = "AD-LW"
    oneshot_layerwise = "AD-OLW"
    cumulative_layerwise = "AD-CLW"


class Android_Architectures(Enum):
    aarch64_android = 'aarch64-android'
    aarch64_android_clang6_0 = 'aarch64-android-clang6.0'
    aarch64_android_clang8_0 = 'aarch64-android-clang8.0'


class X86_Architectures(Enum):
    x86_64_linux_clang = 'x86_64-linux-clang'


class X86_windows_Architectures(Enum):
    x86_64_windows_msvc = 'x86_64-windows-msvc'


class Aarch64_windows_Architectures(Enum):
    wos = 'wos'


class Qnx_Architectures(Enum):
    aarch64_qnx = 'aarch64-qnx'


class Windows_Architectures(Enum):
    wos_remote = 'wos-remote'


class Architecture_Target_Types(Enum):
    # TODO: Fix the target arch name wos-remote to arm64x-windows once libs and bins are shipped in arm64x arch
    target_types = [
        'x86_64-linux-clang', 'aarch64-android', 'aarch64-qnx', 'wos-remote', 'x86_64-windows-msvc',
        'wos'
    ]


class ComponentType(Enum):
    converter = "converter"
    context_binary_generator = "context_binary_generator"
    x86_64_windows_context_binary_generator = "x86_64_windows_context_binary_generator"
    wos_context_binary_generator = "wos_context_binary_generator"
    executor = "executor"
    inference_engine = "inference_engine"
    devices = "devices"
    snpe_quantizer = "snpe_quantizer"
    quantizer = "quantizer"


class Status(Enum):
    off = "off"
    on = "on"
    always = "always"


class Devices_list(Enum):
    devices = [
        "linux-embedded", "android", "x86", "qnx", "wos-remote", "x86_64-windows-msvc", "wos"
    ]


class Device_type(Enum):
    linux_embedded = "linux-embedded"
    android = "android"
    x86 = "x86"
    qnx = "qnx"
    wos_remote = "wos-remote"
    x86_64_windows_msvc = "x86_64-windows-msvc"
    wos = "wos"


# host_device to architecture name
host_device_to_arch = {
    "android": "aarch64-android",
    "x86": "x86_64-linux-clang",
    "wos": "aarch64-windows-msvc",
    "wos-remote": "aarch64-windows-msvc",
    "x86_64-windows-msvc": "x86_64-windows-msvc",
    "qnx": "aarch64-qnx"
}


class AxisFormat(Enum):
    axis_format_mappings = {
        # (spatial-last format, spatial-first format): ((reshape order), (transpose order))
        ('NCDHW', 'NDHWC'): ((0, 4, 1, 2, 3), (0, 2, 3, 4, 1)),
        ('NCHW', 'NHWC'): ((0, 3, 1, 2), (0, 2, 3, 1)),
        ('NCF', 'NHWC'): ((0, 3, 1, 2), (0, 2, 3, 1)),
        ('NCDHW', 'NHWC'): ((0, 3, 1, 2), (0, 2, 3, 1)),
        ('NCDHW', 'NCHW'): ((0, 1, 2, 3), (0, 1, 2, 3)),
        ('NCF', 'NCHW'): ((0, 1, 2, 3), (0, 1, 2, 3)),
        ('NCS', 'NSC'): ((0, 3, 1, 2), (0, 2, 3, 1)),
        ('NCF', 'NFC'): ((0, 2, 1), (0, 2, 1)),
        ('TNF', 'NTF'): ((1, 0, 2), (1, 0, 2)),
        ('IODHW', 'DHWIO'): ((2, 3, 4, 0, 1), (2, 3, 4, 0, 1)),
        ('OIDHW', 'DHWOI'): ((3, 4, 0, 1, 2), (2, 3, 4, 1, 0)),
        ('OIDHW', 'DHWIO'): ((4, 3, 0, 1, 2), (0, 1, 2, 4, 3)),
        ('IOHW', 'HWIO'): ((2, 3, 0, 1), (2, 3, 0, 1)),
        ('OIHW', 'HWOI'): ((2, 3, 0, 1), (2, 3, 0, 1)),
        ('OIHW', 'HWIO'): ((3, 2, 0, 1), (2, 3, 1, 0)),
        ('IOF', 'FIO'): ((1, 2, 0), (2, 0, 1)),
        ('OIF', 'FOI'): ((1, 2, 0), (2, 0, 1)),
        ('OIF', 'FIO'): ((2, 1, 0), (2, 1, 0))
        # need to add below axis formats in mapping
        # NF: ir_graph.AxisFormat.NF,
        # NC: ir_graph.AxisFormat.NC,
        # ANY: ir_graph.AxisFormat.ANY,
        # NONTRIVIAL: ir_graph.AxisFormat.NONTRIVIAL,
    }


class SocTypes(Enum):
    RuntimeSocMap = {
        "dspv79": "sm8750",
        "dspv75": "sm8650",
        "dspv73": "sm8550",
        "dspv69": "sm8450",
        "dspv68": "sm8350"
    }
