# =============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from __future__ import annotations

import os
import sys
import json
import shutil
import struct
import tempfile
import numpy as np
from pathlib import Path
from io import BufferedWriter
from enum import IntEnum, auto
from typing import IO, Any, BinaryIO, Callable, Sequence

#
# constants
#

GGUF_MAGIC             = 0x46554747
GGUF_VERSION           = 2
GGUF_DEFAULT_ALIGNMENT = 32

# general
KEY_GENERAL_ARCHITECTURE         = "general.architecture"
KEY_GENERAL_QUANTIZATION_VERSION = "general.quantization_version"
KEY_GENERAL_ALIGNMENT            = "general.alignment"
KEY_GENERAL_NAME                 = "general.name"
KEY_GENERAL_TOKENIZER            = "general.tokenizer"
KEY_GENERAL_AUTHOR               = "general.author"
KEY_GENERAL_URL                  = "general.url"
KEY_GENERAL_DESCRIPTION          = "general.description"
KEY_GENERAL_LICENSE              = "general.license"
KEY_GENERAL_SOURCE_URL           = "general.source.url"
KEY_GENERAL_SOURCE_HF_REPO       = "general.source.hugginface.repository"
KEY_GENERAL_FILE_TYPE            = "general.file_type"

# LLM
KEY_VOCAB_SIZE            = "model.size.vocabulary"
KEY_CONNECTOR             = "model.architecture.connector"
KEY_ARCH_GATE             = "model.architecture.gating"
KEY_CONTEXT_LENGTH        = "{arch}.context_length"
KEY_EMBEDDING_LENGTH      = "{arch}.embedding_length"
KEY_BLOCK_COUNT           = "{arch}.block_count"
KEY_FEED_FORWARD_LENGTH   = "{arch}.feed_forward_length"
KEY_USE_PARALLEL_RESIDUAL = "{arch}.use_parallel_residual"
KEY_TENSOR_DATA_LAYOUT    = "{arch}.tensor_data_layout"

# Operation
KEY_OPERATION_NORMALIZATION         = "model.operation.normalization"
KEY_OPERATION_ACTIVATION            = "model.operation.activation"
KEY_OPERATION_POSITIONAL_EMBEDDING  = "model.operation.positional_embedding"
KEY_OPERATION_ROPE_COMPLEX_ORG      = "model.operation.rope_complex_organization"
KEY_OPERATION_NORMALIZATION_EPS     = "model.operation.normalization_epsilon"

# attention
KEY_ATTENTION_HEAD_COUNT        = "{arch}.attention.head_count"
KEY_ATTENTION_HEAD_COUNT_KV     = "{arch}.attention.head_count_kv"
KEY_ATTENTION_MAX_ALIBI_BIAS    = "{arch}.attention.max_alibi_bias"
KEY_ATTENTION_CLAMP_KQV         = "{arch}.attention.clamp_kqv"
KEY_ATTENTION_LAYERNORM_EPS     = "{arch}.attention.layer_norm_epsilon"

# RoPE
KEY_ROPE_NUM_ROTATION    = "{arch}.rope.dimension_count"
KEY_ROPE_FREQ_BASE       = "{arch}.rope.freq_base"
KEY_ROPE_SCALE_LINEAR    = "{arch}.rope.scale_linear"


# tokenization
KEY_TOKENIZER_MODEL      = "tokenizer.ggml.model"
KEY_TOKENIZER_LIST       = "tokenizer.ggml.tokens"
KEY_TOKENIZER_TOKEN_TYPE = "tokenizer.ggml.token_type"
KEY_TOKENIZER_SCORES     = "tokenizer.ggml.scores"
KEY_TOKENIZER_MERGES     = "tokenizer.ggml.merges"
KEY_TOKENIZER_BOS_ID     = "tokenizer.ggml.bos_token_id"
KEY_TOKENIZER_EOS_ID     = "tokenizer.ggml.eos_token_id"
KEY_TOKENIZER_UNK_ID     = "tokenizer.ggml.unknown_token_id"
KEY_TOKENIZER_SEP_ID     = "tokenizer.ggml.seperator_token_id"
KEY_TOKENIZER_PAD_ID     = "tokenizer.ggml.padding_token_id"
KEY_TOKENIZER_HF_JSON    = "tokenizer.huggingface.json"
KEY_TOKENIZER_RWKV       = "tokenizer.rwkv.world"


#
# recommended mapping of model tensor names for storage in gguf
#

class MODEL_TENSOR(IntEnum):
    TOKEN_EMBD          : int = auto()
    POS_EMBD            : int = auto()
    OUTPUT              : int = auto()
    OUTPUT_BIAS         : int = auto()
    OUTPUT_NORM         : int = auto()
    OUTPUT_NORM_BIAS    : int = auto()
    ATTN_Q              : int = auto()
    ATTN_Q_BIAS         : int = auto()
    ATTN_K              : int = auto()
    ATTN_K_BIAS         : int = auto()
    ATTN_V              : int = auto()
    ATTN_V_BIAS         : int = auto()
    ATTN_QKV            : int = auto()
    ATTN_QKV_BIAS       : int = auto()
    ATTN_OUT            : int = auto()
    ATTN_OUT_BIAS       : int = auto()
    ATTN_NORM           : int = auto()
    ATTN_NORM_BIAS      : int = auto()
    ATTN_NORM_2         : int = auto()
    ATTN_NORM_2_BIAS    : int = auto()
    ATTN_MASK           : int = auto()
    ATTN_MASK_BIAS      : int = auto()
    FFN_GATE            : int = auto()
    FFN_GATE_BIAS       : int = auto()
    FFN_DOWN            : int = auto()
    FFN_DOWN_BIAS       : int = auto()
    FFN_UP              : int = auto()
    FFN_UP_BIAS         : int = auto()
    FFN_NORM            : int = auto()
    FFN_NORM_BIAS       : int = auto()

MODEL_TENSOR_NAMES: dict[MODEL_TENSOR, str] = {
        MODEL_TENSOR.TOKEN_EMBD         : "token_embd.weight",
        MODEL_TENSOR.POS_EMBD           : "token_embd_pos.weight",
        MODEL_TENSOR.OUTPUT_NORM        : "output_norm.weight",
        MODEL_TENSOR.OUTPUT_NORM_BIAS   : "output_norm.bias",
        MODEL_TENSOR.OUTPUT             : "output.weight",
        MODEL_TENSOR.OUTPUT_BIAS        : "output.bias",
        MODEL_TENSOR.ATTN_NORM          : "blk.{bid}.attn_norm.weight",
        MODEL_TENSOR.ATTN_NORM_BIAS     : "blk.{bid}.attn_norm.bias",
        MODEL_TENSOR.ATTN_Q             : "blk.{bid}.attn_q.weight",
        MODEL_TENSOR.ATTN_Q_BIAS        : "blk.{bid}.attn_q.bias",
        MODEL_TENSOR.ATTN_K             : "blk.{bid}.attn_k.weight",
        MODEL_TENSOR.ATTN_K_BIAS        : "blk.{bid}.attn_k.bias",
        MODEL_TENSOR.ATTN_V             : "blk.{bid}.attn_v.weight",
        MODEL_TENSOR.ATTN_V_BIAS        : "blk.{bid}.attn_v.bias",
        MODEL_TENSOR.ATTN_QKV           : "blk.{bid}.attn_qkv.weight",
        MODEL_TENSOR.ATTN_QKV_BIAS      : "blk.{bid}.attn_qkv.bias",
        MODEL_TENSOR.ATTN_OUT           : "blk.{bid}.attn_output.weight",
        MODEL_TENSOR.ATTN_OUT_BIAS      : "blk.{bid}.attn_output.bias",
        MODEL_TENSOR.FFN_NORM           : "blk.{bid}.ffn_norm.weight",
        MODEL_TENSOR.FFN_NORM_BIAS      : "blk.{bid}.ffn_norm.bias",
        MODEL_TENSOR.FFN_GATE           : "blk.{bid}.ffn_gate.weight",
        MODEL_TENSOR.FFN_GATE_BIAS      : "blk.{bid}.ffn_gate.bias",
        MODEL_TENSOR.FFN_DOWN           : "blk.{bid}.ffn_down.weight",
        MODEL_TENSOR.FFN_DOWN_BIAS      : "blk.{bid}.ffn_down.bias",
        MODEL_TENSOR.FFN_UP             : "blk.{bid}.ffn_up.weight",
        MODEL_TENSOR.FFN_UP_BIAS        : "blk.{bid}.ffn_up.bias",
}

CONFIG_TENSOR_NAMES: dict[str, MODEL_TENSOR] = {
    "tensor.embedding_token_weight"             : MODEL_TENSOR.TOKEN_EMBD,
    "tensor.embedding_position_weight"          : MODEL_TENSOR.POS_EMBD,
    "tensor.output_weight"                      : MODEL_TENSOR.OUTPUT,
    "tensor.output_bias"                        : MODEL_TENSOR.OUTPUT_BIAS,
    "tensor.output_normalization_weight"        : MODEL_TENSOR.OUTPUT_NORM,
    "tensor.output_normalization_bias"          : MODEL_TENSOR.OUTPUT_NORM_BIAS,
    "tensor.attention_normalization_weight"     : MODEL_TENSOR.ATTN_NORM,
    "tensor.attention_normalization_bias"       : MODEL_TENSOR.ATTN_NORM_BIAS,
    "tensor.attention_qkv_weight"               : MODEL_TENSOR.ATTN_QKV,
    "tensor.attention_qkv_bias"                 : MODEL_TENSOR.ATTN_QKV_BIAS,
    "tensor.attention_q_weight"                 : MODEL_TENSOR.ATTN_Q,
    "tensor.attention_q_bias"                   : MODEL_TENSOR.ATTN_Q_BIAS,
    "tensor.attention_k_weight"                 : MODEL_TENSOR.ATTN_K,
    "tensor.attention_k_bias"                   : MODEL_TENSOR.ATTN_K_BIAS,
    "tensor.attention_v_weight"                 : MODEL_TENSOR.ATTN_V,
    "tensor.attention_v_bias"                   : MODEL_TENSOR.ATTN_V_BIAS,
    "tensor.attention_output_weight"            : MODEL_TENSOR.ATTN_OUT,
    "tensor.attention_output_bias"              : MODEL_TENSOR.ATTN_OUT_BIAS,
    "tensor.feedforward_normalization_weight"   : MODEL_TENSOR.FFN_NORM,
    "tensor.feedforward_normalization_bias"     : MODEL_TENSOR.FFN_NORM_BIAS,
    "tensor.feedforward_gate_weight"            : MODEL_TENSOR.FFN_GATE,
    "tensor.feedforward_gate_bias"              : MODEL_TENSOR.FFN_GATE_BIAS,
    "tensor.feedforward_up_weight"              : MODEL_TENSOR.FFN_UP,
    "tensor.feedforward_up_bias"                : MODEL_TENSOR.FFN_UP_BIAS,
    "tensor.feedforward_output_weight"          : MODEL_TENSOR.FFN_DOWN,
    "tensor.feedforward_output_bias"            : MODEL_TENSOR.FFN_DOWN_BIAS,
}

NON_LAYER_NAMES = [
    "tensor.output_normalization_weight",
    "tensor.output_normalization_bias",
    "tensor.output_weight",
    "tensor.output_bias",
    "tensor.embedding_position_weight",
    "tensor.embedding_token_weight",
]

NEEDS_TRANSPOSE = [
    "tensor.attention_qkv_weight",
    "tensor.attention_q_weight",
    "tensor.attention_k_weight",
    "tensor.attention_v_weight",
    "tensor.attention_output_weight",
    "tensor.feedforward_up_weight",
    "tensor.feedforward_gate_weight",
    "tensor.feedforward_output_weight",
    "tensor.feedforward_down_weight",
    "tensor.output_weight",
]

class TokenType(IntEnum):
    NORMAL       = 1
    UNKNOWN      = 2
    CONTROL      = 3
    USER_DEFINED = 4
    UNUSED       = 5
    BYTE         = 6

#
# implementation
#


class GGMLQuantizationType(IntEnum):
    F32      = 0
    F16      = 1
    Q4_0     = 2
    Q4_1     = 3
    Q5_0     = 6
    Q5_1     = 7
    Q8_0     = 8
    Q8_1     = 9
    Q2_K     = 10
    Q3_K     = 11
    Q4_K     = 12
    Q5_K     = 13
    Q6_K     = 14
    Q8_K     = 15
    Z4       = 20
    Z4_FP16  = 21
    Z4_BF16  = 22
    BFloat16 = 201


class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

    @staticmethod
    def get_type(val):
        if isinstance(val, str) or isinstance(val, bytes) or isinstance(val, bytearray):
            return GGUFValueType.STRING
        elif isinstance(val, list):
            return GGUFValueType.ARRAY
        elif isinstance(val, float):
            return GGUFValueType.FLOAT32
        elif isinstance(val, bool):
            return GGUFValueType.BOOL
        elif isinstance(val, int):
            return GGUFValueType.INT32
        # TODO: need help with 64-bit types in Python
        else:
            print("Unknown type: "+str(type(val)))
            sys.exit()


class GGUFWriter:
    fout: BufferedWriter
    arch: str
    offset_tensor = 0
    data_alignment = GGUF_DEFAULT_ALIGNMENT
    kv_data = b""
    kv_data_count = 0
    ti_data = b""
    ti_data_count = 0
    use_temp_file: bool
    temp_file: tempfile.SpooledTemporaryFile[bytes] | None = None
    tensors: list[tuple[np.ndarray[Any, Any], int]]

    def __init__(self, path: os.PathLike[str] | str, alignment: int | None=None, arch: str | None = None, use_temp_file = True):
        self.fout = open(path, "wb")
        if arch:
            self.arch = arch
        self.use_temp_file = use_temp_file
        self.tensors = []
        if alignment:
            self.data_alignment = alignment

    def write_header_to_file(self):
        self.fout.write(struct.pack("<I", GGUF_MAGIC))
        self.fout.write(struct.pack("<I", GGUF_VERSION))
        self.fout.write(struct.pack("<Q", self.ti_data_count))
        self.fout.write(struct.pack("<Q", self.kv_data_count))
        self.flush()

    def write_kv_data_to_file(self):
        self.fout.write(self.kv_data)
        self.flush()

    def write_ti_data_to_file(self):
        self.fout.write(self.ti_data)
        self.flush()

    def add_key(self, key: str):
        self.add_val(key, GGUFValueType.STRING, add_vtype=False)

    def add_uint8(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.UINT8)

    def add_int8(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.INT8)

    def add_uint16(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.UINT16)

    def add_int16(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.INT16)

    def add_uint32(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.UINT32)

    def add_int32(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.INT32)

    def add_float32(self, key: str, val: float):
        self.add_key(key)
        self.add_val(val, GGUFValueType.FLOAT32)

    def add_uint64(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.UINT64)

    def add_int64(self, key: str, val: int):
        self.add_key(key)
        self.add_val(val, GGUFValueType.INT64)

    def add_float64(self, key: str, val: float):
        self.add_key(key)
        self.add_val(val, GGUFValueType.FLOAT64)

    def add_bool(self, key: str, val: bool):
        self.add_key(key)
        self.add_val(val, GGUFValueType.BOOL)

    def add_string(self, key: str, val: str):
        if len(val) == 0:
            return
        self.add_key(key)
        self.add_val(val, GGUFValueType.STRING)

    def add_array(self, key: str, val: Sequence[Any]):
        if not isinstance(val, Sequence):
            raise ValueError("Value must be a sequence for array type")

        self.add_key(key)
        self.add_val(val, GGUFValueType.ARRAY)

    _simple_value_packing = {
        GGUFValueType.UINT8:   "<B",
        GGUFValueType.INT8:    "<b",
        GGUFValueType.UINT16:  "<H",
        GGUFValueType.INT16:   "<h",
        GGUFValueType.UINT32:  "<I",
        GGUFValueType.INT32:   "<i",
        GGUFValueType.FLOAT32: "<f",
        GGUFValueType.UINT64:  "<Q",
        GGUFValueType.INT64:   "<q",
        GGUFValueType.FLOAT64: "<d",
        GGUFValueType.BOOL:    "?" ,
    }
    def add_val(self, val: Any, vtype: GGUFValueType | None = None, add_vtype: bool = True):
        if vtype is None:
            vtype = GGUFValueType.get_type(val)

        if add_vtype:
            self.kv_data += struct.pack("<I", vtype)
            self.kv_data_count += 1

        pack_fmt = self._simple_value_packing.get(vtype)
        if pack_fmt is not None:
            #print(pack_fmt, val)
            self.kv_data += struct.pack(pack_fmt, val)
        elif vtype == GGUFValueType.STRING:
            encoded_val = val.encode("utf8") if isinstance(val, str) else val
            self.kv_data += struct.pack("<Q", len(encoded_val))
            self.kv_data += encoded_val
        elif vtype == GGUFValueType.ARRAY and isinstance(val, Sequence) and len(val) > 0:
            ltype = GGUFValueType.get_type(val[0])
            if not all(GGUFValueType.get_type(i) is ltype for i in val[1:]):
                raise ValueError("All items in a GGUF array should be of the same type")
            self.kv_data += struct.pack("<I", ltype)
            self.kv_data += struct.pack("<Q", len(val))
            for item in val:
                self.add_val(item, add_vtype=False)
        else:
            raise ValueError("Invalid GGUF metadata value type or value")

    @staticmethod
    def ggml_pad(x: int, n: int) -> int:
        return ((x + n - 1) // n) * n

    def add_tensor_info(self, name: str, tensor_shape: Sequence[int], tensor_dtype: np.dtype[np.float16] | np.dtype[np.float32], tensor_nbytes: int, raw_dtype: GGMLQuantizationType | None = None):
        assert raw_dtype is not None or tensor_dtype in (np.float32, np.float16), "Only F32 and F16 tensors are supported for now"
        encoded_name = name.encode("utf8")
        self.ti_data += struct.pack("<Q", len(encoded_name))
        self.ti_data += encoded_name
        n_dims = len(tensor_shape)
        self.ti_data += struct.pack("<I", n_dims)
        for i in range(n_dims):
            self.ti_data += struct.pack("<Q", tensor_shape[n_dims - 1 - i])
        if raw_dtype is None:
            dtype = GGMLQuantizationType.F32 if tensor_dtype == np.float32 else GGMLQuantizationType.F16
        else:
            dtype = raw_dtype
        self.ti_data += struct.pack("<I", dtype)
        self.ti_data += struct.pack("<Q", self.offset_tensor)
        self.offset_tensor += GGUFWriter.ggml_pad(tensor_nbytes, self.data_alignment)
        self.ti_data_count += 1

    def add_tensor_info_from_json(self, name: str, tensor_shape: Sequence[int], tensor_offset: int, raw_dtype: GGMLQuantizationType | None = None):
        assert raw_dtype is not None
        encoded_name = name.encode("utf8")
        self.ti_data += struct.pack("<Q", len(encoded_name))
        self.ti_data += encoded_name
        n_dims = len(tensor_shape)
        self.ti_data += struct.pack("<I", n_dims)
        for i in range(n_dims):
            self.ti_data += struct.pack("<Q", tensor_shape[n_dims - 1 - i])
        dtype = raw_dtype
        self.ti_data += struct.pack("<I", dtype)
        self.ti_data += struct.pack("<Q", tensor_offset)
        self.offset_tensor = tensor_offset
        self.ti_data_count += 1

    def add_tensor(self, name: str, tensor: np.ndarray[Any, Any], raw_shape: Sequence[int] | None = None, raw_dtype: GGMLQuantizationType | None = None):
        if self.use_temp_file and self.temp_file is None:
            fp = tempfile.SpooledTemporaryFile(mode="w+b", max_size=256*1024*1024)
            fp.seek(0)
            self.temp_file = fp

        shape: Sequence[int] = raw_shape if raw_shape is not None else tensor.shape
        self.add_tensor_info(name, shape, tensor.dtype, tensor.nbytes, raw_dtype = raw_dtype)

        pad = GGUFWriter.ggml_pad(tensor.nbytes, self.data_alignment) - tensor.nbytes

        if  self.temp_file is None:
            self.tensors.append((tensor, pad))
            return

        tensor.tofile(self.temp_file)

        if pad != 0:
            self.temp_file.write(bytes([0] * pad))

    def write_padding(self, fp: BinaryIO, n: int, align: int | None = None):
        pad = GGUFWriter.ggml_pad(n, align if align is not None else self.data_alignment) - n
        if pad != 0:
            fp.write(bytes([0] * pad))

    def write_tensor_data(self, tensor: np.ndarray[Any, Any]):
        self.write_padding(self.fout, self.fout.tell())
        tensor.tofile(self.fout)
        self.write_padding(self.fout, tensor.nbytes)

    def write_tensors_to_file(self):
        self.write_ti_data_to_file()

        self.write_padding(self.fout, self.fout.tell())

        if self.temp_file is None:
            for (currtensor, currpad) in self.tensors:
                currtensor.tofile(self.fout)
                if currpad != 0:
                    self.fout.write(bytes([0] * currpad))
            return

        self.temp_file.seek(0)

        shutil.copyfileobj(self.temp_file, self.fout)
        self.flush()
        self.temp_file.close()

    def flush(self):
        self.fout.flush()

    def close(self):
        self.fout.close()

    def add_architecture(self):
        self.add_string(KEY_GENERAL_ARCHITECTURE, self.arch)

    def add_arch(self, architecture: str):
        self.add_string(KEY_GENERAL_ARCHITECTURE, architecture)

    def add_author(self, author: str):
        self.add_string(KEY_GENERAL_AUTHOR, author)

    def add_tensor_data_layout(self, layout: str):
        self.add_string(KEY_TENSOR_DATA_LAYOUT.format(arch=self.arch), layout)

    def add_url(self, url: str):
        self.add_string(KEY_GENERAL_URL, url)

    def add_description(self, description: str):
        self.add_string(KEY_GENERAL_DESCRIPTION, description)

    def add_source_url(self, url: str):
        self.add_string(KEY_GENERAL_SOURCE_URL, url)

    def add_source_hf_repo(self, repo: str):
        self.add_string(KEY_GENERAL_SOURCE_HF_REPO, repo)

    def add_file_type(self, ftype: int):
        self.add_uint32(KEY_GENERAL_FILE_TYPE, ftype)

    def add_name(self, name: str):
        self.add_string(KEY_GENERAL_NAME, name)

    def add_tokenizer(self, tokenizer: str):
        self.add_string(KEY_GENERAL_TOKENIZER, tokenizer)

    def add_quantization_version(self, quantization_version: GGMLQuantizationType):
        self.add_uint32(
            KEY_GENERAL_QUANTIZATION_VERSION, quantization_version)

    def add_custom_alignment(self, alignment: int):
        self.data_alignment = alignment
        self.add_uint32(KEY_GENERAL_ALIGNMENT, alignment)

    def add_context_length(self, length: int):
        self.add_uint32(
            KEY_CONTEXT_LENGTH.format(arch=self.arch), length)

    def add_embedding_length(self, length: int):
        self.add_uint32(
            KEY_EMBEDDING_LENGTH.format(arch=self.arch), length)

    def add_block_count(self, length: int):
        self.add_uint32(
            KEY_BLOCK_COUNT.format(arch=self.arch), length)

    def add_feed_forward_length(self, length: int):
        self.add_uint32(
            KEY_FEED_FORWARD_LENGTH.format(arch=self.arch), length)

    def add_parallel_residual(self, use: bool):
        self.add_bool(
            KEY_USE_PARALLEL_RESIDUAL.format(arch=self.arch), use)

    def add_head_count(self, count: int):
        self.add_uint32(
            KEY_ATTENTION_HEAD_COUNT.format(arch=self.arch), count)

    def add_head_count_kv(self, count: int):
        self.add_uint32(
            KEY_ATTENTION_HEAD_COUNT_KV.format(arch=self.arch), count)

    def add_max_alibi_bias(self, bias: float):
        self.add_float32(
            KEY_ATTENTION_MAX_ALIBI_BIAS.format(arch=self.arch), bias)

    def add_clamp_kqv(self, value: float):
        self.add_float32(
            KEY_ATTENTION_CLAMP_KQV.format(arch=self.arch), value)

    def add_layer_norm_eps(self, value: float):
        self.add_float32(
            KEY_ATTENTION_LAYERNORM_EPS.format(arch=self.arch), value)

    def add_layer_norm_rms_eps(self, value: float):
        self.add_float32(
            KEY_OPERATION_NORMALIZATION_EPS, value)

    def add_num_rotations(self, count: int):
        self.add_uint32(
            KEY_ROPE_NUM_ROTATION.format(arch=self.arch), count)

    def add_rope_freq_base(self, value: float):
        self.add_float32(KEY_ROPE_FREQ_BASE.format(arch=self.arch), value)

    def add_rope_scale_linear(self, value: float):
        self.add_float32(KEY_ROPE_SCALE_LINEAR.format(arch=self.arch), value)

    def add_tokenizer_model(self, model: str):
        self.add_string(KEY_TOKENIZER_MODEL, model)

    def add_token_list(self, tokens: Sequence[str] | Sequence[bytes] | Sequence[bytearray]):
        self.add_array(KEY_TOKENIZER_LIST, tokens)

    def add_token_merges(self, merges: Sequence[str] | Sequence[bytes] | Sequence[bytearray]):
        self.add_array(KEY_TOKENIZER_MERGES, merges)

    def add_token_types(self, types: Sequence[TokenType] | Sequence[int]):
        self.add_array(KEY_TOKENIZER_TOKEN_TYPE, types)

    def add_token_scores(self, scores: Sequence[float]):
        self.add_array(KEY_TOKENIZER_SCORES, scores)

    def add_bos_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_BOS_ID, id)

    def add_eos_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_EOS_ID, id)

    def add_unk_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_UNK_ID, id)

    def add_sep_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_SEP_ID, id)

    def add_pad_token_id(self, id: int):
        self.add_uint32(KEY_TOKENIZER_PAD_ID, id)

    def add_vocab_size(self, vocab: int):
        self.add_int32(KEY_VOCAB_SIZE, vocab)

    def add_connector(self, connector: str):
        self.add_string(KEY_CONNECTOR, connector)

    def add_gating(self, gate: str):
        self.add_string(KEY_ARCH_GATE, gate)

    def add_normalization(self, norm: str):
        self.add_string(KEY_OPERATION_NORMALIZATION, norm)

    def add_activation(self, act: str):
        self.add_string(KEY_OPERATION_ACTIVATION, act)

    def add_pos_embd(self, pos_embd: str):
        self.add_string(KEY_OPERATION_POSITIONAL_EMBEDDING, pos_embd)

    def add_complex_org(self, comp_org: str):
        self.add_string(KEY_OPERATION_ROPE_COMPLEX_ORG, comp_org)

class SpecialVocab:
    load_merges: bool = False
    merges: list[str] = []
    special_token_types: tuple[str, ...] = ('bos', 'eos', 'unk', 'sep', 'pad')
    special_token_ids: dict[str, int] = {}

    def __init__(self, path: Path, load_merges: bool = False, special_token_types: tuple[str, ...] | None = None):
        self.special_token_ids = {}
        self.load_merges = load_merges
        if special_token_types is not None:
            self.special_token_types = special_token_types
        self.load(path)

    def load(self, path: Path):
        if not self.try_load_from_tokenizer_json(path):
            self.try_load_from_config_json(path)

    def try_load_from_tokenizer_json(self, path: Path) -> bool:
        tokenizer_file = path / 'tokenizer.json'
        if not tokenizer_file.is_file():
            from qti.aisw.genai import qnn_genai_transformer_tokenizer
            tokenizer_file = path / 'qwen.tiktoken'
            if not tokenizer_file.is_file():
                return False
            tokenizer = qnn_genai_transformer_tokenizer.QwenTokenizer(path)._create_qwen_bpe()
        else:
            with open(tokenizer_file, 'r', encoding = 'utf-8') as f:
                tokenizer = json.load(f)
        if self.load_merges:
            merges = tokenizer.get('model', {}).get('merges')
            if isinstance(merges, list) and len(merges) > 0 and isinstance(merges[0], str):
                self.merges = merges
        tokenizer_config_file = path / 'tokenizer_config.json'
        added_tokens = tokenizer.get('added_tokens')
        if added_tokens is None or not tokenizer_config_file.is_file():
            return True
        with open(tokenizer_config_file, 'r', encoding = 'utf-8') as f:
            tokenizer_config = json.load(f)
        for typ in self.special_token_types:
            entry = tokenizer_config.get(f'{typ}_token')
            if isinstance(entry, str):
                tc_content = entry
            elif isinstance(entry, dict):
                entry_content = entry.get('content')
                if not isinstance(entry_content, str):
                    continue
                tc_content = entry_content
            else:
                continue
            for maybe_token_id in (atok.get('id') for atok in added_tokens if atok.get('content') == tc_content):
                if isinstance(maybe_token_id, int):
                    self.special_token_ids[typ] = maybe_token_id
                break
        return True

    def try_load_from_config_json(self, path: Path) -> bool:
        config_file = path / 'config.json'
        if not config_file.is_file():
            return False
        with open(config_file, 'r', encoding = 'utf-8') as f:
            config = json.load(f)
        for typ in self.special_token_types:
            maybe_token_id = config.get(f'{typ}_token_id')
            if isinstance(maybe_token_id, int) and maybe_token_id >= 0:
                self.special_token_ids[typ] = maybe_token_id
        return True

    def add_to_gguf(self, gw: GGUFWriter):
        if len(self.merges) > 0:
            print(f'gguf: Adding {len(self.merges)} merge(s).')
            gw.add_token_merges(self.merges)
        for typ, tokid in self.special_token_ids.items():
            handler: Callable[[int], None] | None = getattr(gw, f'add_{typ}_token_id', None)
            if handler is None:
                print(f'gguf: WARNING: No handler for special token type {typ} with id {tokid} - skipping')
                continue
            print(f'gguf: Setting special token type {typ} to {tokid}')
            handler(tokid)

    def __repr__(self):
        return f'<SpecialVocab with {len(self.merges)} merges and special tokens {self.special_token_ids if self.special_token_ids else "unset"}>'
