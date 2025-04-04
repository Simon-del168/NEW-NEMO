#
# Copyright (C) 2016 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Copyright (C) 2016 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

WORKING_DIR := $(call my-dir)
$(info WORKING_DIR: $(WORKING_DIR))
# build libvpx.so
include $(CLEAR_VARS)
LOCAL_PATH := $(WORKING_DIR)
$(info LOCAL_PATH for libvpx: $(LOCAL_PATH))
include $(LOCAL_PATH)/libvpx.mk

# build libwebm.so
include $(CLEAR_VARS)
LOCAL_PATH := $(WORKING_DIR)
include $(LOCAL_PATH)/libvpx/third_party/libwebm/Android.mk

# build vpxdec
include $(CLEAR_VARS)
LOCAL_PATH := $(WORKING_DIR)

VPXDEC_SRCS := libvpx/vpxdec_nemo_ver2.c \
			   libvpx/md5_utils.c \
			   libvpx/args.c \
			   libvpx/ivfdec.c \
	 		   libvpx/tools_common.c \
			   libvpx/y4menc.c \
			   libvpx/webmdec.cc 

LIBYUV_SRCS :=  libvpx/third_party/libyuv/source/convert_argb.cc \
                libvpx/third_party/libyuv/source/convert_from_argb.cc \
                libvpx/third_party/libyuv/source/convert_from.cc \
                libvpx/third_party/libyuv/source/convert_to_argb.cc \
                libvpx/third_party/libyuv/source/convert_to_i420.cc \
                libvpx/third_party/libyuv/source/convert.cc \
                libvpx/third_party/libyuv/source/cpu_id.cc \
                libvpx/third_party/libyuv/source/planar_functions.cc \
                libvpx/third_party/libyuv/source/row_any.cc \
                libvpx/third_party/libyuv/source/row_common.cc \
                libvpx/third_party/libyuv/source/row_gcc.cc \
                libvpx/third_party/libyuv/source/row_mips.cc \
                libvpx/third_party/libyuv/source/row_neon.cc \
                libvpx/third_party/libyuv/source/row_neon64.cc \
                libvpx/third_party/libyuv/source/row_win.cc \
                libvpx/third_party/libyuv/source/scale.cc \
                libvpx/third_party/libyuv/source/scale_any.cc \
                libvpx/third_party/libyuv/source/scale_common.cc \
                libvpx/third_party/libyuv/source/scale_gcc.cc \
                libvpx/third_party/libyuv/source/scale_mips.cc \
                libvpx/third_party/libyuv/source/scale_neon.cc \
                libvpx/third_party/libyuv/source/scale_neon64.cc \
                libvpx/third_party/libyuv/source/scale_win.cc \
                libvpx/third_party/libyuv/source/video_common.cc

LOCAL_MODULE := vpxdec_nemo_ver2
LOCAL_ARM_MODE := arm
LOCAL_CPP_EXTENSION := .cc .cpp
CONFIG_DIR := $(LOCAL_PATH)/libvpx_android_configs/$(TARGET_ARCH_ABI)
LOCAL_C_INCLUDES := $(CONFIG_DIR)
LOCAL_C_INCLUDES += $(LOCAL_PATH)/libvpx/third_party/libyuv/include/
LOCAL_C_INCLUDES += $(LOCAL_PATH)/libvpx/vp9/encoder/
#LOCAL_C_INCLUDES += $(LOCAL_PATH)/snpe/include/SNPE
LOCAL_C_INCLUDES += /workspace/nemo/third_party/snpe/include/SNPE
LOCAL_C_INCLUDES += /workspace/nemo/third_party/snpe/include/SNPE/DlSystem
LOCAL_SRC_FILES := $(VPXDEC_SRCS) 
LOCAL_SRC_FILES += $(LIBYUV_SRCS) 
LOCAL_SRC_FILES += /root/android-ndk-r26c/sources/android/cpufeatures/cpu-features.c

LOCAL_CFLAGS += -mfpu=neon
#LOCAL_CFLAGS += -std=c++11
#LOCAL_CFLAGS += -g
LOCAL_CFLAGS := -g -O0
LOCAL_CPPFLAGS := -g -O0

LOCAL_LDLIBS := -llog -lz -lm -landroid
LOCAL_SHARED_LIBRARIES := libvpx 
LOCAL_STATIC_LIBRARIES := cpufeatures libwebm
include $(BUILD_EXECUTABLE)

$(call import-module,android/cpufeatures)
