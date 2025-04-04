##
##  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##


API_EXPORTS += exports

API_SRCS-$(CONFIG_VP8_ENCODER) += vp8.h
API_SRCS-$(CONFIG_VP8_ENCODER) += vp8cx.h
API_DOC_SRCS-$(CONFIG_VP8_ENCODER) += vp8.h
API_DOC_SRCS-$(CONFIG_VP8_ENCODER) += vp8cx.h
ifeq ($(CONFIG_VP9_ENCODER),yes)
  API_SRCS-$(CONFIG_SPATIAL_SVC) += src/svc_encodeframe.c
  API_SRCS-$(CONFIG_SPATIAL_SVC) += svc_context.h
endif

API_SRCS-$(CONFIG_VP8_DECODER) += vp8.h
API_SRCS-$(CONFIG_VP8_DECODER) += vp8dx.h
API_DOC_SRCS-$(CONFIG_VP8_DECODER) += vp8.h
API_DOC_SRCS-$(CONFIG_VP8_DECODER) += vp8dx.h

API_DOC_SRCS-yes += vpx_codec.h
API_DOC_SRCS-yes += vpx_decoder.h
API_DOC_SRCS-yes += vpx_encoder.h
API_DOC_SRCS-yes += vpx_frame_buffer.h
API_DOC_SRCS-yes += vpx_image.h

API_SRCS-yes += src/vpx_decoder.c
API_SRCS-yes += vpx_decoder.h
API_SRCS-yes += src/vpx_encoder.c
API_SRCS-yes += vpx_encoder.h
API_SRCS-yes += internal/vpx_codec_internal.h
API_SRCS-yes += src/vpx_codec.c
API_SRCS-yes += src/vpx_image.c
API_SRCS-yes += vpx_codec.h
API_SRCS-yes += vpx_codec.mk
API_SRCS-yes += vpx_frame_buffer.h
API_SRCS-yes += vpx_image.h
API_SRCS-yes += vpx_integer.h

API_SRCS-$(CONFIG_SNPE) += snpe/CheckRuntime.cpp
API_SRCS-$(CONFIG_SNPE) += snpe/CheckRuntime.hpp
API_SRCS-$(CONFIG_SNPE) += snpe/CreateGLBuffer.cpp
API_SRCS-$(CONFIG_SNPE) += snpe/CreateGLBuffer.hpp
API_SRCS-$(CONFIG_SNPE) += snpe/CreateGLContext.cpp
API_SRCS-$(CONFIG_SNPE) += snpe/CreateUserBuffer.cpp
API_SRCS-$(CONFIG_SNPE) += snpe/CreateUserBuffer.hpp
API_SRCS-$(CONFIG_SNPE) += snpe/LoadContainer.cpp
API_SRCS-$(CONFIG_SNPE) += snpe/LoadContainer.hpp
API_SRCS-$(CONFIG_SNPE) += snpe/LoadInputTensor.cpp
API_SRCS-$(CONFIG_SNPE) += snpe/LoadInputTensor.hpp
API_SRCS-$(CONFIG_SNPE) += snpe/NV21Load.cpp
API_SRCS-$(CONFIG_SNPE) += snpe/NV21Load.hpp
API_SRCS-$(CONFIG_SNPE) += snpe/PreprocessInput.cpp
API_SRCS-$(CONFIG_SNPE) += snpe/PreprocessInput.hpp
API_SRCS-$(CONFIG_SNPE) += snpe/SaveOutputTensor.cpp
API_SRCS-$(CONFIG_SNPE) += snpe/SaveOutputTensor.hpp
API_SRCS-$(CONFIG_SNPE) += snpe/SetBuilderOptions.cpp
API_SRCS-$(CONFIG_SNPE) += snpe/SetBuilderOptions.hpp
API_SRCS-$(CONFIG_SNPE) += snpe/Util.cpp
API_SRCS-$(CONFIG_SNPE) += snpe/Util.hpp
#API_SRCS-$(CONFIG_SNPE) += snpe/udlExample.cpp
#API_SRCS-$(CONFIG_SNPE) += snpe/udlExample.hpp
API_SRCS-$(CONFIG_SNPE) += snpe/main.cpp
API_SRCS-$(CONFIG_SNPE) += snpe/main.hpp

API_SRCS-yes += vpx_nemo.h
API_SRCS-yes += src/vpx_nemo.c
