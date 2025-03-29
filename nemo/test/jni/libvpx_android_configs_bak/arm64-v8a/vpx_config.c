/* Copyright (c) 2011 The WebM project authors. All Rights Reserved. */
/*  */
/* Use of this source code is governed by a BSD-style license */
/* that can be found in the LICENSE file in the root of the source */
/* tree. An additional intellectual property rights grant can be found */
/* in the file PATENTS.  All contributing project authors may */
/* be found in the AUTHORS file in the root of the source tree. */
#include "vpx/vpx_codec.h"
static const char* const cfg = "--force-target=aarch64-linux-android31 --sdk-path=/root/android-ndk-r26c --enable-neon --enable-internal-stats --disable-examples --disable-docs --enable-realtime-only --disable-vp8 --enable-libyuv --disable-runtime-cpu-detect --enable-snpe --extra-cflags=-isystem /root/android-ndk-r26c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include";
const char *vpx_codec_build_config(void) {return cfg;}
