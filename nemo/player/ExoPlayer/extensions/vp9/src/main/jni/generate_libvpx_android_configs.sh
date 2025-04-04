#!/bin/bash
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

# a bash script that generates the necessary config files for libvpx android ndk
# builds.

set -e
#if [ $# -ne 1 ]; then
#  echo "Usage: ${0} <path_to_android_ndk>"
  #echo "Usage: ${0} </home/hyz/Android/Sdk/ndk/26.2.11394342>"
#    exit
#fi

#ndk="${1}"
#shift 1

ndk="/home/hyz/Android/Sdk/ndk/26.2.11394342"
# 检查 NDK 路径是否有效
if [ ! -d "$ndk" ]; then
  echo "Error: NDK path does not exist: $ndk"
  exit 1
fi

if [ -d libvpx_android_configs ]; then rm -rf libvpx_android_configs; fi

# configuration parameters common to all architectures
common_params="--disable-examples --disable-docs --enable-realtime-only"
common_params+=" --disable-vp8"
common_params+=" --enable-libyuv --disable-runtime-cpu-detect"
common_params+=" --disable-internal-stats"
#common_params+=" --disable-vp8 --disable-vp9-encoder --disable-webm-io"
#common_params+=" --disable-libyuv --disable-runtime-cpu-detect"
#common_params+=" --enable-external-build"

arch[0]="arm64-v8a"
config[0]="--force-target=aarch64-linux-android31 --sdk-path=$ndk --enable-neon --enable-internal-stats --enable-snpe"

# configuration parameters for various architectures
#arch[0]="armeabi-v7a"
#config[0]="--target=armv7-android-gcc --sdk-path=$ndk --enable-neon"
#config[0]+=" --enable-neon-asm"

#arch[1]="armeabi"
#config[1]="--target=armv7-android-gcc --sdk-path=$ndk --disable-neon"
#config[1]+=" --disable-neon-asm"

#arch[2]="mips"
#config[2]="--force-target=mips32-android-gcc --sdk-path=$ndk"

#arch[3]="x86"
#config[3]="--force-target=x86-android-gcc --sdk-path=$ndk --disable-sse2"
#config[3]+=" --disable-sse3 --disable-ssse3 --disable-sse4_1 --disable-avx"
#config[3]+=" --disable-avx2 --enable-pic"

#arch[4]="arm64-v8a"
#config[4]="--force-target=armv8-android-gcc --sdk-path=$ndk --enable-neon"

#arch[5]="x86_64"
#config[5]="--force-target=x86_64-android-gcc --sdk-path=$ndk --disable-sse2"
#config[5]+=" --disable-sse3 --disable-ssse3 --disable-sse4_1 --disable-avx"
#config[5]+=" --disable-avx2 --enable-pic --disable-neon --disable-neon-asm"

#arch[6]="mips64"
#config[6]="--force-target=mips64-android-gcc --sdk-path=$ndk"

limit=$((${#arch[@]} - 1))

# list of files allowed after running configure in each arch directory.
# everything else will be removed.
allowed_files="libvpx_srcs.txt vpx_config.c vpx_config.h vpx_scale_rtcd.h"
allowed_files+=" vp8_rtcd.h vp9_rtcd.h vpx_version.h vpx_config.asm"
#allowed_files+=" vp9_rtcd.h vpx_version.h vpx_config.asm"
allowed_files+=" vpx_dsp_rtcd.h libvpx.ver"

remove_trailing_whitespace() {
  perl -pi -e 's/\s\+$//' "$@"
}

convert_asm() {
  for i in $(seq 0 ${limit}); do
    while read file; do
      case "${file}" in
        *.asm.[sS])
          # Some files may already have been processed (there are duplicated
          # .asm.s files for vp8 in the armeabi/armeabi-v7a configurations).
          file="libvpx/${file}"
          if [[ ! -e "${file}" ]]; then
            asm_file="${file%.[sS]}"
            cat "${asm_file}" | libvpx/build/make/ads2gas.pl > "${file}"
            remove_trailing_whitespace "${file}"
            rm "${asm_file}"
          fi
          ;;
      esac
    done < libvpx_android_configs/${arch[${i}]}/libvpx_srcs.txt
  done
}

extglob_status="$(shopt extglob | cut -f2)"
shopt -s extglob
for i in $(seq 0 ${limit}); do
  mkdir -p "libvpx_android_configs/${arch[${i}]}"
  pushd "libvpx_android_configs/${arch[${i}]}"

  # configure and make
  #echo "build_android_configs: "
  #echo "configure ${config[${i}]} ${common_params}"
  #../../libvpx/configure ${config[${i}]} ${common_params} --extra-cflags=" \
  #  -isystem $ndk/sysroot/usr/include/arm-linux-androideabi \
  #  -isystem $ndk/sysroot/usr/include \
  #"

  # configure and make
  echo "build_android_configs: "
  echo "configure ${config[${i}]} ${common_params}"
  /root/docker/nemo/nemo/nemo/player/ExoPlayer/extensions/vp9/src/main/jni/libvpx/configure ${config[${i}]} ${common_params} --extra-cflags="-isystem $ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/c++/v1 -isystem $ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include"

  #rm -f libvpx_srcs.txt
  for f in ${allowed_files}; do
    # the build system supports multiple different configurations. avoid
    # failing out when, for example, vp8_rtcd.h is not part of a configuration
    make "${f}" || true
  done

  # remove files that aren't needed
  #rm -rf !(${allowed_files// /|})
  #remove_trailing_whitespace *

  popd
done

# restore extglob status as it was before
if [[ "${extglob_status}" == "off" ]]; then
  shopt -u extglob
fi

convert_asm

echo "Generated android config files."