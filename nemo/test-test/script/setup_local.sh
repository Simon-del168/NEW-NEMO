#!/bin/bash

python ${NEMO_CODE_ROOT}/nemo/test-test/setup_local.py --libvpx_dir ${NEMO_CODE_ROOT}/third_party/nemo-libvpx --binary_dir ${NEMO_CODE_ROOT}/nemo/cache_profile/bin --jni_dir ${NEMO_CODE_ROOT}/nemo/test-test/jni --ndk_dir /android-ndk-r26c --snpe_dir ${NEMO_CODE_ROOT}/third_party/snpe
