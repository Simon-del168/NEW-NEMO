#!/bin/bash

function _usage() {
    cat << EOF
用法: $(basename "${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}") [-g GPU_INDEX] [-c CONTENT] [-q QUALITY] [-i INPUT_RESOLUTION] [-o OUTPUT_RESOLUTION]
EOF
}

function _set_conda() {
    # >>> conda initialize >>>
    __conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
            . "/opt/conda/etc/profile.d/conda.sh"
        else
            export PATH="/opt/conda/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
    conda deactivate
    conda activate nemo_py3.10  # 修改为你的虚拟环境
}

function _set_bitrate() {
    case "$1" in
        240) bitrate=512 ;;
        360) bitrate=1024 ;;
        480) bitrate=1600 ;;
        *) echo "[错误] 不支持的分辨率"; exit 1 ;;
    esac
}

function _set_num_blocks() {
    case "$1" in
        240) num_blocks=8 ;;
        360) num_blocks=4 ;;
        480) num_blocks=4 ;;
        *) echo "[错误] 不支持的分辨率"; exit 1 ;;
    esac
}

function _set_num_filters() {
    case "$1" in
        240)
            case "$2" in
                low) num_filters=9 ;;
                medium) num_filters=21 ;;
                high) num_filters=32 ;;
                *) echo "[错误] 不支持的质量"; exit 1 ;;
            esac
            ;;
        360)
            case "$2" in
                low) num_filters=8 ;;
                medium) num_filters=18 ;;
                high) num_filters=29 ;;
                *) echo "[错误] 不支持的质量"; exit 1 ;;
            esac
            ;;
        480)
            case "$2" in
                low) num_filters=4 ;;
                medium) num_filters=9 ;;
                high) num_filters=18 ;;
                *) echo "[错误] 不支持的质量"; exit 1 ;;
            esac
            ;;
        *) echo "[错误] 不支持的分辨率"; exit 1 ;;
    esac
}

# 检查参数数量
[[ $# -ge 1 ]] || { echo "[错误] 参数数量无效。使用 -h 查看帮助."; exit 1; }

# 解析命令行选项
while getopts ":g:c:q:i:o:h" opt; do
    case $opt in
        h) _usage; exit 0 ;;
        g) gpu_index="$OPTARG" ;;
        c) content="$OPTARG" ;;
        q) quality="$OPTARG" ;;
        i) input_resolution="$OPTARG" ;;
        o) output_resolution="$OPTARG" ;;
        \?) exit 1 ;;
    esac
done

# 验证所需变量
for var in gpu_index content quality input_resolution output_resolution; do
    if [ -z "${!var+x}" ]; then
        echo "[错误] $var 未设置"
        exit 1
    fi
done

_set_conda
_set_bitrate "$input_resolution"
_set_num_blocks "$input_resolution"
_set_num_filters "$input_resolution" "$quality"

# 运行训练脚本
CUDA_VISIBLE_DEVICES="$gpu_index" python "${NEMO_CODE_ROOT}/nemo/dnn/train_video.py" \
    --data_dir "${NEMO_DATA_ROOT}" \
    --content "${content}" \
    --lr_video_name "${input_resolution}p_${bitrate}kbps_s0_d300.webm" \
    --hr_video_name "2160p_12000kbps_s0_d300.webm" \
    --num_blocks "$num_blocks" \
    --num_filters "$num_filters" \
    --load_on_memory \
    --output_height "$output_resolution"
