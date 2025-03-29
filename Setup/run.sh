#!/bin/bash

sudo docker run --privileged -v /dev/bus/usb:/dev/bus/usb \
    -it \
    --name nemo_hyz_new_new \
    --network host \
    --dns=8.8.8.8 \
    -p 5003:5003 \
    --gpus all \
    --ipc=host \
    -v ${HOME}/docker/nemo:/workspace \
    -e http_proxy="http://127.0.0.1:7890" \
    -e https_proxy="http://127.0.0.1:7890" \
    nemo_hyz:latest \
    /bin/bash
