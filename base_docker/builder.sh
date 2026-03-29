#!/bin/bash

USER_ID=$(id -u) 
GROUP_ID=$(id -g)
HOME_DIR=/app

build_flash_attn(){
# 构建 builder 阶段并打标签，flash_attn大约需要50分钟。
docker build --progress=plain \
 -t llm-flash-attn-builder:latest \
-f Dockerfile.flash_attn .

# 输出构建文件
docker run -it --rm \
    -e UID=$USER_ID -e GID=$GROUP_ID \
    --user $USER_ID:$GROUP_ID \
    -e "HOME=$HOME_DIR" \
    -v ./:/mnt \
    llm-flash-attn-builder:latest \
    /bin/bash -c "cp /app/flash_attn-* /mnt/flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
}
# vLLM
build_vllm(){
# 构建 builder 阶段并打标签，flash_attn大约需要50分钟。
docker build --progress=plain \
 -t vllm-builder:latest \
-f Dockerfile .

# 输出构建文件
docker run -it --rm \
    -e UID=$USER_ID -e GID=$GROUP_ID \
    --user $USER_ID:$GROUP_ID \
    -e "HOME=$HOME_DIR" \
    -v ./:/mnt \
    vllm-builder:latest \
    /bin/bash -c "cp /app/vllm-* /mnt"
}

# nunchaku
build_nunchaku(){
# 构建 builder 阶段并打标签，flash_attn大约需要50分钟。
docker build --progress=plain \
 -t nunchaku-builder:latest \
-f Dockerfile .

# 输出构建文件
docker run -it --rm \
    -e UID=$USER_ID -e GID=$GROUP_ID \
    --user $USER_ID:$GROUP_ID \
    -e "HOME=$HOME_DIR" \
    -v ./:/mnt \
    nunchaku-builder:latest \
    /bin/bash -c "cp /app/nunchaku-* /mnt"
}

 build_flash_attn
# build_vllm   'vllm==0.16.0' --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu130   vllm-0.16.0-cp38-abi3-manylinux_2_31_x86_64.whl
# build_nunchaku

