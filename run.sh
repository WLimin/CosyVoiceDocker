#!/bin/bash
# 用什么nvidia/cuda:tag
# https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
# 12.4.1-base-ubuntu22.04 (12.4.1/ubuntu22.04/base/Dockerfile)
# 12.4.1-cudnn-devel-ubuntu22.04 (12.4.1/ubuntu22.04/devel/cudnn/Dockerfile)
# 12.4.1-cudnn-runtime-ubuntu22.04 (12.4.1/ubuntu22.04/runtime/cudnn/Dockerfile)
# 12.4.1-devel-ubuntu22.04 (12.4.1/ubuntu22.04/devel/Dockerfile)
# 12.4.1-runtime-ubuntu22.04 (12.4.1/ubuntu22.04/runtime/Dockerfile)
# auto,plain
#docker build -t cosyvoice-gpu --progress=plain -f docker/Dockerfile .
# docker run -d -it --name cosy-voice --network=openwebui-net -v ${PWD}/pretrained_models:/workspace/CosyVoice/pretrained_models  -v ${PWD}/asset:/workspace/CosyVoice/asset -p 8086:8080 -p 8087:8000 -e CUDA_ENABLED=false -e MODEL_PATH=pretrained_models/CosyVoice2-0.5B cosyvoice
# curl -X POST "http://127.0.0.1:8087/v1/audio/speech"  -H "Content-Type: application/json"  -d '{ "input": "Hello, 中文和英文混合测试。this is a test of the MeloTTS API.", "voice": "中文女", "response_format": "wav","speed": 1.0 }' --output /tmp/nfs/output.wav

VOLUMES=$PWD/
which nvidia-smi
if [ $? -eq 0 ]; then #有gpu支持
RUN_USE_GPU="--name cosy-voice --gpus all"
else
RUN_USE_GPU="--name cosy-voice "
fi

docker run -itd $RUN_USE_GPU \
--network=openwebui-net \
 -p 8086:8080 -p 8087:8000 \
-v ${VOLUMES}/pretrained_models:/workspace/CosyVoice/pretrained_models \
 -v ${VOLUMES}/asset:/workspace/CosyVoice/asset \
 -e CUDA_ENABLED=false \
 -e MODEL_PATH=pretrained_models/CosyVoice2-0.5B \
 cosyvoice-gpu

