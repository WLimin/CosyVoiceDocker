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
# curl  "http://127.0.0.1:8087/v1/audio/voices" |jq .
VOLUMES=$PWD/

# 检查专属网络是否创建
DOCKER_NET=openwebui-net
docker network ls --format '{{.Name}}' | grep "${DOCKER_NET}"
if [ $? -ne 0 ]; then
    docker network create ${DOCKER_NET}
fi
which nvidia-smi
if [ $? -eq 0 ]; then #有gpu支持
RUN_USE_GPU="--name cosy-voice --gpus all"
else
RUN_USE_GPU="--name cosy-voice "
fi
#debug force use CPU
#RUN_USE_GPU="--name cosy-voice "

#CAPABILITIES=api|web|all
CAPABILITIES=api
docker run -itd $RUN_USE_GPU \
    --network=${DOCKER_NET} \
	-p 8086:8080 -p 8087:8000 \
	-v ${VOLUMES}/pretrained_models:/workspace/CosyVoice/pretrained_models \
	-v ${VOLUMES}/asset:/workspace/CosyVoice/asset \
	-e CUDA_ENABLED=false \
	-e CAPABILITIES=${CAPABILITIES} \
	-e MODEL_PATH=pretrained_models/CosyVoice2-0.5B \
 cosyvoice-gpu
 
# 容器运行补丁
# 从外置文件生成内置音色错误
docker cp cosy-voice:/workspace/CosyVoice/cosyvoice/cli/cosyvoice.py /tmp/cosyvoice.py
patch -Np1 /tmp/cosyvoice.py < ${VOLUMES}/docker/zero_shot_sft.patch
docker cp /tmp/cosyvoice.py cosy-voice:/workspace/CosyVoice/cosyvoice/cli/cosyvoice.py

#解决让人讨厌的不联网出错退出，本地应该下载模型文件
# wetext: 确认文件方法： grep -Er 'wetext' ${VOLUMES}/CosyVoice
WETXT_CACHE=${VOLUMES}/pretrained_models/modelscope/hub/pengzhendong/wetext
USER_NAME=webui
if [ -d ${WETXT_CACHE} ]; then
    docker cp  cosy-voice:/opt/conda/lib/python3.11/site-packages/wetext/wetext.py /tmp/wetext.py
    sed -i -e "s#snapshot_download(\"pengzhendong/wetext\")#\"/home/${USER_NAME}/.cache/modelscope/hub/pengzhendong/wetext\"#g" /tmp/wetext.py 
    docker cp  /tmp/wetext.py cosy-voice:/opt/conda/lib/python3.11/site-packages/wetext/wetext.py 
fi
