#!/bin/bash

# curl -X POST "http://127.0.0.1:8087/v1/audio/speech"  -H "Content-Type: application/json"  -d '{ "input": "Hello, 中文和英文混合测试。this is a test of the MeloTTS API.久しぶりです。最近何をしていますか？ CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。CosyVoice is undergoing a comprehensive upgrade, providing more accurate, stable, faster, and better voice generation capabilities.", "voice": "步非烟女", "response_format": "wav","speed": 1.0 }' --output /tmp/output.wav;mpv /tmp/output.wav
# curl  "http://127.0.0.1:8087/v1/audio/voices" |jq .
#关于种子：
# 21986*中文女 2345678*步非烟女 48271500*神经女 1986*bjcx.wav:郑州话

VOLUMES=$PWD/

# 检查专属网络是否创建，用于OpenWebui+ollama的语音交互
DOCKER_NET=openwebui-net
docker network ls --format '{{.Name}}' | grep "${DOCKER_NET}"
if [ $? -ne 0 ]; then
    docker network create ${DOCKER_NET}
fi

# 宿主机是否有 nvidia GPU
which nvidia-smi
if [ $? -eq 0 ]; then #有gpu支持
    RUN_USE_GPU="--name cosy-voice --gpus all"
else
    RUN_USE_GPU="--name cosy-voice "
fi
# Debug: force use CPU
#RUN_USE_GPU="--name cosy-voice "

# 提供的服务。由于暂时不打算提供模型共享，可以选择api用于对话服务。webui界面主要完成语音复刻和测试。
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
 
# 容器运行补丁。目前已在Dockerfile中应用。
if false ; then

# 从外置文件生成内置音色错误
docker cp cosy-voice:/workspace/CosyVoice/cosyvoice/cli/cosyvoice.py /tmp/cosyvoice.py
patch -Np1 /tmp/cosyvoice.py < ${VOLUMES}/docker/zero_shot_sft.patch
docker cp /tmp/cosyvoice.py cosy-voice:/workspace/CosyVoice/cosyvoice/cli/cosyvoice.py

# 不提供prompt文件，直接使用 zero_shot_id 进行自然语言控制
docker cp cosy-voice:/workspace/CosyVoice/cosyvoice/cli/frontend.py /tmp/new/frontend.py
patch -Np1 /tmp/new/frontend.py < ${VOLUMES}/docker/frontend_zero_shot_del_key.patch
docker cp /tmp/new/frontend.py cosy-voice:/workspace/CosyVoice/cosyvoice/cli/frontend.py

# Remove: /workspace/CosyVoice/cosyvoice/cli/model.py:104: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
docker cp cosy-voice:/workspace/CosyVoice/cosyvoice/cli/model.py /tmp/model.py
patch -Np1 /tmp/model.py < ${VOLUMES}/docker/torch.cuda.amp.autocast.patch
docker cp /tmp/model.py cosy-voice:/workspace/CosyVoice/cosyvoice/cli/model.py

fi

#解决让人讨厌的 modelscope 不联网出错退出。本地应该完成下载模型文件
# wetext: 确认文件方法： grep -Er 'wetext' ${VOLUMES}/CosyVoice
WETXT_CACHE=${VOLUMES}/pretrained_models/modelscope/hub/pengzhendong/wetext
# 或者不采用用户名下的缓存路径，直接定向到 /workspace/CosyVoice//pretrained_models/modelscope/hub/pengzhendong/wetext
USER_NAME=webui
if [ -d ${WETXT_CACHE} ]; then
    docker cp  cosy-voice:/opt/conda/lib/python3.11/site-packages/wetext/wetext.py /tmp/wetext.py
    sed -i -e "s#snapshot_download(\"pengzhendong/wetext\")#\"/home/${USER_NAME}/.cache/modelscope/hub/pengzhendong/wetext\"#g" /tmp/wetext.py 
    docker cp  /tmp/wetext.py cosy-voice:/opt/conda/lib/python3.11/site-packages/wetext/wetext.py 
fi
