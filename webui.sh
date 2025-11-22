#!/bin/bash
# 启动cosy-voice的webui服务
VOLUMES=$PWD/
CONTAINER_NAME=cosy-voice
SHELL_FOLDER="$( dirname "${BASH_SOURCE[0]}" )"
NS=$(docker ps -a --format '{{json .Names}},{{json .State}}' | grep "${CONTAINER_NAME}")
if [ $? -eq 0 ]; then
    # 已存在
    echo "Start Cosy-Voice TTS server..."
    docker start ${CONTAINER_NAME}
else
    echo "Call ${SHELL_FOLDER}/run.sh to create Cosy-Voice TTS server..."
    source ${SHELL_FOLDER}/run.sh
fi
echo "Start Cosy-Voice TTS WebUi server, press the Ctrl+C to exit.."
MODEL_PATH="pretrained_models/CosyVoice2-0.5B"
docker exec -it ${CONTAINER_NAME} /bin/bash -c "python3 webui.py --port 8080 --model_dir ${MODEL_PATH}"
 
