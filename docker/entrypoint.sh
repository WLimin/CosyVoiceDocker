#!/bin/bash

CUDA_ENABLED=${CUDA_ENABLED:-true}
DEVICE=""

if [ "${CUDA_ENABLED}" != "true" ]; then
    DEVICE="--device cpu"
fi
which python3
if [ ! -d ${HOME}/.cache/modelscope/hub ] ; then
	mkdir -p ${HOME}/.cache/
 	ln -s /workspace/CosyVoice/pretrained_models/modelscope/ ${HOME}/.cache/modelscope
fi
case ${CAPABILITIES} in
    api)
        exec python3 openai-api.py
    ;;
    web)
        exec python3 webui.py --port 8080 --model_dir ${MODEL_PATH} 
    ;;
    *)
        #all
        python3 webui.py --port 8080 --model_dir ${MODEL_PATH} &
        python3 openai-api.py
    ;;
esac

