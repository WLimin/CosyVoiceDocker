#!/bin/bash

CUDA_ENABLED=${CUDA_ENABLED:-true}
DEVICE=""

if [ "${CUDA_ENABLED}" != "true" ]; then
    DEVICE="--device cpu"
fi

case ${CAPABILITIES} in
    api)
        exec /opt/conda/envs/${VENV}/bin/python3 api.py
    ;;
    web)
        exec /opt/conda/envs/${VENV}/bin/python3 webui.py --port 8080 --model_dir ${MODEL_PATH} 
    ;;
    *)
        #all
        /opt/conda/envs/${VENV}/bin/python3 webui.py --port 8080 --model_dir ${MODEL_PATH} &
        exec /opt/conda/envs/${VENV}/bin/python3 api.py
    ;;
esac

