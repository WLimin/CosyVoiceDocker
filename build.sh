#!/bin/bash
# --progress=auto,plain
docker build -t cosyvoice-gpu --progress=plain  --build-arg  ARG_USER_NAME=webui \
     --build-arg MODEL_PATH_ARG=pretrained_models/CosyVoice2-0.5B \
     -f docker/Dockerfile .

