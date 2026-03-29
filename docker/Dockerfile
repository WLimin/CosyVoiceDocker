# 用pytorch的devel版本编译 flash-attn https://github.com/Dao-AILab/flash-attention
# FROM llm-flash-attn-builder:latest AS builder

# This image is based on the latest official PyTorch image, because it already contains CUDA, CuDNN, and PyTorch
# /opt/conda/bin/python3.11, uv, conda, pip3, venv
FROM aibase:torch291_cuda130
# ubuntu 22.04 lts, jammy, python 3.11

ARG UID=1000
ARG ARG_USER_NAME=ubuntu
ARG VENV_NAME="torch_cuda"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Asia/Shanghai \
    PATH=/opt/conda/bin:$PATH  \
    LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH \
    PYTORCH_VERSION=2.9.1

ENV USER_NAME=$ARG_USER_NAME
ENV VENV=$VENV_NAME

SHELL ["/bin/bash", "--login", "-c"]

# Installs Git, because ComfyUI and the ComfyUI Manager are installed by cloning their respective Git repositories
# Install system dependencies
RUN mkdir -p /models /app && chown -R ${USER_NAME}:${USER_NAME} /models /app # Create directory for models (will be mounted from host)

USER ${USER_NAME}
ENV HOME=/home/${USER_NAME}
# Let .pyc files be stored in one place
ENV PYTHONPYCACHEPREFIX="${HOME}/.pycache"

RUN mkdir -p ${HOME}/.local/bin ${HOME}/.pycache \
 && /opt/conda/bin/python3.11 -m venv  --system-site-packages ${HOME}/${VENV} \
 && echo "source $HOME/${VENV}/bin/activate" >> ${HOME}/.bashrc && source $HOME/${VENV}/bin/activate

# activate venv
ENV PATH=$HOME/${VENV}/bin:${HOME}/.local/bin:$PATH \
    PYTHONPATH="/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS"

WORKDIR /app

# Installs Git, because ComfyUI and the ComfyUI Manager are installed by cloning their respective Git repositories
# Install system dependencies
# ENV CONDA_DEFAULT_ENV=${VENV}
# ENV PATH=/opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH


# RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
COPY --chown=${USER_NAME}:${USER_NAME} ./CosyVoice ./docker/requirements.txt /app/CosyVoice/

RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple --upgrade pip && \
    cd CosyVoice && python -m pip install  --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple  --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt  

COPY --chown=${USER_NAME}:${USER_NAME} ./docker/ /app/CosyVoice/

RUN chmod u+x /app/CosyVoice/entrypoint.sh \
  && sed -i -e 's#"https://huggingface.co#"https://hf-mirror.com#g' ${HOME}/${VENV}/lib/python3.1*/site-packages/huggingface_hub/constants.py 
   #&& chmod -R 777 ${HOME}/${VENV}/lib/python3.11/site-packages/tn/
   
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

RUN patch -Np1 /app/CosyVoice/cosyvoice/cli/cosyvoice.py < /app/CosyVoice/zero_shot_sft.patch \
 && patch -Np1 /app/CosyVoice/cosyvoice/cli/model.py < /app/CosyVoice/torch.cuda.amp.autocast.patch \
 && patch -Np1 /app/CosyVoice/cosyvoice/cli/frontend.py < /app/CosyVoice/frontend_zero_shot_del_key.patch \
 && patch -Np1 /app/CosyVoice/cosyvoice/utils/file_utils.py < /app/CosyVoice/file_utils.patch

ARG MODEL_PATH_ARG=pretrained_models/CosyVoice2-0.5B

ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    HF_ENDPOINT=https://hf-mirror.com
ENV MODEL_PATH=${MODEL_PATH_ARG}

EXPOSE ${API_PORT} 8080
WORKDIR /app/CosyVoice
ENTRYPOINT ["./entrypoint.sh"]

