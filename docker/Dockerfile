FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime AS condabuilder
#python 3.11

ARG PYTORCH_VERSION=v2.7.1
ARG UID=1000
ARG ARG_USER_NAME=webui
ENV USER=$ARG_USER_NAME
ARG VENV_NAME="cosyvoice"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

# 更新并安装需要的程序
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    sed -i -e 's#archive.ubuntu.com#mirrors.ustc.edu.cn#g' -e 's#security.ubuntu.com#mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && apt-get update -y --fix-missing \
    && apt-get install -y git build-essential curl wget ffmpeg unzip git git-lfs sox libsox-dev tar bash vim \
    && apt-get clean && git lfs install && git config --global --add safe.directory '*' \
    && python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && useradd -l -u $UID -U -m -s /bin/bash ${USER}

ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH=/opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

WORKDIR /workspace
ENV PYTHONPATH="/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"

# RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
COPY ./CosyVoice /workspace/CosyVoice/
COPY ./docker/requirements.txt /workspace/CosyVoice/

RUN --mount=type=cache,target=/root/.cache/pip  \
   cd CosyVoice && python -m pip install  --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt  

COPY ./docker/ /workspace/CosyVoice/

RUN rm -rf /root/.cache/pip && conda clean --all && chown -R ${USER}:${USER} /workspace/ && chmod u+x /workspace/CosyVoice/entrypoint.sh \
  && sed -i -e 's#"https://huggingface.co#"https://hf-mirror.com#g' /opt/conda/lib/python3.1*/site-packages/huggingface_hub/constants.py 
   #&& chmod -R 777 /opt/conda/lib/python3.11/site-packages/tn/
   
USER ${USER}
ENV HOME=/home/${USER}
WORKDIR $HOME

ENV PATH=/opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH
RUN /opt/conda/bin/python3.11 -m venv  --system-site-packages ${VENV} \
 && echo "source $HOME/${VENV}/bin/activate" >> ${HOME}/.bashrc && source $HOME/${VENV}/bin/activate \
 && patch -Np1 /workspace/CosyVoice/cosyvoice/cli/cosyvoice.py < /workspace/CosyVoice/zero_shot_sft.patch \
 && patch -Np1 /workspace/CosyVoice/cosyvoice/cli/model.py < /workspace/CosyVoice/torch.cuda.amp.autocast.patch \
 && patch -Np1 /workspace/CosyVoice/cosyvoice/cli/frontend.py < /workspace/CosyVoice/frontend_zero_shot_del_key.patch 

ARG MODEL_PATH_ARG=pretrained_models/CosyVoice2-0.5B

ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MODEL_PATH=${MODEL_PATH_ARG}
ENV HF_ENDPOINT=https://hf-mirror.com
EXPOSE ${API_PORT} 8080
WORKDIR /workspace/CosyVoice
ENTRYPOINT ["./entrypoint.sh"]

