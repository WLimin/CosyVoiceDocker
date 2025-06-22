FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS condabuilder

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
   && apt-get install -y git build-essential curl wget ffmpeg unzip git git-lfs sox libsox-dev tar bash vim && \
    apt-get clean && \
    git lfs install

FROM condabuilder AS cosyvoicebuilder
# ==================================================================
# conda install and conda forge channel as default
# ------------------------------------------------------------------
# Install miniforge
# RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
# https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-25.3.0-3-Linux-x86_64.sh
# COPY ./miniforge.sh /tmp/miniforge.sh
COPY ./Miniforge3-25.3.0-3-Linux-x86_64.sh /tmp/miniforge.sh

RUN /bin/bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /opt/nvidia/entrypoint.d/100.conda.sh && \
    echo "conda activate ${VENV}" >> /opt/nvidia/entrypoint.d/110.conda_default_env.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> $HOME/.bashrc

ENV PATH=/opt/conda/condabin:$PATH

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict
# ------------------------------------------------------------------
# ~conda
# ==================================================================

RUN conda create -y -n ${VENV} python=3.10
ENV CONDA_DEFAULT_ENV=${VENV}
ENV PATH=/opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

WORKDIR /workspace
ENV PYTHONPATH="/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"

# RUN conda init 
RUN conda activate ${VENV} && conda install -y -c conda-forge pynini
#pynini==2.1.5

# RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
COPY ./CosyVoice /workspace/CosyVoice/
COPY ./docker/requirements.txt /workspace/CosyVoice/
COPY ./add/ttsfrd*.whl /tmp/
# COPY ./pretrained_models/CosyVoice2-0.5B /workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B/

RUN conda activate ${VENV} && cd CosyVoice && \
    pip install -i https://mirrors.ustc.edu.cn/pypi/simple -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt  \
    && pip install -i https://mirrors.ustc.edu.cn/pypi/simple /tmp/ttsfrd_dependency-0.1-py3-none-any.whl /tmp/ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl \
    && rm /tmp/ttsfrd*.whl && rm -rf /root/.cache/pip && conda clean --all

FROM condabuilder AS fine
COPY --from=cosyvoicebuilder /opt/conda/ /opt/conda/
COPY --from=cosyvoicebuilder /opt/nvidia/ /opt/nvidia/
COPY --from=cosyvoicebuilder /workspace/CosyVoice/ /workspace/CosyVoice/

COPY ./docker/api.py /workspace/CosyVoice/
COPY ./docker/entrypoint.sh /workspace/CosyVoice/
COPY ./docker/webui.py /workspace/CosyVoice/

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${VENV}" >> $HOME/.bashrc

ENV PATH=/opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

ARG MODEL_PATH_ARG=pretrained_models/CosyVoice2-0.5B

WORKDIR /workspace/CosyVoice
EXPOSE 8000 8080
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MODEL_PATH=${MODEL_PATH_ARG}
ENV HF_ENDPOINT=https://hf-mirror.com
ENTRYPOINT ["./entrypoint.sh"]
#ENTRYPOINT ["/opt/conda/envs/cosyvoice/bin/python3", "webui.py", "--port 8000", "--model_dir ${MODEL_PATH}" ]
#CMD ["/opt/conda/envs/cosyvoice/bin/python3", "api.py"]
