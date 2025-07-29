# CosyVoiceDocker

[CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 的docker文件。
## 功能：
*    默认使用 CosyVoice2。
*    提供 WebUi 界面和 OpenAI API (V1) 兼容两套独立服务，可通过 CAPABILITIES=[api|web|all]环境变量选择。
*    WebUi 界面中进行了加强。
+        在3秒急速复刻中增加了保存、删除扩展音色功能，后续可以和“中文女”等类似使用，并支持语音合成、3秒急速复刻和自然语言控制功能。
+        增加了外置音色.pt文件和音频wav,mp3等文件支持。
+        增加了FunASR自动识别上传或外置音频文件内容文本功能。
+        调整了部分界面布局。
*    OpenAI API接口服务：
+        支持Open Webui按照OpenAI API接口调用。
+        通过"随机数种子*角色或文件名:控制指令"的格式，利用voice角色字段进行了功能增强。
+        支持同时设置随机数种子、指定讲话角色或外置音频文件、直接给出自然语言控制指令等功能，自动选择推理方式。
*    同时支持 CPU 和 GPU。
*    非根用户webui运行。

## 依赖：
*    python 3.11
*    cuda 12.8.1
*    torch 2.7.1


## 模型和缓存目录
需要手工下载模型到指定目录下。参考目录和文件如下：
```bash
tree -A pretrained_models

pretrained_models
├── CosyVoice2-0.5B
│   ├── asset
│   │   └── dingding.png
│   ├── campplus.onnx
│   ├── configuration.json
│   ├── cosyvoice2.yaml
│   ├── CosyVoice-BlankEN
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── merges.txt
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── flow.cache.pt
│   ├── flow.decoder.estimator.fp32.onnx
│   ├── flow.encoder.fp16.zip
│   ├── flow.encoder.fp32.zip
│   ├── flow.pt
│   ├── hift.pt
│   ├── llm.pt
│   ├── README.md
│   ├── speech_tokenizer_v2.onnx
│   └── spk2info.pt
├── modelscope
│   └── hub
│       ├── iic
│       │   └── SenseVoiceSmall
│       │       ├── am.mvn
│       │       ├── chn_jpn_yue_eng_ko_spectok.bpe.model
│       │       ├── configuration.json
│       │       ├── config.yaml
│       │       ├── example
│       │       │   ├── en.mp3
│       │       │   ├── ja.mp3
│       │       │   ├── ko.mp3
│       │       │   ├── yue.mp3
│       │       │   └── zh.mp3
│       │       ├── fig
│       │       │   ├── aed_figure.png
│       │       │   ├── asr_results.png
│       │       │   ├── inference.png
│       │       │   ├── sensevoice.png
│       │       │   ├── ser_figure.png
│       │       │   └── ser_table.png
│       │       ├── model.pt
│       │       ├── README.md
│       │       └── tokens.json
│       └── pengzhendong
│           └── wetext
│               ├── configuration.json
│               ├── en
│               │   └── tn
│               │       ├── tagger.fst
│               │       └── verbalizer.fst
│               ├── README.md
│               └── zh
│                   ├── itn
│                   │   ├── tagger_enable_0_to_9.fst
│                   │   ├── tagger.fst
│                   │   └── verbalizer.fst
│                   └── tn
│                       ├── tagger.fst
│                       ├── verbalizer.fst
│                       └── verbalizer_remove_erhua.fst
└── voices
    ├── 火山 - 猴哥.pt
    ├── 火山 - 湾湾小何.pt
    ├── 雷军.pt
    └── jok老师.pt
```

## Build 
十分抱歉，我只有一台配置 Nvida GeForce RTX 5070 Ti 12G显卡的 Debian Linux 13(trixie)系统，该显卡似乎只支持 cuda 12.8。因此，选择pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime作为基础镜像。

进入顶层目录，通过以下指令构建 docker image。
```bash
docker build -t cosyvoice-gpu --progress=plain  --build-arg  ARG_USER_NAME=webui \
     --build-arg MODEL_PATH_ARG=pretrained_models/CosyVoice2-0.5B \
     -f docker/Dockerfile .
```
## 运行容器
```bash
#!/bin/bash
VOLUMES=$PWD/
mkdir -p ${VOLUMES}/pretrained_models/CosyVoice2-0.5B
which nvidia-smi
if [ $? -eq 0 ]; then #有gpu支持
RUN_USE_GPU="--name cosy-voice --gpus all"
else
RUN_USE_GPU="--name cosy-voice "
fi

#CAPABILITIES=api|web|all
CAPABILITIES=api
docker run -itd $RUN_USE_GPU \
	-p 8086:8080 -p 8087:8000 \
	-v ${VOLUMES}/pretrained_models:/workspace/CosyVoice/pretrained_models \
	-v ${VOLUMES}/asset:/workspace/CosyVoice/asset \
	-e CUDA_ENABLED=false \
	-e CAPABILITIES=${CAPABILITIES} \
	-e MODEL_PATH=pretrained_models/CosyVoice2-0.5B \
 cosyvoice-gpu

#解决让人讨厌的不联网出错退出，本地应该下载模型文件
# wetext: 确认文件方法： grep -Er 'wetext' ${VOLUMES}/CosyVoice
WETXT_CACHE=${VOLUMES}/pretrained_models/modelscope/hub/pengzhendong/wetext
USER_NAME=webui
if [ -d ${WETXT_CACHE} ]; then
    docker cp  cosy-voice:/opt/conda/lib/python3.11/site-packages/wetext/wetext.py /tmp/wetext.py
    sed -i -e "s#snapshot_download(\"pengzhendong/wetext\")#\"/home/${USER_NAME}/.cache/modelscope/hub/pengzhendong/wetext\"#g" /tmp/wetext.py 
    docker cp  /tmp/wetext.py cosy-voice:/opt/conda/lib/python3.11/site-packages/wetext/wetext.py 
fi
```
## API接口调用测试示例
**返回讲话角色**
    curl  "http://127.0.0.1:8087/v1/audio/voices" |jq .

**文本合成语音能力**
```bash
curl -X POST "http://127.0.0.1:8087/v1/audio/speech"  -H "Content-Type: application/json"  -d '{ "input": "Hello, 中文和英文混合测试。this is a test of the MeloTTS API.久しぶりです。最近何をしていますか？ CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。CosyVoice is undergoing a comprehensive upgrade, providing more accurate, stable, faster, and better voice generation capabilities.", "voice": "3456732*hua_zh.wav:四川话", "response_format": "wav","speed": 1.0 }' --output /tmp/output.wav;mpv /tmp/output.wav
```
（hua_zh.wav是外置的声音文件。嗯，加上四川话指令后不认识日语了，去掉就正常。）

## 致谢
    使用了大量代码和注释：https://github.com/jianchang512/cosyvoice-api
    音色资源文件：voices/*.pt https://github.com/journey-ad/CosyVoice2-Ex
    我自己^_^。理由：从网上、从AI对话中复制、粘贴并手搓了一堆我看不懂的 python 代码来显得自己很忙。
