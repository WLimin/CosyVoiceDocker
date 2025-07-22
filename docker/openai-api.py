from flask import Flask, request, render_template, jsonify,  send_from_directory, send_file, Response, stream_with_context, make_response
import torch
import torchaudio
from flask_cors import CORS
import base64
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
import random
import datetime
import shutil
import subprocess
#from logging.handlers import RotatingFileHandler
#import logging
import os, io, sys
import time
import json
from pathlib import Path
root_dir = Path(__file__).parent.as_posix()

# ffmpeg 路径，实际不需要
if sys.platform == 'win32':
    os.environ['PATH'] = root_dir + f';{root_dir}\\ffmpeg;' + os.environ['PATH']+f';{root_dir}/third_party/Matcha-TTS'
else:
    os.environ['PATH'] = root_dir + f':{root_dir}/ffmpeg:' + os.environ['PATH']
    os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':third_party/Matcha-TTS'

sys.path.append(f'{root_dir}/third_party/Matcha-TTS')

model_dir = Path(f'{root_dir}/pretrained_models/CosyVoice2-0.5B').as_posix()
voices_dir = Path(f'{root_dir}/pretrained_models/voices').as_posix()
asset_dir = Path(f'{root_dir}/asset').as_posix()

tmp_dir = Path(f'/tmp/api').as_posix()
logs_dir = Path(f'/tmp/logs').as_posix()
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# 下载模型(Dockerfile 已经完成)
# from modelscope import snapshot_download
# snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
# snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')

#
# app logs
#
# 配置日志
# 禁用 Werkzeug 默认的日志处理器
log = logging.getLogger('werkzeug')
log.handlers[:] = []
log.setLevel(logging.WARNING)

root_log = logging.getLogger()  # Flask的根日志记录器
root_log.handlers = []
root_log.setLevel(logging.WARNING)

app = Flask(__name__, static_folder=tmp_dir, static_url_path='/tmp')
app.logger.setLevel(logging.WARNING)
# 创建 RotatingFileHandler 对象，设置写入的文件路径和大小限制
#file_handler = RotatingFileHandler(logs_dir+f'/{datetime.datetime.now().strftime("%Y%m%d")}.log', maxBytes=1024 * 1024, backupCount=5)
# 创建日志的格式
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 设置文件处理器的级别和格式
#file_handler.setLevel(logging.WARNING)
#file_handler.setFormatter(formatter)
# 将文件处理器添加到日志记录器中
#app.logger.addHandler(file_handler)

CORS(app, cors_allowed_origins="*")
# CORS(app, supports_credentials=True)

prompt_sr = 16000 #默认提示采样率
tts_model = None
default_seed = 21986 # 随便写的
seed = default_seed  # random.randint(1, 100000000)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
default_voices = [] # 默认模型音色
spk_custom = [] # 自定义音色库，存放在voices_dir
asset_wav_list = []  # wav等文件目录列表, ${root_dir/asset/}参考音频选择列表

# 建立模型实例，考虑空间问题，不再支持V1版本
def check_tts_model():
    global tts_model, seed
    if tts_model is None:
        # flag = False
        # try:
        #    tts_model = CosyVoice(model_dir)
        # except Exception:
        try:
            #        flag = True
            tts_model = CosyVoice2(model_dir, load_jit=False, fp16=False if device_str == 'cpu' else True) #, load_trt=False)
        except Exception:
            raise TypeError('no valid model_type!')
    logging.info(f"set all random seed to {seed}.")
    set_all_random_seed(seed)

# ========== 工具函数 =============
def base64_to_wav(encoded_str, output_path):
    if not encoded_str:
        raise ValueError("Base64 encoded string is empty.")

    # 将base64编码的字符串解码为字节
    wav_bytes = base64.b64decode(encoded_str)

    # 检查输出路径是否存在，如果不存在则创建
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 将解码后的字节写入文件
    with open(output_path, "wb") as wav_file:
        wav_file.write(wav_bytes)

    logging.info(f"WAV file has been saved to {output_path}")

# 获取请求参数
def get_params(req):
    params = {
        "text": "", "lang": "", "role": "中文女",
        "reference_audio": None, "reference_text": "",
        "speed": 1.0, "streaming": 0
    }
    # 原始字符串
    params['text'] = req.args.get("text", "").strip() or req.form.get("text", "").strip()

    # 字符串语言代码
    params['lang'] = req.args.get("lang", "").strip().lower() or req.form.get("lang", "").strip().lower()
    # 兼容 ja语言代码
    if params['lang'] == 'ja':
        params['lang'] = 'jp'
    elif params['lang'][:2] == 'zh':
        # 兼容 zh-cn zh-tw zh-hk
        params['lang'] = 'zh'

    # 角色名
    role = req.args.get("role", "").strip() or req.form.get("role", '')
    if role:
        params['role'] = role

    # 流式输出
    streaming = req.args.get("streaming", "").strip() or req.form.get("streaming", '')
    streaming = 1 if streaming == '1' else 0
    params['streaming'] = streaming

    # 要克隆的音色文件
    params['reference_audio'] = req.args.get("reference_audio", None) or req.form.get("reference_audio", None)
    encode = req.args.get('encode', '') or req.form.get('encode', '')
    if encode == 'base64':
        tmp_name = f'tmp/{time.time()}-clone-{len(params["reference_audio"])}.wav'
        base64_to_wav(params['reference_audio'], root_dir+'/'+tmp_name)
        params['reference_audio'] = tmp_name
    # 音色文件对应文本
    params['reference_text'] = req.args.get("reference_text", '').strip() or req.form.get("reference_text", '')

    return params

def del_tmp_files(tmp_files: list):
    """ 删除临时碎片语音文件 """
    logging.info('正在删除缓存文件...')
    for f in tmp_files:
        if os.path.exists(f):
            logging.info('删除缓存文件:', f)
            os.remove(f)

def process_audio(tts_speeches, sample_rate=prompt_sr, format="wav"):
    """处理音频数据并返回响应"""
    buffer = io.BytesIO()
    # 原始采样率（CosyVoice 默认为22050）
    original_sr = 22050

    audio_data = torch.concat(tts_speeches if tts_speeches  else [torch.zeros(1, int(original_sr * 0.2))], dim=1)
    # 如果目标采样率与原始采样率不同，进行重采样
    if sample_rate != original_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=sample_rate)
        audio_data = resampler(audio_data)

    torchaudio.save(buffer, audio_data, sample_rate, format=format)
    buffer.seek(0)
    return buffer

def load_voice_to_tmp(voice_data):
    """加载音色文件中内置的音频数据和文本（或许），16000,1ch,wav
    输入：内置扩展音色Tensor
    返回：临时wav文件名，参考音频识别文本
    备注：绕过内置扩展音色进行语音控制推理时只采用zero_shot_spk_id的错误。
    """
    text_ref = ''
    #生成临时音频文件并返回
    ref_audio = f"/tmp/t-refaudio.wav"
    buffer = io.BytesIO()
    try:
        audio_ref= voice_data.get('audio_ref')
        torchaudio.save(buffer, audio_ref, prompt_sr, format="wav")  # ERROR: Input tensor has to be on CPU.
        buffer.seek(0)
        # 打开文件用于写入二进制数据
        with open(ref_audio,'wb') as file:
            file.write(buffer.getvalue())

        text_ref = voice_data.get('text_ref') if voice_data else None
    except Exception as e:
        ref_audio = ''
        logging.error(f"保存音色文件失败: {e}")

    return ref_audio, text_ref

def batch(tts_type, outname, params):
    """ 实际批量合成完毕后连接为一个文件
    条件符合/冲突时，优先级安排
        内置音色：只支持tts，忽略其它输入
        扩展音色：支持tts、instruct（有instruct text）
        外置音色：3s复刻、指令。会写入临时音频文件并加载到prompt wav.
        外置声音文件：3s复刻、指令。

        若指定了外置声音文件、上传声音文件或录音文件，优先采用指令中指定的文件，即外置或临时声音文件
    """
    global tts_model
    logging.info(f"tts_type={tts_type}, outname={outname}\nparams={params}")
    # 根据传入参数的模式，检查是否需要加载参考音频文件
    prompt_speech_16k = None
    zero_shot_spk_id = ''
    if tts_type != 'tts':
        if params['role'] in default_voices and params['reference_audio'] is None:        # 内置扩展音色不需要加载参考音频
            zero_shot_spk_id = params['role']
            logging.info(f"内置扩展音色推理模式: {zero_shot_spk_id}，转外置音色模式处理")
            #BUG Around: 考虑加载内置扩展音色，转外置音色模式处理
            prompt_speech_16k = None
            #BUG AROUND: 考虑导出为临时文件绕过
            [prompt_wav, prompt_speech_text] = load_voice_to_tmp(tts_model.frontend.spk2info[zero_shot_spk_id])

            ref_audio = f"{tmp_dir}/t-refaudio-{time.time()}.wav"
            try:
                subprocess.run(["ffmpeg", "-hide_banner", "-ignore_unknown", "-y", "-i", f"{prompt_wav}", "-ar", f"{prompt_sr}", ref_audio],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", check=True, text=True,
                            creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW)
            except Exception as e:
                raise Exception(f'处理参考音频失败:{e}')
            prompt_speech_16k = load_wav(ref_audio, prompt_sr)
            if prompt_speech_16k is not None:
                zero_shot_spk_id = ''

        else:
            if not params['reference_audio'] or not os.path.exists(f"{params['reference_audio']}"):
                raise Exception(f'参考音频未传入或不存在 {params["reference_audio"]}')

            ref_audio = f"{tmp_dir}/t-refaudio-{time.time()}.wav"
            try:
                subprocess.run(["ffmpeg", "-hide_banner", "-ignore_unknown", "-y", "-i", f"{params['reference_audio']}", "-ar", f"{prompt_sr}", ref_audio],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", check=True, text=True,
                            creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW)
            except Exception as e:
                raise Exception(f'处理参考音频失败:{e}')
            prompt_speech_16k = load_wav(ref_audio, prompt_sr)

    streaming = bool(int(params.get('streaming', 0)))
    format = params.get('format', 'wav')
    text = params['text']
    speeding = params['speed']
    audio_list = []
    check_tts_model()   # 主要目的是设置 seed

    if tts_type == 'tts':
        inference_func = lambda: tts_model.inference_sft(text, params['role'], stream=streaming, speed=speeding)
    elif tts_type == 'clone_eq' and params.get('reference_text'):
        inference_func = lambda: tts_model.inference_zero_shot(text, params.get('reference_text'), prompt_speech_16k, stream=streaming, speed=speeding)
    elif tts_type == 'instruct' and params.get('instruct_text'):
        inference_func = lambda: tts_model.inference_instruct2(text, params.get('instruct_text'), prompt_speech_16k, stream=streaming, zero_shot_spk_id = zero_shot_spk_id)
    else:  # default clone or clone_mul
        inference_func = lambda: tts_model.inference_cross_lingual(text, prompt_speech_16k, stream=streaming, speed=speeding)

    # 处理流式输出
    if streaming:
        def generate():
            for _, i in enumerate(inference_func()):
                buffer = process_audio([i['tts_speech']], format="ogg")
                yield buffer.read()

        response = make_response(generate())
        response.headers.update({
            'Content-Type': 'audio/ogg',
            'Content-Disposition': 'attachment; filename=sound.ogg'
        })
        return response

    # 处理非流式输出
    for i, j in enumerate(inference_func()):
        audio_list.append(j['tts_speech'])

    audio_data = torch.concat(audio_list if audio_list  else [torch.zeros(1, int(tts_model.sample_rate * 0.2))], dim=1)

    # 根据模型设置采样率
    if tts_type == 'tts':
        torchaudio.save(tmp_dir + '/' + outname, audio_data, 22050, format=format)
    else:
        torchaudio.save(tmp_dir + '/' + outname, audio_data, 24000, format=format)

    logging.info(f"音频文件生成成功：{tmp_dir}/{outname}")
    return send_file(tmp_dir + '/' + outname, mimetype=f'audio/{format}')

# ============= 4 种合成方式选择 ===========
@app.route('/tts', methods=['GET', 'POST'])
def tts():
    """根据内置角色合成文字"""
    params = get_params(request)
    if not params['text']:
        # 设置状态码为500
        return make_response(jsonify({"code": 1, "msg": '缺少待合成的文本'}), 500)

    try:
        # 仅文字合成语音
        outname = f"tts-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-')}.wav"
        return batch(tts_type='tts', outname=outname, params=params)

    except Exception as e:
        logging.error(e)
        # 设置状态码为500
        return make_response(jsonify({"code": 2, "msg": str(e)}), 500)
#    else:
#        return send_file(outname, mimetype='audio/x-wav')

@app.route('/clone_mul', methods=['GET', 'POST'])
@app.route('/clone', methods=['GET', 'POST'])
def clone():
    """ 跨语言文字合成语音 """
    try:
        params = get_params(request)
        if not params['text']:
            # 设置状态码为500
            return make_response(jsonify({"code": 6, "msg": '缺少待合成的文本'}), 500)

        outname = f"clone-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-')}.wav"
        return batch(tts_type='clone', outname=outname, params=params)
    except Exception as e:
        # 设置状态码为500
        return make_response(jsonify({"code": 8, "msg": str(e)}), 500)
#    else:
#        return send_file(outname, mimetype='audio/x-wav')

@app.route('/clone_eq', methods=['GET', 'POST'])
def clone_eq():
    """ 同语言克隆音色合成 """
    try:
        params = get_params(request)
        if not params['text']:
            # 设置状态码为500
            return make_response(jsonify({"code": 6, "msg": '缺少待合成的文本'}), 500)
        if not params['reference_text']:
            # 设置状态码为500
            return make_response(jsonify({"code": 6, "msg": '同语言克隆必须传递引用文本'}), 500)

        outname = f"clone-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-')}.wav"
        return batch(tts_type='clone_eq', outname=outname, params=params)
    except Exception as e:
        # 设置状态码为500
        return make_response(jsonify({"code": 8, "msg": str(e)}), 500)
#    else:
#        return send_file(outname, mimetype='audio/x-wav')

# ========= OpenAI api 兼容 ==============
@app.route("/speakers", methods=['GET', 'POST'])
@app.route('/v1/audio/voices', methods=['GET', 'POST'])
def get_voices():
    """获取角色名称"""
    speakers = default_voices + spk_custom + asset_wav_list
    return {"available_speakers": list(speakers)}

def decode_input_instruct(inputstr):
    """ 处理指令列表字符串
        指令格式：随机数种子*角色名:其它指令
            角色名可为内置音色名、内置扩展音色名、外置音色名(不含扩展名.pt)或外置音频文件名(含扩展名)
        指令举例：
            中文女
            步非烟女
            21986*中文女
            2345678*步非烟女

            1986*bjcx.wav
            1986*bjcx.wav:郑州话
            66668*电台播音女:四川话
    """
    # 提取随机数种子，第一个 * 分割随机数种子和后续指令
    if '*' in inputstr:
        [seedstr, voicestr] = inputstr.split('*', 1)
    else:
        seedstr = ''
        voicestr = inputstr

    # 分离角色/音色和指令，第一个:分割角色/音色和指令
    if ':' in voicestr: # ：符号分割音色和推理指令
        [voicestr, instruct] = voicestr.split(':', 1)
    else:
        instruct = ''
    return seedstr, voicestr, instruct

@app.route('/v1/audio/speech', methods=['POST'])
def audio_speech():
    """
    兼容 OpenAI /v1/audio/speech API 的接口
    """
    global seed
    if not request.is_json:
        logging.info(f'请求必须是 JSON 格式')
        return jsonify({"error": "请求必须是 JSON 格式，参考OpenAI规范"}), 400
    # 初始化空白参数
    params = get_params(request)
    data = request.get_json()

    # 检查请求中是否包含必要的参数
    if 'input' not in data or 'voice' not in data:
        logging.info(f'请求缺少必要的参数： input, voice')
        return jsonify({"error": "请求缺少必要的参数：input:文本, voice:角色名"}), 400

    text = data.get('input')
    speed = float(data.get('speed', 1.0))

    voice = data.get('voice', '中文女') # 此处保证 voice!=''

    params['text'] = text
    params['speed'] = speed
    api_name = 'tts'

    [seedstr, voicestr, instruct] = decode_input_instruct(voice)
    try:
        if bool(seedstr):
            seed = int(seedstr) & 0xffffffff        # <4294967295,0xffffffff, 实际上，训练时候用的数值不超过100000。
    except Exception as e:
         logging.error(f"设置随机数种子失败。检查指令是否正确。解析：'{seedstr}'，错误：{e}")

    logging.info(f'Api:input={voice} => [Seed={seedstr}({seed}), Role={voicestr},  Instruct={instruct}]')
    if voicestr == '':
        logging.info(f'必要参数请求voice内容为空错误：{voice}')
        return jsonify({"error": "必要参数请求voice内容错误：{voice}"}), 400

    # 此时正交条件有：
    #           内置音色 扩展音色 外置音色      外置声音文件
    # 无指令    tts       tts     clone_eq       clone_eq
    # refaudio  None      None    load+text      load+text
    # 有指令    tts/err  instruct instruct       instruct
    # refaudio  None     None     load+text     load+text

    if voicestr in default_voices: # 内置音色/扩展音色
        params['role'] = voicestr
        api_name = 'tts'
        if instruct != '':
            api_name = 'instruct'
            params['instruct_text'] = instruct

        if 'flow_prompt_speech_token' in tts_model.frontend.spk2info[voicestr].keys(): #不是内置sft
            logging.info(f"内置扩展音色: {voicestr}")
        else: # 内置sft，‘中文女’等不支持指令模式
            if api_name != 'tts':
                logging.info(f"内置音色: {voicestr} 不支持指令模式。")
                api_name = 'tts'
                params['instruct_text'] = ''
            else:
                logging.info(f"内置音色: {voicestr}")

    elif voicestr in spk_custom:    # 处理外置音色
        full_path = f"{voices_dir}/{voicestr}.pt"
        if Path(full_path).exists():
            #生成临时音频文件并返回
            ref_audio = f"{tmp_dir}/t-refaudio.wav"
            try:
                voice_data = torch.load(full_path, map_location=torch.device(device_str))
                buffer = io.BytesIO()
                audio_ref= voice_data.get('audio_ref').to('cpu')
                torchaudio.save(buffer, audio_ref, prompt_sr, format="wav")  # ERROR: Input tensor has to be on CPU.
                buffer.seek(0)
                # 打开文件用于写入二进制数据
                with open(ref_audio,'wb') as file:
                    file.write(buffer.getvalue())

                # 打开参考文本文件并读取所有内容
                reference_text = voice_data.get('text_ref')
                params['reference_text'] = reference_text
                params['reference_audio'] = ref_audio
                logging.info(f"成功加载外置音色文件'{full_path}'")
            except Exception as e:
                logging.error(f"加载外置音色文件'{full_path}'失败")
                return jsonify({"error": {"message": f"加载外置音色文件'{full_path}'失败", "Exception": f'{e}', "type": e.__class__.__name__, "param": f'speed={speed},voice={voice},input={text}', "code": 400}}), 500

            api_name = 'clone' if instruct == '' else 'instruct'
            params['role'] = voicestr   # 区分是否为.pt
            params['instruct_text'] = instruct
        else:
            logging.error(f"参考外置音色文件'{full_path}'不存在")
            return jsonify({"error": {"message": f"参考外置音色文件'{full_path}'不存在", "param": f'speed={speed},voice={voice},input={text}', "code": 400}}), 500
    elif voicestr in asset_wav_list:
        ref_audio = f'{asset_dir}/{voicestr}'
        if Path(ref_audio).exists():
            params['reference_audio'] = ref_audio
            params['instruct_text'] = instruct
            api_name = 'clone' if instruct == '' else 'instruct'

            # 打开参考文本文件并读取所有内容
            if os.path.exists(f"{ref_audio}.txt"):
                with open(f"{ref_audio}.txt", 'r', encoding='utf-8') as file:
                    reference_text = file.read()
                    params['reference_text'] = reference_text
            else:
                logging.error(f"参考音频提示文件'{ref_audio}.txt'不存在")
                return jsonify({"error": {"message": f"参考音频提示文件'{ref_audio}.txt'不存在", "param": f'speed={speed},voice={voice},input={text}', "code": 400}}), 500
        else:
            logging.error(f"参考音频文件'{ref_audio}'不存在")
            return jsonify({"error": {"message": f"参考音频文件'{ref_audio}'不存在", "param": f'speed={speed},voice={voice},input={text}', "code": 400}}), 500
    else:
        logging.error(f"必须填写配音角色名或参考音频路径")
        return jsonify({"error": {"message": f"必须填写配音角色名或参考音频路径", "param": f'speed={speed},voice={voice},input={text}', "code": 400}}), 500

    filename = f'openai-l{len(text)}-s{speed}-{time.time()}-r{seed}-{random.randint(1000,99999)}.wav'
    try:
        return batch(tts_type=api_name, outname=filename, params=params)
#        return send_file(outname, mimetype='audio/x-wav')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": {"message": f"{e}", "type": e.__class__.__name__, "param": f'speed={speed},voice={voice},input={text}', "code": 400}}), 500


if __name__ == '__main__':
    host = os.getenv('API_HOST', '0.0.0.0')
    port = os.getenv('API_PORT', '8000')
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    logging.getLogger().setLevel(getattr(logging, log_level))

    if not shutil.which("ffmpeg"):
        logging.error('必须安装 ffmpeg')

    logging.info(f"初始化加载模型文件……")
    check_tts_model()
    # 默认模型音色
    default_voices = tts_model.list_available_spks()
    # 自定义音色库，存放在voices_dir
    for name in os.listdir(f"{voices_dir}/"):
        spk_custom.append(name.replace(".pt", ""))

    # wav等文件目录列表
    #${root_dir/asset/}参考音频选择列表
    files = [(entry.name, entry.stat().st_mtime) for entry in os.scandir(f"{asset_dir}") if entry.is_file() and os.path.splitext(entry.name)[1].lower() in ['.wav', '.mp3']]
    files.sort(key=lambda x: x[0], reverse=False)  # 按名字排序
    asset_wav_list = [f[0] for f in files]

    logging.info(f"    默认音色: {default_voices}")
    logging.info(f"  自定义音色: {spk_custom}")
    logging.info(f"外置声音文件: {asset_wav_list}")

    logging.info(f'\n启动api: http://{host}:{port}\n')
    try:
        from waitress import serve
    except Exception:
        app.run(host=host, port=port)
    else:
        serve(app, host=host, port=port)

'''


## 根据内置角色合成文字
- 接口地址:  /tts
- 单纯将文字合成语音，不进行音色克隆
- 必须设置的参数：
 `text`:需要合成语音的文字
 `role`: '中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女' 选择一个
- 成功返回:wav音频数据
- 示例代码
```
data={
    "text":"你好啊亲爱的朋友们",
    "reference_audio":"10.wav"
}

response=requests.post(f'http://127.0.0.1:9933/tts',data=data,timeout=3600)
```
## 同语言克隆音色合成
- 地址：/clone_eq
参考音频发音语言和需要合成的文字语言一致，例如参考音频是中文发音，同时需要根据该音频将中文文本合成为语音
- 必须设置参数:
`text`： 需要合成语音的文字
`reference_audio`：需要克隆音色的参考音频
`reference_text`：参考音频对应的文字内容 *参考音频相对于 api.py 的路径/asset/，例如引用1.wav，该文件和api.py在同一文件夹/asset/内，则填写 `1.wav`*
- 成功返回:wav数据
- 示例代码
```
data={
    "text":"你好啊亲爱的朋友们。",
    "reference_audio":"10.wav",
    "reference_text":"希望你过的比我更好哟。"
}

response=requests.post(f'http://127.0.0.1:9933/tts',data=data,timeout=3600)
```
## 不同语言音色克隆:
- 地址： /clone  /clone_mul
参考音频发音语言和需要合成的文字语言不一致，例如需要根据中文发音的参考音频，将一段英文文本合成为语音。
- 必须设置参数:
`text`： 需要合成语音的文字
`reference_audio`：需要克隆音色的参考音频 *参考音频相对于 api.py 的路径，例如引用1.wav，该文件和api.py在同一文件夹内，则填写 `1.wav`*
- 成功返回:wav数据

- 示例代码
```
data={
    "text":"親友からの誕生日プレゼントを遠くから受け取り、思いがけないサプライズと深い祝福に、私の心は甘い喜びで満たされた！。",
    "reference_audio":"10.wav"
}

response=requests.post(f'http://127.0.0.1:9933/tts',data=data,timeout=3600)
```
## 兼容openai tts
- 接口地址 /v1/audio/speech
- 请求方法  POST
- 请求类型  Content-Type: application/json
- 请求参数
    `input`: 要合成的文字
    `model`: 固定 tts-1, 兼容openai参数，实际未使用。非tts-1, 表示指令式tts
    `speed`: 语速，默认1.0
    `reponse_format`：返回格式，固定wav音频数据
    `voice`: 仅用于文字合成时，取其一 '中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女'

> 用于克隆时，填写引用的参考音频相对于 api.py 的路径，例如引用1.wav，该文件和api.py在同一文件夹内，则填写 `1.wav`

Open WebUI请求的json只有两项："input"和"voice"
"voice"：
用于文字合成时，取其一 '中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女'
其他，用于3s克隆。表示存放在 asset/ 路径下的参考音频文件名，包括同名的附加.txt扩展名文件。可以是ffmpeg认识的格式。
如果以:分割，后边表示指令。

示例
zero_shot_prompt.wav -> /workspace/CosyVoice/asset/zero_shot_prompt.wav，以及 zero_shot_prompt.wav.txt
zero_shot_prompt.wav:上海话 -> /workspace/CosyVoice/asset/zero_shot_prompt.wav，以及 zero_shot_prompt.wav.txt，指令为“上海话”

- 示例代码

```
from openai import OpenAI

client = OpenAI(api_key='12314', base_url='http://127.0.0.1:9933/v1')
with  client.audio.speech.with_streaming_response.create(
                    model='tts-1',
                    voice='中文女',
                    input='你好啊，亲爱的朋友们',
                    speed=1.0
                ) as response:
    with open('./test.wav', 'wb') as f:
       for chunk in response.iter_bytes():
            f.write(chunk)

curl 'http://172.16.1.105:8000/v1/audio/speech' -X POST -H 'Accept-Encoding: gzip, deflate, br, zstd' -H 'Content-Type: application/json' --data-raw $'{"input":"或者如果您有具体的问题或需要帮助，请告诉我。","voice":"中文女"}' -o /tmp/nfs/a.wav
curl 'http://172.16.1.105:8000/v1/audio/speech' -X POST -H 'Accept-Encoding: gzip, deflate, br, zstd' -H 'Content-Type: application/json' --data-raw $'{"input":"或者如果您有具体的问题或需要帮助，请告诉我。","voice":"cross_lingual_prompt.wav:四川话"}' -o /tmp/nfs/a.wav

注意, 需要下面的模型文件:
pretrained_models/CosyVoice2-0.5B
使用四川话等方言增强或许需要，可以用wetext代替：
pretrained_models/CosyVoice-ttsfrd
```

'''
