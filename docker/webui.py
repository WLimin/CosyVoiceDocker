# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import platform
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import shutil
import time
from pathlib import Path
import io

# 设置环境变量禁用tokenizers并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(__file__).parent.as_posix()
print(f"ROOT_DIR={ROOT_DIR}")
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

voices_dir = Path(f'{ROOT_DIR}/pretrained_models/voices').as_posix()    #外置音色文件目录，扩展名 .pt
asset_dir = Path(f'{ROOT_DIR}/asset').as_posix()    #参考音频文件目录
print(f"voices_dir={voices_dir}\nasset_dir={asset_dir}")

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

#from modelscope import snapshot_download
#snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
# 内置音色列表文件
#try:
#    shutil.copy2('spk2info.pt', 'pretrained_models/CosyVoice2-0.5B/spk2info.pt')
#except Exception as e:
#    logging.warning(f'复制文件失败: {e}')

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮\n4. (可选)保存音色模型',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮\n3. (可选)保存音色模型',
                 '自然语言控制': '1. 选择预训练音色(V2模型需要选择或录入prompt音频)\n2. 输入instruct文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8
model_versions = None
default_seed = 21986 # 随便写的
seed = default_seed  # random.randint(1, 100000000)
#device_str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

def refresh_sft_spk():
    """刷新音色选择列表 """
    # 获取自定义音色
    files = [(entry.name, entry.stat().st_mtime) for entry in os.scandir(f"{voices_dir}")]
    files.sort(key=lambda x: x[0], reverse=False) # 按名字排序

    # 添加预训练音色
    choices = [f[0].replace(".pt", "") for f in files] + cosyvoice.list_available_spks()

    if not choices:
        choices = ['']

    return {"choices": choices, "__type__": "update"}

def refresh_sfts_prompt():
    """刷新 ${voices_dir} 提示音色选择列表"""
    files = [(entry.name, entry.stat().st_mtime) for entry in os.scandir(f"{voices_dir}") if entry.is_file() and os.path.splitext(entry.name)[1].lower() in ['.pt']]
    #files.sort(key=lambda x: x[1], reverse=True)  # 按时间排序
    files.sort(key=lambda x: x[0])
    choices = ["请选择提示音色"] + [f[0].replace(".pt", "") for f in files]

    if not choices:
        choices = ['']

    return {"choices": choices, "__type__": "update"}

def refresh_prompt_wav():
    """刷新${root_dir/asset/}参考音频选择列表"""
    files = [(entry.name, entry.stat().st_mtime) for entry in os.scandir(f"{asset_dir}") if entry.is_file() and os.path.splitext(entry.name)[1].lower() in ['.wav', '.mp3']]
    files.sort(key=lambda x: x[0], reverse=False)  # 按名字排序
    choices = ["请选择参考音频或者自己上传"] + [f[0] for f in files]

    if not choices:
        choices = ['']

    return {"choices": choices, "__type__": "update"}

def change_sfts_prompt(filename):
    """切换外置音色文件，待改进
    输入：
        外置音色文件.pt
    返回：
        临时wav文件名
    备注：
        .pt 文件中已经包含了该音色所需要的张量数据。但是存在载入位置GPU/CPU和fp32/fp16问题。

        暂时采用生成临时语音文件的办法来绕过。
    """
    full_path = f"{voices_dir}/{filename}.pt"
    if os.path.exists(full_path):
        #生成临时音频文件并返回
        ref_audio = f"/tmp/t-refaudio.wav"
        try:
            voice_data = torch.load(full_path, map_location=torch.device(device_str))
            buffer = io.BytesIO()
            audio_ref= voice_data.get('audio_ref').to('cpu')
            torchaudio.save(buffer, audio_ref, prompt_sr, format="wav")  # ERROR: Input tensor has to be on CPU.
            buffer.seek(0)
            # 打开文件用于写入二进制数据
            with open(ref_audio,'wb') as file:
                file.write(buffer.getvalue())
            full_path=ref_audio
        except Exception as e:
            logging.error(f"change_sfts_prompt 加载外置音色文件失败: {e}")
            return None
    else:
        logging.warning(f"外置音色文件不存在: {full_path}")
        return None

    return full_path

def change_prompt_wav(filename):
    """切换参考音频文件"""
    full_path = f"{asset_dir}/{filename}"
    if not os.path.exists(full_path):
        logging.warning(f"音频文件不存在: {full_path}")
        return None

    return full_path

def save_voice_model(voice_name, prompt_text, prompt_wav_upload, prompt_wav_record):
    """保存为内置音色模型"""
    if not voice_name:
        gr.Info("音色名称不能为空")
        return False
    if voice_name in cosyvoice.list_available_spks(): #是否已经存在
        gr.Info("音色名称已经存在。如更新，需删除后保存。")
        return False

    # 处理prompt音频输入
    if prompt_wav_upload is not None:
        prompt_speech_16k = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_speech_16k = prompt_wav_record
    else:
        prompt_speech_16k = None
    # 验证输入
    if not prompt_speech_16k:
        gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
        return False
    if torchaudio.info(prompt_speech_16k).sample_rate < prompt_sr:
        gr.Warning(f'prompt音频采样率{torchaudio.info(prompt_wav).sample_rate}低于{prompt_sr}')
        return False
    if prompt_text == '':
        gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
        return False
    try:
        # logging.info(prompt_text, prompt_speech_16k, voice_name)
        prompt_speech = load_wav(prompt_speech_16k, prompt_sr)
        if cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech, voice_name):
            # Hack for save, cosyvoice.py 可以不修改。
            cosyvoice.frontend.spk2info[voice_name]['embedding'] = cosyvoice.frontend.spk2info[voice_name]['llm_embedding']
            cosyvoice.frontend.spk2info[voice_name]['audio_ref'] = prompt_speech    # 为以后准备，可以不保存
            cosyvoice.frontend.spk2info[voice_name]['text_ref'] = prompt_text
            cosyvoice.save_spkinfo()
            gr.Info(f"音色成功保存为:'{voice_name}'.")
            return True
    except Exception as e:
        logging.error(f"保存音色失败: {e}")
        gr.Warning("保存音色失败")

    return False

def remove_voice_model(voice_name):
    """删除指定的内置音色模型"""
    if not voice_name:
        gr.Info("音色名称不能为空")
        return False
    if voice_name not in cosyvoice.list_available_spks(): #是否已经存在
        gr.Info(f"音色名称'{voice_name}'不存在。")
        return False
    try:
        del cosyvoice.frontend.spk2info[voice_name]
        cosyvoice.save_spkinfo()
        gr.Info(f"成功删除音色:'{voice_name}'.")
    except Exception as e:
        logging.error(f"删除音色失败: {e}")
        gr.Warning("删除音色失败")
        return False
    return True

def generate_seed():
    """生成随机种子"""
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

def change_instruction(mode_checkbox_group):
    """切换模式的处理"""
    voice_dropdown_visible = mode_checkbox_group in ['预训练音色', '3s极速复刻', '自然语言控制']
    save_btn_visible = mode_checkbox_group in ['3s极速复刻']
    return (
        instruct_dict[mode_checkbox_group],
        gr.update(visible=voice_dropdown_visible),
        gr.update(visible=save_btn_visible)
    )

def prompt_wav_recognition(prompt_wav):
    """参考或上传音频文件，用funasr识别文本
    输入：音频文件名
    返回：识别的文本
    """
    if prompt_wav is None:
        return ''

    try:
        res = asr_model.generate(
            input=prompt_wav,
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
        )
        return res[0]["text"].split('|>')[-1]
    except Exception as e:
        logging.error(f"音频识别文本失败: {e}")
        gr.Warning("识别文本失败，请检查音频是否包含人声内容")
        return ''

def load_voice_pt(full_path):
    """加载音色文件中内置的音频数据和文本（或许），16000,1ch,wav
    输入：外置音色文件全路径
    返回：临时wav文件名，参考音频识别文本
    """
    text_ref = ''
    ref_audio = f"/tmp/t-refaudio.wav"
    if os.path.exists(full_path):
        #生成临时音频文件并返回
        buffer = io.BytesIO()
        try:
            voice_data = torch.load(full_path, map_location=torch.device(device_str))
            audio_ref= voice_data.get('audio_ref').to('cpu')
            torchaudio.save(buffer, audio_ref, prompt_sr, format="wav")  # ERROR: Input tensor has to be on CPU.
            buffer.seek(0)
            # 打开文件用于写入二进制数据
            with open(ref_audio,'wb') as file:
                file.write(buffer.getvalue())

            text_ref = voice_data.get('text_ref') if voice_data else None
        except Exception as e:
            logging.error(f"change_sfts_prompt 加载外置音色文件失败: {e}")
            ref_audio = ''

    else:
        logging.warning(f"外置音色文件不存在: {full_path}")
        ref_audio = ''

    return ref_audio, text_ref

def load_voice_to_tmp(voice_data):
    """加载音色文件中内置的音频数据和文本（或许），16000,1ch,wav
    输入：内置扩展音色Tensor
    返回：临时wav文件名，参考音频识别文本
    """
    text_ref = ''
    #生成临时音频文件并返回
    ref_audio = f"/tmp/t-refaudio.wav"
    buffer = io.BytesIO()
    try:
        audio_ref= voice_data.get('audio_ref').to('cpu') # ERROR: Input tensor has to be on CPU.
        torchaudio.save(buffer, audio_ref, prompt_sr, format="wav")
        buffer.seek(0)
        # 打开文件用于写入二进制数据
        with open(ref_audio,'wb') as file:
            file.write(buffer.getvalue())

        text_ref = voice_data.get('text_ref') if voice_data else None
    except Exception as e:
        ref_audio = ''
        logging.error(f"保存音色文件失败: {e}")

    return ref_audio, text_ref

def validate_input(mode, tts_text, sft_dropdown, prompt_text, prompt_wav, instruct_text):
    """验证输入参数的合法性

    Args:
        mode: 推理模式
        tts_text: 合成文本
        sft_dropdown: 预训练音色
        prompt_text: prompt文本
        prompt_wav: prompt音频
        instruct_text: instruct文本

    Returns:
        bool: 验证是否通过
        str: 错误信息
    """

    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode in ['自然语言控制']:
        if cosyvoice.instruct is False and model_versions == 'V1':
            return False, f'您正在使用自然语言控制模式, {args.model_dir}模型不支持此模式, 请使用iic/CosyVoice-300M-Instruct模型'
        if (prompt_wav is not None or prompt_text != '') and model_versions == 'V1':
            gr.Info('您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略')
        if not instruct_text:
            return False, '您正在使用自然语言控制模式, 请输入instruct文本'
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    elif mode in ['跨语种复刻']:
        if cosyvoice.instruct is True:
            return False, f'您正在使用跨语种复刻模式, {args.model_dir}模型不支持此模式, 请使用iic/CosyVoice-300M模型'
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
        gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
        if not prompt_wav:
            return False, '您正在使用跨语种复刻模式, 请提供prompt音频'
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    elif mode in ['3s极速复刻', '跨语种复刻']:
        if (not prompt_wav) and (not is_zero_shot_spk(sft_dropdown)): # 需要修正兼容内置扩展音色
            return False, 'prompt音频为空，您是否忘记输入prompt音频？'
        if bool(prompt_wav) and (torchaudio.info(prompt_wav).sample_rate < prompt_sr):
            return False, f'prompt音频采样率{torchaudio.info(prompt_wav).sample_rate}低于{prompt_sr}'
    # sft mode only use sft_dropdown
    elif mode in ['预训练音色']:
        if instruct_text != '' or bool(prompt_wav) or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
        if not sft_dropdown:
            return False, '没有可用的预训练音色！'

    # zero_shot mode only use prompt_wav prompt text
    if mode in ['3s极速复刻']:
        if prompt_text == '' and not is_zero_shot_spk(sft_dropdown): # 需要修正兼容内置扩展音色
            return False, 'prompt文本为空，您是否忘记输入prompt文本？'
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

    return True, ''

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """音频后处理方法"""
    # 修剪静音部分
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )

    # 音量归一化
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val

    # 添加尾部静音
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

def process_audio(speech_generator, stream):
    """处理音频生成

    Args:
        speech_generator: 音频生成器
        stream: 是否流式处理

    Returns:
        tuple: (音频数据列表, 总时长)
    """
    tts_speeches = []
    total_duration = 0
    for i in speech_generator:
        tts_speeches.append(i['tts_speech'])
        total_duration += i['tts_speech'].shape[1] / cosyvoice.sample_rate
        if stream:
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten()), None

    if not stream:
        audio_data = torch.concat(tts_speeches if tts_speeches  else [torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
        yield None, (cosyvoice.sample_rate, audio_data.numpy().flatten())

    yield total_duration

def is_zero_shot_spk(id):
    """ 是spk2info 内置扩展音色 返回 True
    全局变量：cosyvoice
    """
    return True if id in cosyvoice.list_available_spks() and 'flow_prompt_speech_token' in cosyvoice.frontend.spk2info[id].keys() else False

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    """生成音频的主函数

    Args:
        tts_text: 合成文本
        mode_checkbox_group: 推理模式
        sft_dropdown: 预训练音色
        prompt_text: prompt文本
        prompt_wav_upload: 上传的prompt音频
        prompt_wav_record: 录制的prompt音频
        instruct_text: instruct文本
        seed: 随机种子
        stream: 是否流式推理
        speed: 语速

    Yields:
        tuple: 音频数据
    """
    if model_versions == 'V2':
        if stream:
            stream = False
            gr.Warning('您正在使用V2版本模型, 不支持流式推理, 将使用非流式模式.')

    start_time = time.time()
    logging.info(f"开始生成音频 - 模式: {mode_checkbox_group}, 文本长度: {len(tts_text)}")
    # 处理prompt音频输入
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    # 验证 WEB 界面输入条件是否符合要求
    is_valid, error_msg = validate_input(mode_checkbox_group, tts_text, sft_dropdown, prompt_text, prompt_wav, instruct_text)
    if not is_valid:
        gr.Warning(error_msg)
        yield (cosyvoice.sample_rate, default_data), None
        return

    # 设置随机种子
    set_all_random_seed(seed)

    # 根据不同模式处理
    if mode_checkbox_group == '预训练音色':
        # logging.info('get sft inference request')
        if sft_dropdown in cosyvoice.list_available_spks():
            logging.info(f"预训练音色:{sft_dropdown}")
            generator = cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed)
            #修改了保存的spk2info信息。若不修改，可判断是内置扩展后，调用inference_zero_shot
        else:
            # 处理外置 prompt音频输入，修改为生成临时文件
            voice_path = f"{voices_dir}/{sft_dropdown}.pt"
            logging.info(f"用'3s极速复刻'模式处理预置音色:{sft_dropdown}，需要'{voice_path}'文件。")
            [prompt_wav, prompt_text] = load_voice_pt(voice_path)
            logging.info(f"prompt_text='{prompt_text}'")
            if not prompt_wav:
                gr.Warning(f'预置音色{sft_dropdown}需要外置 {voice_path} 文件支持！')
                yield (cosyvoice.sample_rate, default_data), None
                return

            if prompt_text is None:
                gr.Warning('预置音色文件中缺少prompt_text数据！')

            prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr)) if Path(prompt_wav).exists() else None
            generator = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed)

    elif mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request')
        zero_shot_spk_id = sft_dropdown if is_zero_shot_spk(sft_dropdown) else ''
        if bool(prompt_wav):
            prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
            zero_shot_spk_id = ''
        else:
            prompt_speech_16k = None
        if zero_shot_spk_id == '' and prompt_speech_16k is None:
            logging.info(f'选择的预训练音色 {sft_dropdown} 无法正常加载数据。或需要上传wav或录音等提示音色文件！')
            gr.Warning(f'无法正常加载{sft_dropdown}数据，或需要上传wav或录音等提示音色文件！')
            yield (cosyvoice.sample_rate, default_data), None
            return
        generator = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed, zero_shot_spk_id=zero_shot_spk_id)

    elif mode_checkbox_group == '跨语种复刻':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        generator = cosyvoice.inference_cross_lingual(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed)

    elif mode_checkbox_group == '自然语言控制':
        logging.info('get instruct inference request')
        # 优先级安排：优选 sft_dropdown 内置扩展音色、外置扩展音色文件，当选择 内置sft 时，选择上传文件和录音。
        # sft_dropdown 里边有3种情况：内置sft、内置扩展音色、外置扩展音色文件
        if sft_dropdown in cosyvoice.list_available_spks(): # 内置sft、内置扩展音色
            if 'flow_prompt_speech_token' in cosyvoice.frontend.spk2info[sft_dropdown].keys(): #不是内置sft
                logging.info(f"内置扩展音色: {sft_dropdown}")
                zero_shot_spk_id = sft_dropdown
                prompt_speech_16k = None
            else: # 内置sft，‘中文女’等不支持
                logging.info(f"内置音色: {sft_dropdown}，不支持指令模式")
                gr.Warning(f"内置音色: {sft_dropdown}，不支持指令模式！将使用声音文件。")
                prompt_speech_16k = None
                zero_shot_spk_id = ''
        elif Path(f"{voices_dir}/{sft_dropdown}.pt").exists(): # 检查是否存在外置扩展音色文件.pt
            voice_path = f"{voices_dir}/{sft_dropdown}.pt"
            logging.info(f"外置扩展音色，加载文件: {voice_path}")
            [prompt_wav, prompt_speech_text] = load_voice_pt(voice_path)
            prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr)) if bool(prompt_wav) and Path(prompt_wav).exists() else None
            if prompt_speech_16k is not None:
                logging.info(f" 成功加载文件: {voice_path}")
            else:
                logging.info(f" 加载文件: {voice_path} 失败。")
            zero_shot_spk_id=''

        if zero_shot_spk_id == '' and prompt_speech_16k is None:
            #检查外置wav文件
            logging.info(f'选择外置音色，需选择wav、上传或录音。使用文件：{prompt_wav}！')
            prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr)) if bool(prompt_wav) and Path(prompt_wav).exists() else None
            zero_shot_spk_id = ''
            if zero_shot_spk_id == '' and prompt_speech_16k is None:
                logging.info(f'选择的预训练音色 {sft_dropdown} 需要上传的wav或录音等提示音色文件！')
                gr.Warning(f'无法确定{sft_dropdown}、上传的wav或录音等提示音色文件！')
                yield (cosyvoice.sample_rate, default_data), None
                return
        logging.info(f"instruct_text={instruct_text}, zero_shot_spk_id='{zero_shot_spk_id}', prompt_speech_16k is {'None' if prompt_speech_16k is None else 'not None'}.")
        if model_versions == 'V1':
            generator = cosyvoice.inference_instruct(tts_text, zero_shot_spk_id, instruct_text, stream=stream, speed=speed)
        elif model_versions == 'V2':
            # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
            # tts_text, instruct_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True
            generator = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, stream=stream, speed=speed, zero_shot_spk_id=zero_shot_spk_id)
        else:
            gr.Warning('非预期的模型版本！')
    else:
        gr.Warning('非预期的选项！')

    # 处理音频生成并获取总时长
    audio_generator = process_audio(generator, stream)
    total_duration = 0

    # 收集所有音频输出
    for output in audio_generator:
        if isinstance(output, (float, int)):  # 如果是总时长
            total_duration = output
        else:  # 如果是音频数据
            yield output

    processing_time = time.time() - start_time
    rtf = processing_time / total_duration if total_duration > 0 else 0
    logging.info(f"音频生成完成 耗时: {processing_time:.2f}秒, rtf: {rtf:.2f}")

def update_audio_visibility(stream_enabled):
    """更新音频组件的可见性"""
    return [
        gr.update(visible=stream_enabled),  # 流式音频组件
        gr.update(visible=not stream_enabled)  # 非流式音频组件
    ]

def main():
    with gr.Blocks() as demo:
        # 页面标题和说明
        gr.Markdown(f"### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) {model_versions}\
                    预训练模型 [CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) \
                    [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown(f"### 3s急速复刻后，可保存为内置扩展音色，能在语音合成、3s复刻、自然语言控制中使用。支持添加、删除等管理。")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        # 主要输入区域
        tts_text = gr.Textbox(label="输入合成文本", lines=1, value="CosyVoice迎来全面升级，提供更准、更稳、更快、 更好的语音生成能力。CosyVoice is undergoing a comprehensive upgrade, providing more accurate, stable, faster, and better voice generation capabilities.")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0], scale=1)
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=3)
            # 音色选择部分
            with gr.Column(scale=1) as choice_sft_spk:
                sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练/外置音色', value=sft_spk[0])
                refresh_voice_button = gr.Button("刷新音色")

            # 流式控制和速度调节
            with gr.Column(scale=1):
                stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1])
                speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            # 随机种子控制
            with gr.Column(scale=1):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=default_seed, label="随机推理种子")

        # 音频输入区域
        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz', scale=2)
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件', scale=2)
            with gr.Column(scale=1):
                ref_sfts_dropdown = gr.Dropdown(
                    label="外置音色列表",
                    choices=ref_sfts_prompts,
                    value="请选择提示音色",
                    interactive=True
                )
                refresh_ref_sfts_button = gr.Button("刷新提示音色")
            with gr.Column(scale=1):
                wavs_dropdown = gr.Dropdown(
                    label="参考音频列表",
                    choices=reference_wavs,
                    value="请选择参考音频或者自己上传",
                    interactive=True
                )
                refresh_button = gr.Button("刷新参考音频")

        # 文本输入区域
        with gr.Row():
            prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，支持自动识别，您可以自行修正识别结果...", value='')
            instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本。例如: 用四川话说这句话。", value='')

         # 保存音色按钮（默认隐藏）
        with gr.Row(visible=False) as save_spk_btn:
            new_name = gr.Textbox(label="输入新的音色名称", lines=1, placeholder="输入新的音色名称.", value='', scale=2)
            with gr.Column(scale=1):
                save_button = gr.Button(value="保存音色模型", scale=1)
                remove_button = gr.Button(value="删除音色模型", scale=1)

        # 生成按钮
        generate_button = gr.Button("生成音频")

        # 音频输出区域
        with gr.Group() as audio_group:
            audio_output_stream = gr.Audio(
                label="合成音频(流式)",
                value=None,
                streaming=True,
                autoplay=True,
                show_label=True,
                show_download_button=True,
                visible=False
            )
            audio_output_normal = gr.Audio(
                label="合成音频",
                value=None,
                streaming=False,
                autoplay=True,
                show_label=True,
                show_download_button=True,
                visible=True
            )

        # 绑定事件
        refresh_voice_button.click(fn=refresh_sft_spk, inputs=[], outputs=[sft_dropdown])
        refresh_button.click(fn=refresh_prompt_wav, inputs=[], outputs=[wavs_dropdown])
        wavs_dropdown.change(change_prompt_wav, inputs=[wavs_dropdown], outputs=[prompt_wav_upload])

        ref_sfts_dropdown.change(change_sfts_prompt, inputs=[ref_sfts_dropdown], outputs=[prompt_wav_upload]) #提示音色列表选择
        refresh_ref_sfts_button.click(fn=refresh_sfts_prompt, inputs=[], outputs=[ref_sfts_dropdown]) #刷新提示音色

        save_button.click(save_voice_model, inputs=[new_name, prompt_text, prompt_wav_upload, prompt_wav_record])		# 保存为内置音色
        remove_button.click(remove_voice_model, inputs=[new_name])		# 删除指定的内置音色

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output_stream, audio_output_normal])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text, choice_sft_spk, save_spk_btn])
        prompt_wav_upload.change(fn=prompt_wav_recognition, inputs=[prompt_wav_upload], outputs=[prompt_text])
        prompt_wav_record.change(fn=prompt_wav_recognition, inputs=[prompt_wav_record], outputs=[prompt_text])

        stream.change(
            fn=update_audio_visibility,
            inputs=[stream],
            outputs=[audio_output_stream, audio_output_normal]
        )

    # 配置队列和启动服务
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port, inbrowser=args.open)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--asr_model_dir',
                        type=str,
                        default='iic/SenseVoiceSmall',
                        help='local path or modelscope repo id of iic/SenseVoiceSmall')
    parser.add_argument('--open',
                        action='store_true',
                        help='open in browser')
    parser.add_argument('--log_level',
                        type=str,
                        default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='set log level')
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        cosyvoice = CosyVoice(args.model_dir)
        model_versions = 'V1'
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir, fp16=False if device_str == 'cpu' else True)
            model_versions = 'V2'
        except Exception:
            raise TypeError('no valid model_type!')

    sft_spk = refresh_sft_spk()['choices']
    reference_wavs = refresh_prompt_wav()['choices']
    ref_sfts_prompts = refresh_sfts_prompt()['choices']

    if len(sft_spk) == 0:
        sft_spk = ['']

    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)

    #model_dir = "iic/SenseVoiceSmall" # $HOME/.cache/modelscope/hub/iic/SenseVoiceSmall
    if args.asr_model_dir == 'iic/SenseVoiceSmall':
        asr_model_dir = Path(f'{ROOT_DIR}/pretrained_models/modelscope/hub/iic/SenseVoiceSmall').as_posix()
    else:
        asr_model_dir = args.asr_model_dir
    logging.info(f"device=f'{device_str}'\nasr_model_dir={asr_model_dir}")
    asr_model = AutoModel( model=asr_model_dir, disable_update=True, log_level=args.log_level, device='cpu') #device_str)
    main()
