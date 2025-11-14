import os
import sys
import io
import time
import base64
import random
import string
import logging
import json
import re
from typing import List, Dict

import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


# 路径注入，便于导入 CosyVoice SDK 与工具
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '{}/../../..'.format(ROOT_DIR))
sys.path.insert(0, '{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torch


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


# 后端 cosyvoice fastapi 服务配置（由 runtime/python/fastapi/server.py 启动）
BACKEND_HOST = os.getenv('COSYVOICE_BACKEND_HOST', '127.0.0.1')
BACKEND_PORT = int(os.getenv('COSYVOICE_BACKEND_PORT', '50000'))
BACKEND_BASE_URL = f'http://{BACKEND_HOST}:{BACKEND_PORT}'

# 模型目录（用于 SDK 注册新说话人和获取采样率）
MODEL_DIR = os.getenv('COSYVOICE_MODEL_DIR', 'iic/CosyVoice-300M')


_cv = None


def get_cosyvoice():
    global _cv
    if _cv is None:
        logger.info('Initializing CosyVoice with model_dir=%s', MODEL_DIR)
        # Try CosyVoice2 first (for newer models), fall back to CosyVoice
        try:
            _cv = CosyVoice2(MODEL_DIR)
            logger.info('Loaded CosyVoice2 model')
        except (ValueError, FileNotFoundError) as e:
            logger.info('CosyVoice2 not available (%s), trying CosyVoice', str(e))
            try:
                _cv = CosyVoice(MODEL_DIR)
                logger.info('Loaded CosyVoice model')
            except ModuleNotFoundError as me:
                if 'matcha' in str(me).lower():
                    logger.error('Matcha-TTS module not found. Please run: git submodule init && git submodule update')
                    raise RuntimeError('Matcha-TTS submodule not initialized. Run: git submodule init && git submodule update') from me
                raise
    return _cv



def _pcm16le_to_wav_bytes(pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    import wave
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return wav_io.getvalue()


def _merge_torch_audio_frames(frames: List[torch.Tensor]) -> torch.Tensor:
    # frames: list of (1, T) float tensors in [-1, 1]
    if not frames:
        return torch.zeros(1, 0)
    return torch.cat(frames, dim=1)


def _float_to_pcm16le_bytes(wav_float: torch.Tensor) -> bytes:
    # wav_float: (1, T) in [-1, 1]
    wav_np = (wav_float.clamp(-1, 1).squeeze(0).cpu().numpy() * (2 ** 15)).astype('<i2')
    return wav_np.tobytes()


def _build_subtitles_by_ratio(text: str, total_seconds: float) -> List[Dict]:
    # 按句号等分句，按字符数比例分配时长（启发式字幕）
    if not text:
        return []
    parts = [p.strip() for p in re.split(r'([。！？.!?…]+)', text)]
    # 重组为句子（内容+终止符）
    sents: List[str] = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and re.match(r'[。！？.!?…]+', parts[i + 1] or ''):
            sents.append((parts[i] or '') + (parts[i + 1] or ''))
            i += 2
        else:
            if parts[i]:
                sents.append(parts[i])
            i += 1
    sents = [s for s in sents if s]
    total_chars = sum(len(s) for s in sents) or 1
    cur = 0.0
    subs: List[Dict] = []
    for idx, s in enumerate(sents):
        dur = total_seconds * (len(s) / total_chars)
        start = cur
        end = cur + dur
        subs.append({'index': idx + 1, 'start': round(start, 3), 'end': round(end, 3), 'text': s})
        cur = end
    return subs


def _seconds_to_timestamp(seconds: float, use_comma: bool = True) -> str:
    # SRT 使用逗号，VTT 使用点
    ms = int(round(seconds * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    sep = ',' if use_comma else '.'
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _format_srt(subs: List[Dict]) -> str:
    lines: List[str] = []
    for item in subs:
        start = _seconds_to_timestamp(item['start'], use_comma=True)
        end = _seconds_to_timestamp(item['end'], use_comma=True)
        lines.append(str(item['index']))
        lines.append(f"{start} --> {end}")
        lines.append(item['text'])
        lines.append("")
    return "\n".join(lines)


def _format_vtt(subs: List[Dict]) -> str:
    lines: List[str] = ["WEBVTT", ""]
    for item in subs:
        start = _seconds_to_timestamp(item['start'], use_comma=False)
        end = _seconds_to_timestamp(item['end'], use_comma=False)
        lines.append(f"{start} --> {end}")
        lines.append(item['text'])
        lines.append("")
    return "\n".join(lines)


def _gen_spk_id(prefix: str = 'spk') -> str:
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{int(time.time())}_{suffix}"


@app.post('/register_spk')
async def register_spk(prompt_wav: UploadFile = File(),
                       prompt_text: str = Form(default=''),
                       spk_id: str = Form(default='')):
    """
    通过零样本提示音注册新说话人，返回 spk_id 并持久化到 spk2info.pt。
    注意：现有后端服务会在启动时加载 spk2info，新增说话人后需重启后端服务方可通过 /inference_sft 使用新 spk_id。
    """
    cv = get_cosyvoice()
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    if not spk_id:
        spk_id = _gen_spk_id('spk')
    ok = cv.add_zero_shot_spk(prompt_text, prompt_speech_16k, spk_id)
    if ok is not True:
        return JSONResponse(status_code=500, content={'message': 'failed to add zero shot spk'})
    cv.save_spkinfo()
    return {
        'spk_id': spk_id,
        'message': 'success, please restart backend cosyvoice server to take effect for /inference_sft'
    }


@app.get('/list_spk')
async def list_spk():
    cv = get_cosyvoice()
    try:
        spks = cv.list_available_spks()
    except Exception:
        # CosyVoice2 复用 spk2info 结构
        spks = list(getattr(cv.frontend, 'spk2info', {}).keys())
    return {'spk_ids': spks}


@app.post('/delete_spk')
async def delete_spk(spk_id: str = Form()):
    cv = get_cosyvoice()
    spk2info = getattr(cv.frontend, 'spk2info', {})
    if spk_id not in spk2info:
        return JSONResponse(status_code=404, content={'message': 'spk_id not found'})
    del spk2info[spk_id]
    cv.save_spkinfo()
    return {'message': 'deleted', 'spk_id': spk_id}



# 1) 创建命名音色 + 朗读预览文本（基于 CosyVoice2 instruct2 或 CosyVoice zero_shot）
# preview_text - required
# 用于预览朗读的文本内容
# 注册完音色后，会用这个文本生成一段预览音频，让你听听效果
# 返回的音频就是朗读这段文本的结果

@app.post('/voice/create_and_preview')
async def voice_create_and_preview(name: str = Form(),
                                  preview_text: str = Form(),
                                  prompt_wav: UploadFile = File(None),
                                  prompt_text: str = Form(default=''),
                                  instruct_text: str = Form(default='')):
    """
    创建音色并朗读预览文本。支持两种模式：
    
    模式1：零样本克隆（需要 prompt_wav + prompt_text）
    - 上传音频文件和对应文本，克隆该音色
    
    模式2：声音描述生成（需要 instruct_text，仅 CosyVoice2）
    - 不上传音频，用文字描述想要的声音特征（如"用温柔的女声"）
    
    返回注册的 spk_id（如果适用）与预览音频 Base64 WAV。
    """
    cv = get_cosyvoice()
    
    # 判断使用哪种模式
    has_prompt_wav = prompt_wav is not None and prompt_wav.filename
    has_prompt_text = prompt_text and prompt_text.strip()
    has_instruct_text = instruct_text and instruct_text.strip()
    
    # 模式1：零样本克隆
    if has_prompt_wav or has_prompt_text:
        # 验证必需参数
        if not has_prompt_wav:
            return JSONResponse(status_code=400, content={
                'message': 'prompt_wav is required when using zero-shot cloning mode.'
            })
        if not has_prompt_text:
            return JSONResponse(status_code=400, content={
                'message': 'prompt_text is required when using zero-shot cloning mode. Please provide the text content of the prompt audio.'
            })
        
        # 1) 注册/覆盖该名称的说话人（使用零样本提示音）
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        if not name:
            name = _gen_spk_id('voice')
        ok = cv.add_zero_shot_spk(prompt_text, prompt_speech_16k, name)
        if ok is not True:
            return JSONResponse(status_code=500, content={'message': 'failed to add zero shot spk'})
        cv.save_spkinfo()
        
        # 2) 朗读预览文本
        frames: List[torch.Tensor] = []
        if isinstance(cv, CosyVoice2):
            # CosyVoice2: 使用 instruct2
            for out in cv.inference_instruct2(preview_text, instruct_text, prompt_speech_16k, zero_shot_spk_id='', stream=False):
                frames.append(out['tts_speech'])
        else:
            # CosyVoice: 使用 zero_shot（不支持 instruct_text）
            for out in cv.inference_zero_shot(preview_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False):
                frames.append(out['tts_speech'])
        
        merged = _merge_torch_audio_frames(frames)
        pcm = _float_to_pcm16le_bytes(merged)
        wav_bytes = _pcm16le_to_wav_bytes(pcm, cv.sample_rate)
        total_seconds = (merged.shape[1] / cv.sample_rate) if merged.shape[1] > 0 else 0.0
        subs = _build_subtitles_by_ratio(preview_text, total_seconds)
        subs_srt = _format_srt(subs)
        subs_vtt = _format_vtt(subs)
        
        return {
            'mode': 'zero_shot',
            'spk_id': name,
            'audio_wav_base64': base64.b64encode(wav_bytes).decode('ascii'),
            'sample_rate': cv.sample_rate,
            'model_type': 'CosyVoice2' if isinstance(cv, CosyVoice2) else 'CosyVoice',
            'subtitles': subs,
            'subtitles_srt': subs_srt,
            'subtitles_vtt': subs_vtt
        }
    
    # 模式2：声音描述生成（仅 CosyVoice2）
    elif has_instruct_text:
        if not isinstance(cv, CosyVoice2):
            return JSONResponse(status_code=400, content={
                'message': 'Instruct mode requires CosyVoice2 model. Current model does not support inference_instruct.'
            })
        
        # 用 instruct 生成音频
        frames: List[torch.Tensor] = []
        try:
            for out in cv.inference_instruct(preview_text, spk_id, instruct_text, stream=False):
                frames.append(out['tts_speech'])
        except Exception as e:
            logger.error(f'inference_instruct failed: {e}')
            return JSONResponse(status_code=500, content={
                'message': f'Failed to generate audio: {str(e)}'
            })
        
        merged = _merge_torch_audio_frames(frames)
        
        # 如果提供了 name，将生成的音频注册为新的说话人
        registered_spk_id = None
        if not name:
            name = _gen_spk_id('voice')
        # 将生成的音频重采样到 16kHz 并注册
        if cv.sample_rate != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(cv.sample_rate, 16000)
            prompt_speech_16k = resampler(merged)
        else:
            prompt_speech_16k = merged

        # 使用预览文本作为提示文本注册说话人
        ok = cv.add_zero_shot_spk(preview_text, prompt_speech_16k, name)
        if ok is True:
            cv.save_spkinfo()
            registered_spk_id = name
            logger.info(f'Registered instruct-generated voice as spk_id: {name}')
        else:
            logger.warning(f'Failed to register instruct-generated voice as spk_id: {name}')
        
        pcm = _float_to_pcm16le_bytes(merged)
        wav_bytes = _pcm16le_to_wav_bytes(pcm, cv.sample_rate)
        total_seconds = (merged.shape[1] / cv.sample_rate) if merged.shape[1] > 0 else 0.0
        subs = _build_subtitles_by_ratio(preview_text, total_seconds)
        subs_srt = _format_srt(subs)
        subs_vtt = _format_vtt(subs)
        
        result = {
            'mode': 'instruct',
            'audio_wav_base64': base64.b64encode(wav_bytes).decode('ascii'),
            'sample_rate': cv.sample_rate,
            'model_type': 'CosyVoice2',
            'instruct_text': instruct_text,
            'subtitles': subs,
            'spk_id': name,
            'subtitles_srt': subs_srt,
            'subtitles_vtt': subs_vtt,
            'message': f'Voice registered as {registered_spk_id}, can be reused with /voice/tts_by_name'
        }
        return result
    
    # 两种模式都没有提供必需参数
    else:
        return JSONResponse(status_code=400, content={
            'message': 'Please provide either (prompt_wav + prompt_text) for zero-shot cloning, or (instruct_text) for voice description mode.'
        })


# 2) 使用名称朗读任意文本（复用 1 中注册的音色）
@app.post('/voice/tts_by_name')
async def voice_tts_by_name(name: str = Form(),
                            tts_text: str = Form(),
                            instruct_text: str = Form(default='')):
    """
    使用已注册名称的音色朗读文本。
    CosyVoice2 使用 instruct2；CosyVoice 使用 zero_shot（忽略 instruct_text）。
    """
    cv = get_cosyvoice()
    
    # 检查说话人是否存在
    if name not in cv.frontend.spk2info:
        return JSONResponse(status_code=404, content={'message': f'spk_id {name} not found, please register first'})

    frames: List[torch.Tensor] = []
    # 使用已注册的说话人ID进行推理
    # 通过add_zero_shot_spk注册的说话人使用零样本推理
    # 使用空的prompt_speech因为说话人信息已经在spk2info中
    dummy_prompt = torch.zeros(1, 16000)
    for out in cv.inference_zero_shot(tts_text, '', dummy_prompt, zero_shot_spk_id=name, stream=False):
        frames.append(out['tts_speech'])
    
    merged = _merge_torch_audio_frames(frames)
    pcm = _float_to_pcm16le_bytes(merged)
    wav_bytes = _pcm16le_to_wav_bytes(pcm, cv.sample_rate)
    total_seconds = (merged.shape[1] / cv.sample_rate) if merged.shape[1] > 0 else 0.0
    subs = _build_subtitles_by_ratio(tts_text, total_seconds)
    subs_srt = _format_srt(subs)
    subs_vtt = _format_vtt(subs)
    return {
        'spk_id': name,
        'audio_wav_base64': base64.b64encode(wav_bytes).decode('ascii'),
        'sample_rate': cv.sample_rate,
        'model_type': 'CosyVoice2' if isinstance(cv, CosyVoice2) else 'CosyVoice',
        'instruct_text': instruct_text if isinstance(cv, CosyVoice2) else 'N/A (CosyVoice does not support instruct)',
        'subtitles': subs,
        'subtitles_srt': subs_srt,
        'subtitles_vtt': subs_vtt
    }


# 3) 使用名称朗读结构化文本（情绪/口音/语气/停顿）
@app.post('/voice/tts_by_name_structured')
async def voice_tts_by_name_structured(name: str = Form(),
                                       segments: str = Form(),  # JSON 字符串：{"segments":[{"text":"...","instruct_text":"...","pause_ms":300}, ...]}
                                       default_instruct_text: str = Form(default='')):
    """
    结构化朗读：segments 为 JSON 字符串，形如：
      {
        "segments": [
          {"text": "第一句", "instruct_text": "更开心", "pause_ms": 300},
          {"text": "第二句", "instruct_text": "用四川口音"}
        ]
      }
    未提供 instruct_text 的片段将回落到 default_instruct_text。
    将在片段间插入指定时长静音（pause_ms, 毫秒）。
    CosyVoice2 支持 instruct；CosyVoice 忽略 instruct_text。
    """
    cv = get_cosyvoice()
    
    # 检查说话人是否存在
    if name not in cv.frontend.spk2info:
        return JSONResponse(status_code=404, content={'message': f'spk_id {name} not found, please register first'})

    try:
        spec = json.loads(segments)
        segs = spec.get('segments', [])
        assert isinstance(segs, list)
    except Exception:
        return JSONResponse(status_code=400, content={'message': 'invalid segments json'})

    sr = cv.sample_rate
    frames: List[torch.Tensor] = []
    # 精确字幕：逐段统计音频长度
    subs: List[Dict] = []
    cur_time = 0.0
    
    for idx, seg in enumerate(segs):
        text = seg.get('text', '') or ''
        instr = (seg.get('instruct_text') or default_instruct_text or '')
        seg_frames: List[torch.Tensor] = []
        
        if text.strip():
            # 使用已注册的说话人ID进行推理
            # 通过add_zero_shot_spk注册的说话人使用零样本推理
            # 使用空的prompt_speech因为说话人信息已经在spk2info中
            dummy_prompt = torch.zeros(1, 16000)
            for out in cv.inference_zero_shot(text, '', dummy_prompt, zero_shot_spk_id=name, stream=False):
                seg_frames.append(out['tts_speech'])
            
            if seg_frames:
                seg_tensor = _merge_torch_audio_frames(seg_frames)
                frames.append(seg_tensor)
                seg_dur = seg_tensor.shape[1] / sr
                subs.append({'index': len(subs) + 1, 'start': round(cur_time, 3), 'end': round(cur_time + seg_dur, 3), 'text': text})
                cur_time += seg_dur
        
        pause_ms = int(seg.get('pause_ms', 0) or 0)
        if pause_ms > 0:
            pause_len = int(sr * (pause_ms / 1000.0))
            if pause_len > 0:
                frames.append(torch.zeros(1, pause_len))
                cur_time += (pause_len / sr)

    merged = _merge_torch_audio_frames(frames)
    pcm = _float_to_pcm16le_bytes(merged)
    wav_bytes = _pcm16le_to_wav_bytes(pcm, sr)
    subs_srt = _format_srt(subs)
    subs_vtt = _format_vtt(subs)
    return {
        'spk_id': name,
        'audio_wav_base64': base64.b64encode(wav_bytes).decode('ascii'),
        'sample_rate': sr,
        'model_type': 'CosyVoice2' if isinstance(cv, CosyVoice2) else 'CosyVoice',
        'subtitles': subs,
        'subtitles_srt': subs_srt,
        'subtitles_vtt': subs_vtt
    }


# 4) 根据声音提示词生成音频-不注册spk_id（无需提示音频）
@app.post('/voice/tts_by_instruct')
async def voice_tts_by_instruct(instruct_text: str = Form(),
                                tts_text: str = Form()):
    """
    根据声音提示词（如"用温柔的女声"、"用低沉的男声"、"用四川口音"等）生成音频。
    仅支持 CosyVoice2 模型的 inference_instruct 方法。
    不需要上传提示音频，模型会根据描述自动生成对应音色。
    
    参数：
    - instruct_text: 声音提示词，描述想要的音色特征
    - tts_text: 要朗读的文本内容
    - spk_id: 可选，指定预训练说话人ID（如果为空则使用默认）
    """
    cv = get_cosyvoice()
    
    # 检查是否为 CosyVoice2 模型
    if not isinstance(cv, CosyVoice2):
        return JSONResponse(status_code=400, content={
            'message': 'This feature requires CosyVoice2 model. Current model does not support inference_instruct.'
        })
    
    if not instruct_text or not instruct_text.strip():
        return JSONResponse(status_code=400, content={
            'message': 'instruct_text is required. Please provide voice description like "用温柔的女声" or "用低沉的男声".'
        })
    
    if not tts_text or not tts_text.strip():
        return JSONResponse(status_code=400, content={
            'message': 'tts_text is required. Please provide the text to be spoken.'
        })
    
    frames: List[torch.Tensor] = []
    try:
        for out in cv.inference_instruct(tts_text, spk_id, instruct_text, stream=False):
            frames.append(out['tts_speech'])
    except Exception as e:
        logger.error(f'inference_instruct failed: {e}')
        return JSONResponse(status_code=500, content={
            'message': f'Failed to generate audio: {str(e)}'
        })
    
    merged = _merge_torch_audio_frames(frames)
    pcm = _float_to_pcm16le_bytes(merged)
    wav_bytes = _pcm16le_to_wav_bytes(pcm, cv.sample_rate)
    total_seconds = (merged.shape[1] / cv.sample_rate) if merged.shape[1] > 0 else 0.0
    subs = _build_subtitles_by_ratio(tts_text, total_seconds)
    subs_srt = _format_srt(subs)
    subs_vtt = _format_vtt(subs)
    
    return {
        'audio_wav_base64': base64.b64encode(wav_bytes).decode('ascii'),
        'sample_rate': cv.sample_rate,
        'instruct_text': instruct_text,
        'subtitles': subs,
        'subtitles_srt': subs_srt,
        'subtitles_vtt': subs_vtt
    }



if __name__ == '__main__':
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50010)
    parser.add_argument('--backend_host', type=str, default=BACKEND_HOST)
    parser.add_argument('--backend_port', type=int, default=BACKEND_PORT)
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR)
    args = parser.parse_args()
    os.environ['COSYVOICE_BACKEND_HOST'] = args.backend_host
    os.environ['COSYVOICE_BACKEND_PORT'] = str(args.backend_port)
    os.environ['COSYVOICE_MODEL_DIR'] = args.model_dir
    uvicorn.run(app, host='0.0.0.0', port=args.port)


