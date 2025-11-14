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
MODEL_DIR_V2 = os.getenv('COSYVOICE_MODEL_DIR_V2', 'iic/CosyVoice2-0.5B')


_cv = None
_cv2 = None


def get_cosyvoice():
    """获取 CosyVoice 实例（降级方案）"""
    global _cv
    if _cv is None:
        logger.info('Initializing CosyVoice with model_dir=%s', MODEL_DIR)
        try:
            _cv = CosyVoice(MODEL_DIR)
            logger.info('Loaded CosyVoice model')
        except ModuleNotFoundError as me:
            if 'matcha' in str(me).lower():
                logger.error('Matcha-TTS module not found. Please run: git submodule init && git submodule update')
                raise RuntimeError('Matcha-TTS submodule not initialized. Run: git submodule init && git submodule update') from me
            raise
        except Exception as e:
            logger.error(f'Failed to load CosyVoice: {e}')
            raise
    return _cv


def get_cosyvoice2():
    """获取 CosyVoice2 实例（优先使用）"""
    global _cv2
    if _cv2 is None:
        logger.info('Initializing CosyVoice2 with model_dir=%s', MODEL_DIR_V2)
        try:
            _cv2 = CosyVoice2(MODEL_DIR_V2)
            logger.info('Loaded CosyVoice2 model')
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f'CosyVoice2 not available: {e}')
            return None
        except Exception as e:
            logger.warning(f'Failed to load CosyVoice2: {e}')
            return None
    return _cv2


def get_primary_model():
    """
    获取主模型：优先使用 CosyVoice2，不可用时降级到 CosyVoice
    """
    cv2 = get_cosyvoice2()
    if cv2 is not None:
        return cv2
    logger.info('CosyVoice2 not available, falling back to CosyVoice')
    return get_cosyvoice()


def get_instruct_model():
    """
    获取支持 instruct 的模型（CosyVoice-Instruct）
    """
    cv = get_cosyvoice()
    if hasattr(cv, 'instruct') and cv.instruct:
        return cv
    return None



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
    cv = get_primary_model()
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    if not spk_id:
        spk_id = _gen_spk_id('spk')
    ok = cv.add_zero_shot_spk(prompt_text, prompt_speech_16k, spk_id)
    if ok is not True:
        return JSONResponse(status_code=500, content={'message': 'failed to add zero shot spk'})
    cv.save_spkinfo()
    return {
        'spk_id': spk_id,
        'model_type': 'CosyVoice2' if isinstance(cv, CosyVoice2) else 'CosyVoice',
        'message': 'success, please restart backend cosyvoice server to take effect for /inference_sft'
    }


@app.get('/list_spk')
async def list_spk():
    cv = get_primary_model()
    try:
        spks = cv.list_available_spks()
    except Exception:
        # CosyVoice2 复用 spk2info 结构
        spks = list(getattr(cv.frontend, 'spk2info', {}).keys())
    return {
        'spk_ids': spks,
        'model_type': 'CosyVoice2' if isinstance(cv, CosyVoice2) else 'CosyVoice'
    }


@app.post('/delete_spk')
async def delete_spk(spk_id: str = Form()):
    cv = get_primary_model()
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
    - 使用 CosyVoice 模型
    
    模式2：声音描述生成（需要 instruct_text）
    - 不上传音频，用文字描述想要的声音特征（如"用温柔的女声"）
    - 自动使用 CosyVoice2 模型
    
    返回注册的 spk_id 与预览音频 Base64 WAV。
    """
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
        
        # 使用主模型进行零样本克隆（优先 CosyVoice2）
        cv = get_primary_model()
        
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
            # CosyVoice2: 使用 instruct2（支持 instruct_text）
            for out in cv.inference_instruct2(preview_text, instruct_text, prompt_speech_16k, zero_shot_spk_id='', stream=False):
                frames.append(out['tts_speech'])
        else:
            # CosyVoice: 使用 zero_shot
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
    
    # 模式2：声音描述生成（需要 CosyVoice-Instruct 模型）
    elif has_instruct_text:
        frames: List[torch.Tensor] = []
        used_model = None
        used_sample_rate = 0
        base_spk_id_resolved = None

        # 优先尝试使用 CosyVoice2 的 instruct2（需要已有说话人信息）
        cv2_try = get_cosyvoice2()
        if cv2_try is not None:
            candidate_spk = (name or '').strip()
            spk2info_cv2 = getattr(cv2_try.frontend, 'spk2info', {})
            if candidate_spk and candidate_spk not in spk2info_cv2:
                logger.warning(f'spk_id {candidate_spk} not found in CosyVoice2, ignoring provided spk_id')
                candidate_spk = ''
            if not candidate_spk and spk2info_cv2:
                candidate_spk = next(iter(spk2info_cv2.keys()))
                logger.info(f'Using default CosyVoice2 spk_id: {candidate_spk}')
            if candidate_spk:
                try:
                    dummy_prompt = torch.zeros(1, 16000)
                    for out in cv2_try.inference_instruct2(preview_text, instruct_text, dummy_prompt,
                                                           zero_shot_spk_id=candidate_spk, stream=False):
                        frames.append(out['tts_speech'])
                    if frames:
                        used_model = cv2_try
                        used_sample_rate = cv2_try.sample_rate
                        base_spk_id_resolved = candidate_spk
                except Exception as e:
                    logger.warning(f'CosyVoice2 instruct2 failed: {e}, falling back to CosyVoice-Instruct')
                    frames = []

        # 若 CosyVoice2 不可用或失败，则尝试 CosyVoice-Instruct
        if used_model is None:
            cv_instruct = get_instruct_model()
            if cv_instruct is None:
                return JSONResponse(status_code=400, content={
                    'message': 'Voice description mode requires CosyVoice-Instruct model or a CosyVoice2 speaker. Please use a model with "-Instruct" in its name, provide a valid spk_id, or switch to zero-shot cloning mode with prompt_wav + prompt_text.'
                })

            candidate_spk = (name or '').strip()
            spk2info_inst = getattr(cv_instruct.frontend, 'spk2info', {})
            if candidate_spk and candidate_spk not in spk2info_inst:
                logger.warning(f'spk_id {candidate_spk} not found in CosyVoice-Instruct, ignoring provided spk_id')
                candidate_spk = ''
            if not candidate_spk:
                try:
                    available_spks = cv_instruct.list_available_spks()
                    if available_spks:
                        candidate_spk = available_spks[0]
                        logger.info(f'Using default CosyVoice-Instruct spk_id: {candidate_spk}')
                except Exception:
                    if spk2info_inst:
                        candidate_spk = list(spk2info_inst.keys())[0]
                        logger.info(f'Using default CosyVoice-Instruct spk_id from spk2info: {candidate_spk}')

            if not candidate_spk:
                return JSONResponse(status_code=400, content={
                    'message': 'No speaker available. Please provide spk_id or use zero-shot cloning mode with prompt_wav + prompt_text.'
                })

            try:
                for out in cv_instruct.inference_instruct(preview_text, candidate_spk, instruct_text, stream=False):
                    frames.append(out['tts_speech'])
            except Exception as e:
                logger.error(f'inference_instruct failed: {e}')
                return JSONResponse(status_code=500, content={
                    'message': f'Failed to generate audio: {str(e)}'
                })

            used_model = cv_instruct
            used_sample_rate = cv_instruct.sample_rate
            base_spk_id_resolved = candidate_spk

        if used_model is None or not frames:
            return JSONResponse(status_code=500, content={
                'message': 'Failed to generate audio with provided instruct_text.'
            })

        merged = _merge_torch_audio_frames(frames)

        # 如果提供了 name，将生成的音频注册（用于后续复用）
        registered_spk_id = None
        if name:
            if used_sample_rate != 16000:
                import torchaudio
                resampler = torchaudio.transforms.Resample(used_sample_rate, 16000)
                prompt_speech_16k = resampler(merged)
            else:
                prompt_speech_16k = merged
            ok = used_model.add_zero_shot_spk(preview_text, prompt_speech_16k, name)
            if ok is True:
                used_model.save_spkinfo()
                registered_spk_id = name
                logger.info(f'Registered instruct-generated voice as spk_id: {name}')
            else:
                logger.warning(f'Failed to register voice as spk_id: {name}')

        pcm = _float_to_pcm16le_bytes(merged)
        wav_bytes = _pcm16le_to_wav_bytes(pcm, used_sample_rate)
        total_seconds = (merged.shape[1] / used_sample_rate) if merged.shape[1] > 0 else 0.0
        subs = _build_subtitles_by_ratio(preview_text, total_seconds)
        subs_srt = _format_srt(subs)
        subs_vtt = _format_vtt(subs)

        if isinstance(used_model, CosyVoice2):
            model_type = 'CosyVoice2'
        else:
            model_type = 'CosyVoice-Instruct' if getattr(used_model, 'instruct', False) else 'CosyVoice'

        result = {
            'mode': 'instruct',
            'audio_wav_base64': base64.b64encode(wav_bytes).decode('ascii'),
            'sample_rate': used_sample_rate,
            'model_type': model_type,
            'instruct_text': instruct_text,
            'base_spk_id': base_spk_id_resolved,
            'subtitles': subs,
            'subtitles_srt': subs_srt,
            'subtitles_vtt': subs_vtt
        }

        if registered_spk_id:
            result['spk_id'] = registered_spk_id
            result['message'] = f'Voice registered as {registered_spk_id}, can be reused with /voice/tts_by_name'

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
    优先使用 CosyVoice2，支持 instruct_text；降级到 CosyVoice 时忽略 instruct_text。
    """
    cv = get_primary_model()
    
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
    优先使用 CosyVoice2（支持 instruct）；降级到 CosyVoice 时忽略 instruct_text。
    """
    cv = get_primary_model()
    
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


