import os
import argparse
import base64
import json
import requests


def save_b64_wav(b64_str: str, out_path: str) -> None:
    audio_bytes = base64.b64decode(b64_str)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(audio_bytes)


def save_subtitles(response_json: dict, base_path: str) -> None:
    """保存字幕文件（SRT和VTT格式）"""
    if 'subtitles_srt' in response_json:
        srt_path = base_path.rsplit('.', 1)[0] + '.srt'
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(response_json['subtitles_srt'])
        print(f"  └─ Subtitles (SRT) saved -> {srt_path}")
    
    if 'subtitles_vtt' in response_json:
        vtt_path = base_path.rsplit('.', 1)[0] + '.vtt'
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write(response_json['subtitles_vtt'])
        print(f"  └─ Subtitles (VTT) saved -> {vtt_path}")
    
    if 'subtitles' in response_json:
        json_path = base_path.rsplit('.', 1)[0] + '_subtitles.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(response_json['subtitles'], f, ensure_ascii=False, indent=2)
        print(f"  └─ Subtitles (JSON) saved -> {json_path}")


def main():
    host = args.host
    port = args.port
    base_url = f"http://{host}:{port}"

    # 1) 使用零样本提示音创建命名音色并生成预览
    create_url = f"{base_url}/voice/create_and_preview"
    with open(args.prompt_wav, 'rb') as f:
        files = {
            'prompt_wav': ('prompt.wav', f, 'audio/wav')
        }
        data = {
            'name': args.name,
            'preview_text': args.preview_text,
            'prompt_text': args.prompt_text,
            'instruct_text': args.instruct_text
        }
        resp = requests.post(create_url, data=data, files=files, timeout=300)
    resp.raise_for_status()
    j = resp.json()
    spk_id = j.get('spk_id', args.name)
    save_b64_wav(j['audio_wav_base64'], args.out_preview)
    print(f"[create_and_preview] spk_id={spk_id}, preview saved -> {args.out_preview}, sample_rate={j.get('sample_rate')}")
    save_subtitles(j, args.out_preview)

    # 2) 使用名称合成正式文本（不再上传提示音）
    tts_url = f"{base_url}/voice/tts_by_name"
    data = {
        'name': spk_id,
        'tts_text': args.tts_text,
        'instruct_text': args.instruct_text_tts or ''
    }
    resp = requests.post(tts_url, data=data, timeout=300)
    resp.raise_for_status()
    j = resp.json()
    save_b64_wav(j['audio_wav_base64'], args.out_tts)
    print(f"[tts_by_name] saved -> {args.out_tts}, sample_rate={j.get('sample_rate')}")
    save_subtitles(j, args.out_tts)

    # 3) 可选：列出当前已注册名称
    try:
        list_url = f"{base_url}/list_spk"
        j = requests.get(list_url, timeout=30).json()
        print(f"[list_spk] {j}")
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=50010)
    parser.add_argument('--name', type=str, default='voice_demo')
    parser.add_argument('--prompt_wav', type=str, default='../../../asset/zero_shot_prompt.wav')
    parser.add_argument('--preview_text', type=str, default='你好，这是音色预览。')
    parser.add_argument('--prompt_text', type=str, default='希望你以后能够做的比我还好呦。')
    parser.add_argument('--instruct_text', type=str, default='')
    parser.add_argument('--tts_text', type=str, default='现在开始正式的文本合成演示。')
    parser.add_argument('--instruct_text_tts', type=str, default='')
    parser.add_argument('--out_preview', type=str, default='preview.wav')
    parser.add_argument('--out_tts', type=str, default='tts.wav')
    args = parser.parse_args()
    main()


