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

    print("=" * 60)
    print("示例1：零样本克隆模式（需要提示音频）")
    print("=" * 60)
    
    # 1) 使用零样本提示音创建命名音色并生成预览
    create_url = f"{base_url}/voice/create_and_preview"
    if os.path.exists(args.prompt_wav):
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
        print(f"[create_and_preview] mode={j.get('mode')}, spk_id={spk_id}")
        print(f"  └─ Preview audio saved -> {args.out_preview}, sample_rate={j.get('sample_rate')}")
        save_subtitles(j, args.out_preview)

        # 2) 使用名称合成正式文本（不再上传提示音）
        print("\n使用已注册的音色生成新文本...")
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
    else:
        print(f"提示音频文件不存在: {args.prompt_wav}")
        print("跳过零样本克隆模式示例\n")

    print("\n" + "=" * 60)
    print("示例2：声音描述模式（无需提示音频，仅 CosyVoice2）")
    print("=" * 60)
    
    # 3) 使用声音描述生成音频并注册音色
    instruct_spk_id = None
    if args.instruct_only_text:
        data = {
            'name': args.instruct_voice_name,  # 提供 name 参数以注册音色
            'preview_text': args.instruct_only_text,
            'instruct_text': args.instruct_only_desc,
            'spk_id': args.instruct_only_spk
        }
        resp = requests.post(create_url, data=data, timeout=300)
        if resp.status_code == 200:
            j = resp.json()
            save_b64_wav(j['audio_wav_base64'], args.out_instruct)
            print(f"[create_and_preview] mode={j.get('mode')}, instruct_text='{j.get('instruct_text')}'")
            print(f"  └─ Audio saved -> {args.out_instruct}, sample_rate={j.get('sample_rate')}")
            if 'spk_id' in j:
                instruct_spk_id = j['spk_id']
                print(f"  └─ Voice registered as: {instruct_spk_id}")
                print(f"  └─ {j.get('message', '')}")
            save_subtitles(j, args.out_instruct)
            
            # 3.1) 复用刚才创建的音色生成新文本
            if instruct_spk_id and args.instruct_reuse_text:
                print(f"\n复用音色 '{instruct_spk_id}' 生成新文本...")
                tts_url = f"{base_url}/voice/tts_by_name"
                data = {
                    'name': instruct_spk_id,
                    'tts_text': args.instruct_reuse_text
                }
                resp = requests.post(tts_url, data=data, timeout=300)
                if resp.status_code == 200:
                    j = resp.json()
                    save_b64_wav(j['audio_wav_base64'], args.out_instruct_reuse)
                    print(f"[tts_by_name] saved -> {args.out_instruct_reuse}, sample_rate={j.get('sample_rate')}")
                    save_subtitles(j, args.out_instruct_reuse)
                else:
                    print(f"复用音色失败: {resp.status_code}")
        else:
            print(f"声音描述模式失败: {resp.status_code}")
            print(f"  └─ {resp.json().get('message', 'Unknown error')}")
    else:
        print("未提供 --instruct_only_text 参数，跳过声音描述模式示例")

    print("\n" + "=" * 60)
    print("示例3：直接使用声音提示词生成（tts_by_instruct）")
    print("=" * 60)
    
    # 4) 使用 tts_by_instruct 接口
    if args.direct_instruct_text:
        instruct_url = f"{base_url}/voice/tts_by_instruct"
        data = {
            'instruct_text': args.direct_instruct_desc,
            'tts_text': args.direct_instruct_text,
            'spk_id': args.direct_instruct_spk
        }
        resp = requests.post(instruct_url, data=data, timeout=300)
        if resp.status_code == 200:
            j = resp.json()
            save_b64_wav(j['audio_wav_base64'], args.out_direct_instruct)
            print(f"[tts_by_instruct] instruct_text='{j.get('instruct_text')}'")
            print(f"  └─ Audio saved -> {args.out_direct_instruct}, sample_rate={j.get('sample_rate')}")
            save_subtitles(j, args.out_direct_instruct)
        else:
            print(f"tts_by_instruct 失败: {resp.status_code}")
            print(f"  └─ {resp.json().get('message', 'Unknown error')}")
    else:
        print("未提供 --direct_instruct_text 参数，跳过 tts_by_instruct 示例")

    # 5) 可选：列出当前已注册名称
    print("\n" + "=" * 60)
    try:
        list_url = f"{base_url}/list_spk"
        j = requests.get(list_url, timeout=30).json()
        print(f"[list_spk] 已注册的音色: {j.get('spk_ids', [])}")
    except Exception as e:
        print(f"[list_spk] 获取失败: {e}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CosyVoice 业务接口调用示例')
    
    # 基础配置
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器地址')
    parser.add_argument('--port', type=int, default=50010, help='服务器端口')
    
    # 示例1：零样本克隆模式
    parser.add_argument('--name', type=str, default='voice_demo', help='音色名称')
    parser.add_argument('--prompt_wav', type=str, default='../../../asset/zero_shot_prompt.wav', help='提示音频文件路径')
    parser.add_argument('--preview_text', type=str, default='你好，这是音色预览。', help='预览文本')
    parser.add_argument('--prompt_text', type=str, default='希望你以后能够做的比我还好呦。', help='提示音频对应的文本')
    parser.add_argument('--instruct_text', type=str, default='', help='声音指令（可选）')
    parser.add_argument('--tts_text', type=str, default='现在开始正式的文本合成演示。', help='正式合成的文本')
    parser.add_argument('--instruct_text_tts', type=str, default='', help='正式合成时的声音指令（可选）')
    parser.add_argument('--out_preview', type=str, default='preview.wav', help='预览音频输出路径')
    parser.add_argument('--out_tts', type=str, default='tts.wav', help='正式音频输出路径')
    
    # 示例2：声音描述模式（通过 create_and_preview）
    parser.add_argument('--instruct_voice_name', type=str, default='gentle_female', help='声音描述模式注册的音色名称')
    parser.add_argument('--instruct_only_text', type=str, default='今天天气真不错，适合出去走走。', help='声音描述模式要朗读的文本')
    parser.add_argument('--instruct_only_desc', type=str, default='用温柔的女声', help='声音描述（如"用温柔的女声"）')
    parser.add_argument('--instruct_only_spk', type=str, default='', help='声音描述模式使用的预训练说话人ID（可选）')
    parser.add_argument('--out_instruct', type=str, default='instruct_mode.wav', help='声音描述模式输出路径')
    parser.add_argument('--instruct_reuse_text', type=str, default='这是用同一个音色生成的新内容。', help='复用声音描述模式创建的音色生成的文本')
    parser.add_argument('--out_instruct_reuse', type=str, default='instruct_reuse.wav', help='复用音色输出路径')
    
    # 示例3：直接使用 tts_by_instruct
    parser.add_argument('--direct_instruct_text', type=str, default='欢迎使用语音合成系统，祝您使用愉快。', help='tts_by_instruct 要朗读的文本')
    parser.add_argument('--direct_instruct_desc', type=str, default='用低沉的男声', help='tts_by_instruct 的声音描述')
    parser.add_argument('--direct_instruct_spk', type=str, default='', help='tts_by_instruct 使用的预训练说话人ID（可选）')
    parser.add_argument('--out_direct_instruct', type=str, default='direct_instruct.wav', help='tts_by_instruct 输出路径')
    
    args = parser.parse_args()
    main()


