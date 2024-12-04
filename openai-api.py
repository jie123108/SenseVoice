import logging
log_format = "%(asctime)s | %(process)d | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
logging.basicConfig(level="INFO", format=log_format)
import os
import time
import shutil
import uvicorn
import torch
import torchaudio
import numpy as np
from funasr import AutoModel
from typing import Union
from datetime import datetime
from pydub import AudioSegment
from model import SenseVoiceSmall
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, status
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Initialize the model
device = os.getenv("SENSEVOICE_DEVICE", "cpu")
model_name = "iic/SenseVoiceSmall"
model, init_kwargs = SenseVoiceSmall.from_pretrained(model=model_name,
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    device=device)
model.eval()

emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "🤧",
}

emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

lang_dict = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {
    "🎼",
    "👏",
    "😀",
    "😭",
    "🤧",
    "😷",
}


def format_str_v2(text: str, show_emo=True, show_event=True):
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = text.count(sptk)
        text = text.replace(sptk, "")

    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    if show_emo:
        text = text + emo_dict[emo]

    for e in event_dict:
        if sptk_dict[e] > 0 and show_event:
            text = event_dict[e] + text

    for emoji in emo_set.union(event_set):
        text = text.replace(" " + emoji, emoji)
        text = text.replace(emoji + " ", emoji)

    return text.strip()


def format_str_v3(text: str, show_emo=True, show_event=True):
    def get_emo(s):
        return s[-1] if s[-1] in emo_set else None

    def get_event(s):
        return s[0] if s[0] in event_set else None

    text = text.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        text = text.replace(lang, "<|lang|>")
    parts = [format_str_v2(part, show_emo, show_event).strip(" ") for part in text.split("<|lang|>")]
    new_s = " " + parts[0]
    cur_ent_event = get_event(new_s)
    for i in range(1, len(parts)):
        if len(parts[i]) == 0:
            continue
        if get_event(parts[i]) == cur_ent_event and get_event(parts[i]) is not None:
            parts[i] = parts[i][1:]
        cur_ent_event = get_event(parts[i])
        if get_emo(parts[i]) is not None and get_emo(parts[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += parts[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()


def model_inference(input_wav, language, show_emo=True, show_event=True, output_timestamp=False):
    language = "auto" if len(language) < 1 else language

    if isinstance(input_wav, tuple):
        fs, input_wav = input_wav
        input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
    if len(input_wav) == 0:
        raise ValueError("The provided audio is empty.")

    asr_result = model.inference(
        data_in=input_wav,
        language=language, # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=0,
        output_timestamp=output_timestamp,
        **init_kwargs,
    )

    text = asr_result[0][0]["text"]
    text = format_str_v3(text, show_emo, show_event)
    timestampInfo = None
    if output_timestamp:
        timestampInfo = asr_result[0][0]["timestamp"]
    result = {
        "text": text,
    }
    if timestampInfo:
        result["timestamp"] = timestampInfo

    return result


def audio2text(file_obj, language="auto", timestamp=False):
    try:
        start_time = time.time()
        # 直接从文件对象读取音频
        audio = AudioSegment.from_file(file_obj)
        y = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate

        # 转换为单声道
        if audio.channels > 1:
            y = y.reshape((-1, audio.channels)).mean(axis=1)

        audio_processing_time = time.time() - start_time

        # 语音识别
        input_wav = (sr, y)
        recognition_start_time = time.time()
        result = model_inference(input_wav=input_wav, language=language, show_emo=False, output_timestamp=timestamp)
        recognition_time = time.time() - recognition_start_time

        # 计算音频时长（秒）
        audio_duration = len(audio) / 1000.0
        text = result.get("text", "")
        logging.info(f"音频时长: {audio_duration:.2f}秒, 音频处理: {audio_processing_time:.2f}秒, 语音识别: {recognition_time:.2f}秒, 识别结果: {text}")
        return result
    except Exception as e:
        logging.error(f"无法加载音频文件, err：{str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"无法加载音频文件：{str(e)}")


# 创建线程池，线程数可以根据CPU核心数和内存调整
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

@app.post("/v1/audio/transcriptions")
async def transcriptions(file: Union[UploadFile, None] = File(default=None),
                         language: str = Form(default="auto"), timestamp: bool = Form(default=False)):
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="未提供文件")

    try:
        # 使用线程池异步执行音频处理任务
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(thread_pool, audio2text, file.file, language, timestamp)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
