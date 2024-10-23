
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
from pathlib import Path
from typing import Union
from datetime import datetime
from pydub import AudioSegment
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, status

app = FastAPI()

TMP_DIR = os.environ.get("TMP_DIR", os.path.join(os.path.expanduser("~"), "Downloads", "Audios"))
os.makedirs(TMP_DIR, exist_ok=True)
logging.info(f"临时目录已创建: {TMP_DIR}")

# Initialize the model
model = "iic/SenseVoiceSmall"
model = AutoModel(
    model=model,
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
)

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


def model_inference(input_wav, language, show_emo=True, show_event=True):
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

    merge_vad = True
    text = model.generate(
        input=input_wav,
        cache={},
        language=language,
        use_itn=True,
        batch_size_s=0,
        merge_vad=merge_vad,
    )

    text = text[0]["text"]
    text = format_str_v3(text, show_emo, show_event)

    return text

def audio2text(filename, language="auto"):
    try:
        start_time = time.time()
        
        # 使用 pydub 加载音频
        audio = AudioSegment.from_file(filename)
        y = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate

        # 转换为单声道
        if audio.channels > 1:
            y = y.reshape((-1, audio.channels)).mean(axis=1)

        audio_processing_time = time.time() - start_time

        # 语音识别
        input_wav = (sr, y)
        recognition_start_time = time.time()
        result = model_inference(input_wav=input_wav, language=language, show_emo=False)
        recognition_time = time.time() - recognition_start_time

        # 计算音频时长（秒）
        audio_duration = len(audio) / 1000.0

        logging.info(f"音频时长: {audio_duration:.2f}秒, 音频处理: {audio_processing_time:.2f}秒, 语音识别: {recognition_time:.2f}秒, 识别结果: {result}")
        return result
    except Exception as e:
        logging.error(f"无法加载音频文件 {filename}, err：{str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"无法加载音频文件：{str(e)}")

@app.post("/v1/audio/transcriptions")
async def transcriptions(file: Union[UploadFile, None] = File(default=None), language: str = Form(default="auto")):
    if file is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="未提供文件")

    # 生成带有时间信息的临时文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file_path = Path(TMP_DIR) / f"{timestamp}_{file.filename}"

    try:
        # 保存上传的文件到临时目录
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 调用 audio2text 进行文本识别
        result = audio2text(str(temp_file_path), language=language)
    finally:
        # 删除临时文件
        if temp_file_path.exists():
            temp_file_path.unlink()

    return {"text": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
