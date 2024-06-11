from io import BufferedWriter, BytesIO
from pathlib import Path
from typing import Any, Dict
import ffmpeg
import numpy as np
import av

video_format_dict: Dict[str, str] = {
    "m4a": "mp4",
}

audio_format_dict: Dict[str, str] = {
    "ogg": "libvorbis",
    "mp4": "aac",
}

def wav2(i: BytesIO, o: BufferedWriter, format: str):
    inp = av.open(i, "r")
    format = video_format_dict.get(format, "mp4")
    out = av.open(o, "w", format=format)
    format = audio_format_dict.get(format, "aac")

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def load_audio(file: str, sr: int) -> np.ndarray:
    if not Path(file).exists():
        raise FileNotFoundError(f"File not found: {file}")

    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = str(clean_path(file))  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path: str) -> Path:
    return Path(path.strip(' "\n')).resolve()
