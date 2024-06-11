from io import BufferedWriter, BytesIO
from pathlib import Path
from typing import Dict
import numpy as np
import av
from av.audio.resampler import AudioResampler

video_format_dict: Dict[str, str] = {
    "m4a": "mp4",
}

audio_format_dict: Dict[str, str] = {
    "ogg": "libvorbis",
    "mp4": "aac",
}

def wav2(i: BytesIO, o: BufferedWriter, format: str):
    inp = av.open(i, "r")
    format = video_format_dict.get(format, format)
    out = av.open(o, "w", format=format)
    format = audio_format_dict.get(format, format)

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
        container = av.open(file)
        resampler = AudioResampler(format='fltp', layout='mono', rate=sr)
        decoded_audio = []

        for frame in container.decode(audio=0):
            frame.pts = None  # Clear presentation timestamp to avoid resampling issues
            resampled = resampler.resample(frame)
            decoded_audio.append(resampled.to_ndarray())

        audio = np.concatenate(decoded_audio)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return audio.flatten()


def clean_path(path: str) -> Path:
    return Path(path.strip(' "\n')).resolve()
