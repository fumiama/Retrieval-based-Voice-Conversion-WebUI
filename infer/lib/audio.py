from io import BufferedWriter, BytesIO
from pathlib import Path
from typing import Dict
import numpy as np
import av
import os
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

def downsample_audio(input_path: str, output_path: str, format: str) -> None:
    if not os.path.exists(input_path): return
    
    input_container = av.open(input_path)
    output_container = av.open(output_path, 'w')

    # Create a stream in the output container
    input_stream = input_container.streams.audio[0]
    output_stream = output_container.add_stream(format)

    output_stream.bit_rate = 128_000 # 128kb/s (equivalent to -q:a 2)

    # Copy packets from the input file to the output file
    for packet in input_container.demux(input_stream):
        for frame in packet.decode():
            for out_packet in output_stream.encode(frame):
                output_container.mux(out_packet)

    for packet in output_stream.encode():
        output_container.mux(packet)
    
    # Close the containers
    input_container.close()
    output_container.close()

    try: # Remove the original file
        os.remove(input_path)
    except Exception as e:
        print(f"Failed to remove the original file: {e}")

def resample_audio(input_path: str, output_path: str, format: str, sr: int, layout: str) -> None:
    if not os.path.exists(input_path): return
    
    input_container = av.open(input_path)
    output_container = av.open(output_path, 'w')

    # Create a stream in the output container
    input_stream = input_container.streams.audio[0]
    output_stream = output_container.add_stream(format, rate=sr, layout=layout)
    
    resampler = AudioResampler(format, layout, sr)

    # Copy packets from the input file to the output file
    for packet in input_container.demux(input_stream):
        for frame in packet.decode():
            frame.pts = None  # Clear presentation timestamp to avoid resampling issues
            resampled = resampler.resample(frame)
            for out_packet in output_stream.encode(resampled):
                output_container.mux(out_packet)

    for packet in output_stream.encode():
        output_container.mux(packet)
    
    # Close the containers
    input_container.close()
    output_container.close()

    try: # Remove the original file
        os.remove(input_path)
    except Exception as e:
        print(f"Failed to remove the original file: {e}")

def clean_path(path: str) -> Path:
    return Path(path.strip(' "\n')).resolve()
