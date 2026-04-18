from io import BufferedWriter, BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import os
import math
import wave
import signal
from multiprocessing import Process, Value, Event
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numba import jit
import av
from av.audio.resampler import AudioResampler
from av.audio.frame import AudioFrame
import scipy.io.wavfile as wavfile

video_format_dict: Dict[str, str] = {
    "m4a": "mp4",
}

audio_format_dict: Dict[str, str] = {
    "ogg": "libvorbis",
    "mp4": "aac",
}


@jit(nopython=True)
def float_to_int16(audio: np.ndarray) -> np.ndarray:
    am = int(math.ceil(float(np.abs(audio).max())) * 32768)
    am = 32767 * 32768 // am
    return np.multiply(audio, am).astype(np.int16)


def float_np_array_to_wav_buf(wav: np.ndarray, sr: int, f32=False) -> BytesIO:
    buf = BytesIO()
    if f32:
        wavfile.write(buf, sr, wav.astype(np.float32))
    else:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2 if len(wav.shape) > 1 else 1)
            wf.setsampwidth(2)  # Sample width in bytes
            wf.setframerate(sr)  # Sample rate in Hz
            wf.writeframes(float_to_int16(wav.T if len(wav.shape) > 1 else wav))
    buf.seek(0, 0)
    return buf


def save_audio(path: str, audio: np.ndarray, sr: int, f32=False, format="wav"):
    buf = float_np_array_to_wav_buf(audio, sr, f32)
    if format != "wav":
        transbuf = BytesIO()
        wav2(buf, transbuf, format)
        buf = transbuf
    with open(path, "wb") as f:
        f.write(buf.getbuffer())


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


def load_audio(
    file: Union[str, BytesIO, Path],
    sr: Optional[int] = None,
    format: Optional[str] = None,
    mono=True,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    if (isinstance(file, str) and not Path(file).exists()) or (
        isinstance(file, Path) and not file.exists()
    ):
        raise FileNotFoundError(f"File not found: {file}")
    rate = 0

    container = av.open(file, format=format)
    audio_stream = next(s for s in container.streams if s.type == "audio")
    channels = 1 if audio_stream.layout == "mono" else 2
    container.seek(0)
    resampler = (
        AudioResampler(format="fltp", layout=audio_stream.layout, rate=sr)
        if sr is not None
        else None
    )

    # Estimated maximum total number of samples to pre-allocate the array
    # AV stores length in microseconds by default
    estimated_total_samples = (
        int(container.duration * sr // 1_000_000) if sr is not None else 48000
    )
    decoded_audio = np.zeros(
        (
            estimated_total_samples + 1
            if channels == 1
            else (channels, estimated_total_samples + 1)
        ),
        dtype=np.float32,
    )

    offset = 0

    def process_packet(packet: List[AudioFrame]):
        frames_data = []
        rate = 0
        for frame in packet:
            # frame.pts = None  # 清除时间戳，避免重新采样问题
            resampled_frames = (
                resampler.resample(frame) if resampler is not None else [frame]
            )
            for resampled_frame in resampled_frames:
                frame_data = resampled_frame.to_ndarray()
                rate = resampled_frame.rate
                frames_data.append(frame_data)
        return (rate, frames_data)

    def frame_iter(container):
        for p in container.demux(container.streams.audio[0]):
            yield p.decode()

    for r, frames_data in map(process_packet, frame_iter(container)):
        if not rate:
            rate = r
        for frame_data in frames_data:
            end_index = offset + len(frame_data[0])

            # 检查 decoded_audio 是否有足够的空间，并在必要时调整大小
            if end_index > decoded_audio.shape[1]:
                decoded_audio = np.resize(
                    decoded_audio, (decoded_audio.shape[0], end_index * 4)
                )

            np.copyto(decoded_audio[..., offset:end_index], frame_data)
            offset += len(frame_data[0])

    container.close()

    # Truncate the array to the actual size
    decoded_audio = decoded_audio[..., :offset]

    if mono and decoded_audio.shape[0] > 1:
        decoded_audio = decoded_audio.mean(0)

    if sr is not None:
        return decoded_audio
    return decoded_audio, rate


def resample_audio(
    input_path: str, output_path: str, codec: str, format: str, sr: int, layout: str
) -> None:
    if not os.path.exists(input_path):
        return

    input_container = av.open(input_path)
    output_container = av.open(output_path, "w")

    # Create a stream in the output container
    input_stream = input_container.streams.audio[0]
    output_stream = output_container.add_stream(codec, rate=sr, layout=layout)

    resampler = AudioResampler(format, layout, sr)

    # Copy packets from the input file to the output file
    for packet in input_container.demux(input_stream):
        for frame in packet.decode():
            # frame.pts = None  # Clear presentation timestamp to avoid resampling issues
            out_frames = resampler.resample(frame)
            for out_frame in out_frames:
                for out_packet in output_stream.encode(out_frame):
                    output_container.mux(out_packet)

    for packet in output_stream.encode():
        output_container.mux(packet)

    # Close the containers
    input_container.close()
    output_container.close()


def get_audio_properties(input_path: str) -> Tuple[int, int]:
    container = av.open(input_path)
    audio_stream = next(s for s in container.streams if s.type == "audio")
    channels = 1 if audio_stream.layout == "mono" else 2
    try:
        rate = audio_stream.base_rate
    except:
        rate = audio_stream.sample_rate
    container.close()
    return channels, rate


class AudioIoProcess(Process):
    def __init__(
        self,
        input_device,
        output_device,
        input_audio_block_size: int,
        sample_rate: int,
        channel_num: int = 2,
        is_device_combined: bool = True,
        is_input_wasapi_exclusive: bool = False,
        is_output_wasapi_exclusive: bool = False,
    ):
        super().__init__()
        self.in_dev = input_device
        self.out_dev = output_device
        self.block_size: int = input_audio_block_size
        self.buf_size: int = self.block_size << 1  # 双缓冲
        self.sample_rate: int = sample_rate
        self.channels: int = channel_num
        self.is_device_combined: bool = is_device_combined
        self.is_input_wasapi_exclusive: bool = is_input_wasapi_exclusive
        self.is_output_wasapi_exclusive: bool = is_output_wasapi_exclusive

        self.__rec_ptr = 0
        self.in_ptr = Value("i", 0)  # 当收满一个block时由本进程设置
        self.out_ptr = Value("i", 0)  # 由主进程设置，指示下一次预期写入位置
        self.play_ptr = Value("i", 0)  # 由本进程设置，指示当前音频已经播放到哪里
        self.in_evt = Event()  # 当收满一个block时由本进程设置
        self.stop_evt = Event()  # 当主进程停止音频活动时由主进程设置

        self.latency = Value("d", 114514.1919810)

        self.buf_shape: tuple = (self.buf_size, self.channels)
        self.buf_dtype: np.dtype = np.float32
        self.buf_nbytes: int = int(
            np.prod(self.buf_shape) * np.dtype(self.buf_dtype).itemsize
        )

        self.in_mem = SharedMemory(create=True, size=self.buf_nbytes)
        self.out_mem = SharedMemory(create=True, size=self.buf_nbytes)
        self.in_mem_name: str = self.in_mem.name
        self.out_mem_name: str = self.out_mem.name

        self.in_buf = None
        self.out_buf = None

    def get_in_mem_name(self) -> str:
        return self.in_mem_name

    def get_out_mem_name(self) -> str:
        return self.out_mem_name

    def get_np_shape(self) -> tuple:
        return self.buf_shape

    def get_np_dtype(self) -> np.dtype:
        return self.buf_dtype

    def get_ptrs_and_events(self):
        return self.in_ptr, self.out_ptr, self.play_ptr, self.in_evt, self.stop_evt

    def get_latency(self) -> float:
        return self.latency.value

    def run(self):
        import sounddevice as sd

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        in_mem = SharedMemory(name=self.in_mem_name)
        self.in_buf = np.ndarray(
            self.buf_shape, dtype=self.buf_dtype, buffer=in_mem.buf, order="C"
        )
        self.in_buf.fill(0.0)

        out_mem = SharedMemory(name=self.out_mem_name)
        self.out_buf = np.ndarray(
            self.buf_shape, dtype=self.buf_dtype, buffer=out_mem.buf, order="C"
        )
        self.out_buf.fill(0.0)

        exclusive_settings = sd.WasapiSettings(exclusive=True)

        sd.default.device = (self.in_dev, self.out_dev)

        def output_callback(outdata, frames, time_info, status):
            play_ptr = self.play_ptr.value
            end_ptr = play_ptr + frames

            if end_ptr <= self.buf_size:
                outdata[:] = self.out_buf[play_ptr:end_ptr]
            else:
                first = self.buf_size - play_ptr
                second = end_ptr - self.buf_size
                outdata[:first] = self.out_buf[play_ptr:]
                outdata[first:] = self.out_buf[:second]

            self.play_ptr.value = end_ptr % self.buf_size

        def input_callback(indata, frames, time_info, status):
            # 收录输入数据
            end_ptr = self.__rec_ptr + frames
            if end_ptr <= self.buf_size:  # 整块拷贝
                self.in_buf[self.__rec_ptr : end_ptr] = indata
            else:  # 处理回绕
                first = self.buf_size - self.__rec_ptr
                second = end_ptr - self.buf_size
                self.in_buf[self.__rec_ptr :] = indata[:first]
                self.in_buf[:second] = indata[first:]
            write_pos = self.__rec_ptr
            self.__rec_ptr = end_ptr % self.buf_size

            # 设置信号
            if write_pos < self.block_size and self.__rec_ptr >= self.block_size:
                self.in_ptr.value = 0
                self.in_evt.set()  # 通知主线程来取甲缓冲
            elif write_pos < self.buf_size and self.__rec_ptr < write_pos:
                self.in_ptr.value = self.block_size
                self.in_evt.set()  # 通知主线程来取乙缓冲

        def combined_callback(indata, outdata, frames, time_info, status):
            output_callback(outdata, frames, time_info, status)  # 优先出声
            input_callback(indata, frames, time_info, status)

        if self.is_device_combined:
            with sd.Stream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.buf_dtype,
                latency="low",
                extra_settings=(
                    exclusive_settings
                    if self.is_input_wasapi_exclusive
                    and self.is_output_wasapi_exclusive
                    else None
                ),
                callback=combined_callback,
            ) as s:
                self.latency.value = s.latency[-1]
                self.stop_evt.wait()
                self.out_buf.fill(0.0)
        else:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.buf_dtype,
                latency="low",
                extra_settings=(
                    exclusive_settings if self.is_input_wasapi_exclusive else None
                ),
                callback=input_callback,
            ) as si, sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.buf_dtype,
                latency="low",
                extra_settings=(
                    exclusive_settings if self.is_output_wasapi_exclusive else None
                ),
                callback=output_callback,
            ) as so:
                self.latency.value = si.latency[-1] + so.latency[-1]
                self.stop_evt.wait()
                self.out_buf.fill(0.0)

        # 清理共享内存
        in_mem.close()
        out_mem.close()
        in_mem.unlink()
        out_mem.unlink()
