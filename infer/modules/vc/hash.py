import numpy as np
import torch
import hashlib
import pathlib

from functools import lru_cache
from scipy.fft import fft
from pybase16384 import encode_to_string, decode_from_string

from configs import CPUConfig
from rvc.synthesizer import get_synthesizer

from .pipeline import Pipeline
from .utils import load_hubert


class TorchSeedContext:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)

    def __exit__(self, type, value, traceback):
        torch.random.set_rng_state(self.state)


half_hash_len = 512
expand_factor = 65536 * 8


@lru_cache(None)  # None 表示无限缓存
def original_audio_storage():
    return np.load(pathlib.Path(__file__).parent / "lgdsng.npz")


@lru_cache(None)
def original_audio():
    return original_audio_storage()["a"]


@lru_cache(None)
def original_audio_time_minus():
    return original_audio_storage()["t"]


@lru_cache(None)
def original_audio_freq_minus():
    return original_audio_storage()["f"]


@lru_cache(None)
def original_rmvpe_f0():
    x = original_audio_storage()
    return x["pitch"], x["pitchf"]


def _cut_u16(n):
    if n > 16384:
        n = 16384 + 16384 * (1 - np.exp((16384 - n) / expand_factor))
    elif n < -16384:
        n = -16384 - 16384 * (1 - np.exp((n + 16384) / expand_factor))
    return n


# wave_hash will change time_field, use carefully
def wave_hash(time_field):
    np.divide(time_field, np.abs(time_field).max(), time_field)
    if len(time_field) != 48000:
        raise Exception("time not hashable")
    freq_field = fft(time_field)
    if len(freq_field) != 48000:
        raise Exception("freq not hashable")
    np.add(time_field, original_audio_time_minus(), out=time_field)
    np.add(freq_field, original_audio_freq_minus(), out=freq_field)
    hash = np.zeros(half_hash_len // 2 * 2, dtype=">i2")
    d = 375 * 512 // half_hash_len
    for i in range(half_hash_len // 4):
        a = i * 2
        b = a + 1
        x = a + half_hash_len // 2
        y = x + 1
        s = np.average(freq_field[i * d : (i + 1) * d])
        hash[a] = np.int16(_cut_u16(round(32768 * np.real(s))))
        hash[b] = np.int16(_cut_u16(round(32768 * np.imag(s))))
        hash[x] = np.int16(
            _cut_u16(round(32768 * np.sum(time_field[i * d : i * d + d // 2])))
        )
        hash[y] = np.int16(
            _cut_u16(round(32768 * np.sum(time_field[i * d + d // 2 : (i + 1) * d])))
        )
    return encode_to_string(hash.tobytes())


def model_hash(config, tgt_sr, net_g, if_f0, version):
    pipeline = Pipeline(tgt_sr, config)
    audio = original_audio()
    hbt = load_hubert(config.device, config.is_half)
    audio_opt = pipeline.pipeline(
        hbt,
        net_g,
        0,
        audio,
        [0, 0, 0],
        6,
        original_rmvpe_f0(),
        "",
        0,
        2 if if_f0 else 0,
        3,
        tgt_sr,
        16000,
        0.25,
        version,
        0.33,
    )
    del hbt
    opt_len = len(audio_opt)
    diff = 48000 - opt_len
    if diff > 0:
        audio_opt = np.pad(audio_opt, (diff, 0))
    elif diff < 0:
        n = diff // 2
        n = -n
        audio_opt = audio_opt[n:-n]
    h = wave_hash(audio_opt)
    del pipeline, audio_opt
    return h


def model_hash_ckpt(cpt):
    config = CPUConfig()

    with TorchSeedContext(114514):
        net_g, cpt = get_synthesizer(cpt, config.device)
        tgt_sr = cpt["config"][-1]
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")

        if config.is_half:
            net_g = net_g.half()
        else:
            net_g = net_g.float()

        h = model_hash(config, tgt_sr, net_g, if_f0, version)

        del net_g

    return h


def model_hash_from(path):
    cpt = torch.load(path, map_location="cpu")
    h = model_hash_ckpt(cpt)
    del cpt
    return h


def _extend_difference(n, a, b):
    if n < a:
        n = a
    elif n > b:
        n = b
    n -= a
    n /= b - a
    return n


def hash_similarity(h1: str, h2: str) -> float:
    try:
        h1b, h2b = decode_from_string(h1), decode_from_string(h2)
        if len(h1b) != half_hash_len * 2 or len(h2b) != half_hash_len * 2:
            raise Exception("invalid hash length")
        h1n, h2n = np.frombuffer(h1b, dtype=">i2"), np.frombuffer(h2b, dtype=">i2")
        d = 0
        for i in range(half_hash_len // 4):
            a = i * 2
            b = a + 1
            ax = complex(h1n[a], h1n[b])
            bx = complex(h2n[a], h2n[b])
            if abs(ax) == 0 or abs(bx) == 0:
                continue
            d += np.abs(ax - bx)
        frac = np.linalg.norm(h1n) * np.linalg.norm(h2n)
        cosine = (
            np.dot(h1n.astype(np.float32), h2n.astype(np.float32)) / frac
            if frac != 0
            else 1.0
        )
        distance = _extend_difference(np.exp(-d / expand_factor), 0.5, 1.0)
        return round((abs(cosine) + distance) / 2, 6)
    except Exception as e:
        return str(e)


def hash_id(h: str) -> str:
    d = decode_from_string(h)
    if len(d) != half_hash_len * 2:
        return "invalid hash length"
    return encode_to_string(
        np.frombuffer(d, dtype=np.uint64).sum(keepdims=True).tobytes()
    )[:-2] + encode_to_string(hashlib.md5(d).digest()[:7])
