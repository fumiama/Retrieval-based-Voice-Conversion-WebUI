from typing import Any, Optional

import numpy as np
import pyworld

from .f0 import F0Predictor


class HarvestF0Predictor(F0Predictor):
    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, sampling_rate=44100):
        super().__init__(hop_length, f0_min, f0_max, sampling_rate)

    def compute_f0(self, wav: np.ndarray[Any, np.dtype], p_len: Optional[int] = None):
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        f0, t = pyworld.harvest(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_ceil=self.f0_max,
            f0_floor=self.f0_min,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.fs)
        return self.interpolate_f0(self.resize_f0(f0, p_len))[0]

    def compute_f0_uv(
        self, wav: np.ndarray[Any, np.dtype], p_len: Optional[int] = None
    ):
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        f0, t = pyworld.harvest(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        return self.interpolate_f0(self.resize_f0(f0, p_len))
