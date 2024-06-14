from typing import Any, Optional, Union

import numpy as np
import pyworld
from scipy import signal

from .f0 import F0Predictor


class Harvest(F0Predictor):
    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, sampling_rate=44100):
        super().__init__(hop_length, f0_min, f0_max, sampling_rate)

    def compute_f0(
        self,
        wav: np.ndarray,
        p_len: Optional[int] = None,
        filter_radius: Optional[Union[int, float]] = None,
    ):
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        f0, t = pyworld.harvest(
            wav.astype(np.double),
            fs=self.sampling_rate,
            f0_ceil=self.f0_max,
            f0_floor=self.f0_min,
            frame_period=1000 * self.hop_length / self.sampling_rate,
        )
        f0 = pyworld.stonemask(wav.astype(np.double), f0, t, self.sampling_rate)
        if filter_radius is not None and filter_radius > 2:
            f0 = signal.medfilt(f0, 3)
        return self._interpolate_f0(self._resize_f0(f0, p_len))[0]
