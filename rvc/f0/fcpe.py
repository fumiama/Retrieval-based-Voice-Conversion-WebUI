from typing import Any, Optional, Union

import numpy as np
import torch
from torchfcpe import spawn_bundled_infer_model

from .f0 import F0Predictor


class FCPE(F0Predictor):
    def __init__(
        self,
        hop_length=512,
        f0_min=50,
        f0_max=1100,
        sampling_rate=44100,
        device="cpu",
    ):
        super().__init__(
            hop_length,
            f0_min,
            f0_max,
            sampling_rate,
            device,
        )

        self.model = spawn_bundled_infer_model(self.device)

    def compute_f0(
        self,
        wav: np.ndarray,
        p_len: Optional[int] = None,
        filter_radius: Optional[Union[int, float]] = 0.006,
    ):
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        if not torch.is_tensor(wav):
            wav = torch.from_numpy(wav)
        f0 = (
            self.model.infer(
                wav.float().to(self.device).unsqueeze(0),
                sr=self.sampling_rate,
                decoder_mode="local_argmax",
                threshold=filter_radius,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        return self._interpolate_f0(self._resize_f0(f0, p_len))[0]
