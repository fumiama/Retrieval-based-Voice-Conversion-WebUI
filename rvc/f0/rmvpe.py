from io import BytesIO
import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from rvc.jit import load_inputs, get_jit_model, export_jit_model, save_pickle

from .mel import MelSpectrogram
from .f0 import F0Predictor
from .models import get_rmvpe


def rmvpe_jit_export(
    model_path: str,
    mode: str = "script",
    inputs_path: str = None,
    save_path: str = None,
    device=torch.device("cpu"),
    is_half=False,
):
    if not save_path:
        save_path = model_path.rstrip(".pth")
        save_path += ".half.jit" if is_half else ".jit"
    if "cuda" in str(device) and ":" not in str(device):
        device = torch.device("cuda:0")

    model = get_rmvpe(model_path, device, is_half)
    inputs = None
    if mode == "trace":
        inputs = load_inputs(inputs_path, device, is_half)
    ckpt = export_jit_model(model, mode, inputs, device, is_half)
    ckpt["device"] = str(device)
    save_pickle(ckpt, save_path)
    return ckpt


class RMVPE(F0Predictor):
    def __init__(
        self,
        model_path: str,
        is_half: bool,
        device: str,
        use_jit=False,
    ):
        hop_length = 160
        f0_min = 30
        f0_max = 8000
        sampling_rate = 16000

        super().__init__(
            hop_length,
            f0_min,
            f0_max,
            sampling_rate,
            device,
        )

        self.is_half = is_half
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

        self.mel_extractor = MelSpectrogram(
            is_half=is_half,
            n_mel_channels=128,
            sampling_rate=sampling_rate,
            win_length=1024,
            hop_length=hop_length,
            mel_fmin=f0_min,
            mel_fmax=f0_max,
            device=self.device,
        ).to(self.device)

        if "privateuseone" in str(self.device):
            import onnxruntime as ort

            self.model = ort.InferenceSession(
                "%s/rmvpe.onnx" % os.environ["rmvpe_root"],
                providers=["DmlExecutionProvider"],
            )
        else:

            def rmvpe_jit_model():
                ckpt = get_jit_model(model_path, is_half, self.device, rmvpe_jit_export)
                model = torch.jit.load(BytesIO(ckpt["model"]), map_location=self.device)
                model = model.to(self.device)
                return model

            if use_jit and not (is_half and "cpu" in str(self.device)):
                self.model = rmvpe_jit_model()
            else:
                self.model = get_rmvpe(model_path, self.device, is_half)

    def compute_f0(
        self,
        wav: np.ndarray,
        p_len: Optional[int] = None,
        filter_radius: Optional[Union[int, float]] = None,
    ):
        if p_len is None:
            p_len = wav.shape[0] // self.hop_length
        if not torch.is_tensor(wav):
            wav = torch.from_numpy(wav)
        mel = self.mel_extractor(wav.float().to(self.device).unsqueeze(0), center=True)
        hidden = self._mel2hidden(mel)
        if "privateuseone" not in str(self.device):
            hidden = hidden.squeeze(0).cpu().numpy()
        else:
            hidden = hidden[0]
        if self.is_half == True:
            hidden = hidden.astype("float32")

        f0 = self._decode(hidden, thred=filter_radius)

        return self._interpolate_f0(self._resize_f0(f0, p_len))[0]

    def _to_local_average_cents(self, salience, threshold=0.05):
        center = np.argmax(salience, axis=1)  # 帧长#index
        salience = np.pad(salience, ((0, 0), (4, 4)))  # 帧长,368
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        todo_salience = np.array(todo_salience)  # 帧长，9
        todo_cents_mapping = np.array(todo_cents_mapping)  # 帧长，9
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)  # 帧长
        devided = product_sum / weight_sum  # 帧长
        maxx = np.max(salience, axis=1)  # 帧长
        devided[maxx <= threshold] = 0
        return devided

    def _mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            if "privateuseone" in str(self.device):
                onnx_input_name = self.model.get_inputs()[0].name
                onnx_outputs_names = self.model.get_outputs()[0].name
                hidden = self.model.run(
                    [onnx_outputs_names],
                    input_feed={onnx_input_name: mel.cpu().numpy()},
                )[0]
            else:
                mel = mel.half() if self.is_half else mel.float()
                hidden = self.model(mel)
            return hidden[:, :n_frames]

    def _decode(self, hidden, thred=0.03):
        if thred is None:
            thred = 0.03
        cents_pred = self._to_local_average_cents(hidden, threshold=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        # f0 = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
        return f0
