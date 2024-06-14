from io import BytesIO
import os
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from infer.lib import jit

from .mel import MelSpectrogram
from .e2e import E2E
from .f0 import F0Predictor


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

            def get_jit_model():
                jit_model_path = model_path.rstrip(".pth")
                jit_model_path += ".half.jit" if is_half else ".jit"
                ckpt = None
                if os.path.exists(jit_model_path):
                    ckpt = jit.load(jit_model_path)
                    model_device = ckpt["device"]
                    if model_device != str(self.device):
                        del ckpt
                        ckpt = None

                if ckpt is None:
                    ckpt = jit.rmvpe_jit_export(
                        model_path=model_path,
                        mode="script",
                        inputs_path=None,
                        save_path=jit_model_path,
                        device=self.device,
                        is_half=is_half,
                    )

                model = torch.jit.load(BytesIO(ckpt["model"]), map_location=self.device)
                return model

            def get_default_model():
                model = E2E(4, 1, (2, 2))
                ckpt = torch.load(model_path, map_location="cpu")
                model.load_state_dict(ckpt)
                model.eval()
                if is_half:
                    model = model.half()
                else:
                    model = model.float()
                return model

            if use_jit:
                if is_half and "cpu" in str(self.device):
                    self.model = get_default_model()
                else:
                    self.model = get_jit_model()
            else:
                self.model = get_default_model()

            self.model = self.model.to(self.device)

    def compute_f0(
        self,
        wav: np.ndarray[Any, np.dtype],
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
        cents_pred = self._to_local_average_cents(hidden, threshold=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        # f0 = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
        return f0
