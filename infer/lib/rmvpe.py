from io import BytesIO
import os
from typing import List, Optional, Tuple, Union
import numpy as np
import torch

from infer.lib import jit

try:
    # Fix "Torch not compiled with CUDA enabled"
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init

        ipex_init()
except Exception:  # pylint: disable=broad-exception-caught
    pass
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

from rvc.f0.mel import MelSpectrogram
from rvc.f0.e2e import E2E


class RMVPE:
    def __init__(self, model_path: str, is_half, device=None, use_jit=False):
        self.resample_kernel = {}
        self.resample_kernel = {}
        self.is_half = is_half
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.mel_extractor = MelSpectrogram(
            is_half=is_half,
            n_mel_channels=128,
            sampling_rate=16000,
            win_length=1024,
            hop_length=160,
            mel_fmin=30,
            mel_fmax=8000,
            device=device,
        ).to(device)
        if "privateuseone" in str(device):
            import onnxruntime as ort

            ort_session = ort.InferenceSession(
                "%s/rmvpe.onnx" % os.environ["rmvpe_root"],
                providers=["DmlExecutionProvider"],
            )
            self.model = ort_session
        else:
            if str(self.device) == "cuda":
                self.device = torch.device("cuda:0")

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
                        device=device,
                        is_half=is_half,
                    )

                model = torch.jit.load(BytesIO(ckpt["model"]), map_location=device)
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
                    logger.warning(
                        "Use default rmvpe model. \
                                 Jit is not supported on the CPU for half floating point"
                    )
                    self.model = get_default_model()
                else:
                    self.model = get_jit_model()
            else:
                self.model = get_default_model()

            self.model = self.model.to(device)
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def mel2hidden(self, mel):
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

    def decode(self, hidden, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        # f0 = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        # torch.cuda.synchronize()
        # t0 = ttime()
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        mel = self.mel_extractor(
            audio.float().to(self.device).unsqueeze(0), center=True
        )
        # print(123123123,mel.device.type)
        # torch.cuda.synchronize()
        # t1 = ttime()
        hidden = self.mel2hidden(mel)
        # torch.cuda.synchronize()
        # t2 = ttime()
        # print(234234,hidden.device.type)
        if "privateuseone" not in str(self.device):
            hidden = hidden.squeeze(0).cpu().numpy()
        else:
            hidden = hidden[0]
        if self.is_half == True:
            hidden = hidden.astype("float32")

        f0 = self.decode(hidden, thred=thred)
        # torch.cuda.synchronize()
        # t3 = ttime()
        # print("hmvpe:%s\t%s\t%s\t%s"%(t1-t0,t2-t1,t3-t2,t3-t0))
        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        # t0 = ttime()
        center = np.argmax(salience, axis=1)  # 帧长#index
        salience = np.pad(salience, ((0, 0), (4, 4)))  # 帧长,368
        # t1 = ttime()
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        # t2 = ttime()
        todo_salience = np.array(todo_salience)  # 帧长，9
        todo_cents_mapping = np.array(todo_cents_mapping)  # 帧长，9
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)  # 帧长
        devided = product_sum / weight_sum  # 帧长
        # t3 = ttime()
        maxx = np.max(salience, axis=1)  # 帧长
        devided[maxx <= thred] = 0
        # t4 = ttime()
        # print("decode:%s\t%s\t%s\t%s" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        return devided
