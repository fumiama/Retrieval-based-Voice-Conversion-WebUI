import librosa
import numpy as np
import onnxruntime

from onnx.f0predictor import PMF0Predictor
from onnx.f0predictor import HarvestF0Predictor
from onnx.f0predictor import DioF0Predictor


class ContentVec:
    def __init__(self, vec_path: str, device=None):
        if device == "cpu" or device is None:
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "dml":
            providers = ["DmlExecutionProvider"]
        else:
            raise RuntimeError("Unsportted Device")
        self.model = onnxruntime.InferenceSession(vec_path, providers=providers)

    def __call__(self, wav):
        return self.forward(wav)

    def forward(self, wav):
        if wav.ndim == 2:  # double channels
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        wav = np.expand_dims(np.expand_dims(wav, 0), 0)
        onnx_input = {self.model.get_inputs()[0].name: wav}
        logits = self.model.run(None, onnx_input)[0]
        return logits.transpose(0, 2, 1)


predicters = {
    "pm": PMF0Predictor,
    "harvest": HarvestF0Predictor,
    "dio": DioF0Predictor,
}


def get_f0_predictor(f0_method, hop_length, sampling_rate):
    return predicters[f0_method](hop_length=hop_length, sampling_rate=sampling_rate)


class RVC:
    def __init__(
        self,
        model_path,
        sr=40000,
        hop_size=512,
        vec_path="vec-768-layer-12.onnx",
        device="cpu",
    ):
        self.vec_model = ContentVec(vec_path, device)
        if device == "cpu" or device is None:
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "dml":
            providers = ["DmlExecutionProvider"]
        else:
            raise RuntimeError("Unsportted Device")
        self.model = onnxruntime.InferenceSession(model_path, providers=providers)
        self.sampling_rate = sr
        self.hop_size = hop_size

    def forward(self, hubert, hubert_length, pitch, pitchf, ds, rnd):
        onnx_input = {
            self.model.get_inputs()[0].name: hubert,
            self.model.get_inputs()[1].name: hubert_length,
            self.model.get_inputs()[2].name: pitch,
            self.model.get_inputs()[3].name: pitchf,
            self.model.get_inputs()[4].name: ds,
            self.model.get_inputs()[5].name: rnd,
        }
        return (self.model.run(None, onnx_input)[0] * 32767).astype(np.int16)

    def inference(
        self,
        wav,
        sr,
        sid,
        f0_method="dio",
        f0_up_key=0,
    ):
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0_predictor = get_f0_predictor(
            f0_method,
            self.hop_size,
            self.sampling_rate,
        )
        org_length = len(wav)
        if org_length / sr > 50.0:
            raise RuntimeError("Reached Max Length")

        wav16k = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        wav16k = wav16k

        hubert = self.vec_model(wav16k)
        hubert = np.repeat(hubert, 2, axis=2).transpose(0, 2, 1).astype(np.float32)
        hubert_length = hubert.shape[1]

        pitchf = f0_predictor.compute_f0(wav, hubert_length)
        pitchf = pitchf * 2 ** (f0_up_key / 12)
        pitch = pitchf.copy()
        f0_mel = 1127 * np.log(1 + pitch / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        pitch = np.rint(f0_mel).astype(np.int64)

        pitchf = pitchf.reshape(1, len(pitchf)).astype(np.float32)
        pitch = pitch.reshape(1, len(pitch))
        ds = np.array([sid]).astype(np.int64)

        rnd = np.random.randn(1, 192, hubert_length).astype(np.float32)
        hubert_length = np.array([hubert_length]).astype(np.int64)

        out_wav = self.forward(hubert, hubert_length, pitch, pitchf, ds, rnd).squeeze()
        out_wav = np.pad(out_wav, (0, 2 * self.hop_size), "constant")
        return out_wav[0:org_length]
