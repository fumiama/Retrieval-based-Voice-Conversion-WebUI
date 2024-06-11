import soundfile
import librosa

from rvc.onnx import RVC

hop_size = 512
sampling_rate = 40000  # 采样率
f0_up_key = 0  # 升降调
sid = 0  # 角色ID
f0_method = "dio"  # F0提取算法
model_path = "exported_model.onnx"  # 模型的完整路径
vec_path = "vec-256-layer-9.onnx"  # 需要onnx的vec模型
wav_path = "123.wav"  # 输入路径或ByteIO实例
out_path = "out.wav"  # 输出路径或ByteIO实例

model = RVC(model_path, vec_path=vec_path, hop_len=hop_size, device="cuda")

wav, sr = librosa.load(wav_path, sr=sampling_rate)

audio = model.infer(wav, sr, sampling_rate, sid, f0_method, f0_up_key)

soundfile.write(out_path, audio, sampling_rate)
