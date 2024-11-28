import traceback
import logging
import os

logger = logging.getLogger(__name__)

import numpy as np
import torch
from io import BytesIO

from infer.lib.audio import load_audio, wav2, save_audio, float_np_array_to_wav_buf
from rvc.synthesizer import get_synthesizer, load_synthesizer
from .info import show_model_info
from .pipeline import Pipeline
from .utils import get_index_path_from_model, load_hubert


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if (
                self.hubert_model is not None
            ):  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info("Clean model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                ###楼下不这么折腾清理不干净
                self.net_g, self.cpt = get_synthesizer(self.cpt, self.config.device)
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            return (
                (
                    {"visible": False, "__type__": "update"},
                    to_return_protect0,
                    to_return_protect1,
                    {"value": to_return_protect[2], "__type__": "update"},
                    {"value": to_return_protect[3], "__type__": "update"},
                    {"value": "", "__type__": "update"},
                )
                if to_return_protect
                else {"visible": True, "maximum": 0, "__type__": "update"}
            )

        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

        self.net_g, self.cpt = load_synthesizer(person, self.config.device)
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()
        self.pipeline = Pipeline(self.tgt_sr, self.config)

        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info("Select index: " + index["value"])

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
                show_model_info(self.cpt),
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        if input_audio_path is None:
            return "You need to upload an audio", None
        elif hasattr(input_audio_path, "name"):
            input_audio_path = str(input_audio_path.name)
        f0_up_key = int(f0_up_key)
        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                np.divide(audio, audio_max, audio)
            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config.device, self.config.is_half)

            if file_index:
                if hasattr(file_index, "name"):
                    file_index = str(file_index.name)
                file_index = (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""  # 防止小白写错，自动帮他替换掉

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            ).astype(np.int16)
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = (
                "Index: %s." % file_index
                if os.path.exists(file_index)
                else "Index not used."
            )
            return (
                "Success.\n%s\nTime: npy: %.2fs, f0: %.2fs, infer: %.2fs."
                % (index_info, *times),
                (tgt_sr, audio_opt),
            )
        except Exception as e:
            info = traceback.format_exc()
            logger.warning(info)
            return str(e), None

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        try:
            dir_path = (
                dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    paths = [path.name for path in paths]
            except:
                traceback.print_exc()
                paths = [path.name for path in paths]
            infos = []
            for path in paths:
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    # file_big_npy,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        save_audio(
                            "%s/%s.%s" % (opt_root, os.path.basename(path), format1),
                            audio_opt,
                            tgt_sr,
                            f32=True,
                        )
                    except:
                        info += traceback.format_exc()
                infos.append("%s->%s" % (os.path.basename(path), info))
                yield "\n".join(infos)
            yield "\n".join(infos)
        except:
            yield traceback.format_exc()
