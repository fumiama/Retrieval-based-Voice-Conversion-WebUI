import os
import traceback
import logging

logger = logging.getLogger(__name__)

import av
from av.audio.resampler import AudioResampler
import torch

from configs import Config
from infer.modules.uvr5.mdxnet import MDXNetDereverb
from infer.modules.uvr5.vr import AudioPre, AudioPreDeEcho

config = Config()


def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    infos = []
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15, config.device)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(
                    os.getenv("weight_uvr5_root"), model_name + ".pth"
                ),
                device=config.device,
                is_half=config.is_half,
            )
        is_hp3 = "HP3" in model_name
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            need_reformat = 1
            done = 0
            try:
                container = av.open(inp_path)
                audio_stream = next(s for s in container.streams if s.type == 'audio')

                # Check the audio stream's properties
                if audio_stream.channels == 2 and audio_stream.rate == 44100:
                    pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0, is_hp3=is_hp3)
                    need_reformat = 0
                    done = 1
            except Exception as e:
                need_reformat = 1
                print(f"Exception {e} occured. Will reformat")
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                process_audio(inp_path, tmp_path)
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                try:
                    if done == 0:
                        pre_fun._path_audio_(
                            inp_path, save_root_ins, save_root_vocal, format0
                        )
                    infos.append("%s->Success" % (os.path.basename(inp_path)))
                    yield "\n".join(infos)
                except:
                    infos.append(
                        "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                    )
                    yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Executed torch.cuda.empty_cache()")
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.info("Executed torch.mps.empty_cache()")
    yield "\n".join(infos)

def process_audio(input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path): return
    
    input_container = av.open(input_path)
    output_container = av.open(output_path, 'w')

    # Create a stream in the output container
    input_stream = input_container.streams.audio[0]
    output_stream = output_container.add_stream('pcm_s16le', rate=44100, layout='stereo')
    
    resampler = AudioResampler('pcm_s16le', 'stereo', 44100)

    output_stream.bit_rate = 128_000 # 128kb/s (equivalent to -q:a 2)

    # Copy packets from the input file to the output file
    for packet in input_container.demux(input_stream):
        for frame in packet.decode():
            frame.pts = None  # Clear presentation timestamp to avoid resampling issues
            resampled = resampler.resample(frame)
            for out_packet in output_stream.encode(resampled):
                output_container.mux(out_packet)

    for packet in output_stream.encode():
        output_container.mux(packet)
    
    # Close the containers
    input_container.close()
    output_container.close()

    try: # Remove the original file
        os.remove(input_path)
    except Exception as e:
        print(f"Failed to remove the original file: {e}")