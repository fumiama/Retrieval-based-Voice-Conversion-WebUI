import logging
import os

import gradio as gr
from dotenv import load_dotenv

from configs import Config
from i18n.i18n import I18nAuto
from infer.modules.vc import VC

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

i18n = I18nAuto()
logger.info(i18n)

load_dotenv()
config = Config()
vc = VC(config)

weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
names = []
hubert_model = None
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("在线demo"):
            gr.Markdown(
                value="""
                RVC 在线demo
                """
            )
            sid = gr.Dropdown(label=i18n("Inferencing voice"), choices=sorted(names))
            with gr.Column():
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("Select Speaker/Singer ID"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
            sid.change(fn=vc.get_vc, inputs=[sid], outputs=[spk_item])
            gr.Markdown(
                value=i18n(
                    "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)"
                )
            )
            vc_input3 = gr.Audio(label="上传音频（长度小于90秒）")
            vc_transform0 = gr.Number(
                label=i18n(
                    "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)"
                ),
                value=0,
            )
            f0method0 = gr.Radio(
                label=i18n(
                    "Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': best quality, and little GPU requirement"
                ),
                choices=["pm", "harvest", "crepe", "rmvpe"],
                value="pm",
                interactive=True,
            )
            filter_radius0 = gr.Slider(
                minimum=0,
                maximum=7,
                label=i18n(
                    "If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."
                ),
                value=3,
                step=1,
                interactive=True,
            )
            with gr.Column():
                file_index1 = gr.Textbox(
                    label=i18n(
                        "Path to the feature index file. Leave blank to use the selected result from the dropdown"
                    ),
                    value="",
                    interactive=False,
                    visible=False,
                )
            file_index2 = gr.Dropdown(
                label=i18n("Auto-detect index path and select from the dropdown"),
                choices=sorted(index_paths),
                interactive=True,
            )
            index_rate1 = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Feature searching ratio"),
                value=0.88,
                interactive=True,
            )
            resample_sr0 = gr.Slider(
                minimum=0,
                maximum=48000,
                label=i18n(
                    "Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling"
                ),
                value=0,
                step=1,
                interactive=True,
            )
            rms_mix_rate0 = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n(
                    "Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume"
                ),
                value=1,
                interactive=True,
            )
            protect0 = gr.Slider(
                minimum=0,
                maximum=0.5,
                label=i18n(
                    "Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy"
                ),
                value=0.33,
                step=0.01,
                interactive=True,
            )
            f0_file = gr.File(
                label=i18n(
                    "F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation"
                )
            )
            but0 = gr.Button(i18n("Convert"), variant="primary")
            vc_output1 = gr.Textbox(label=i18n("Output information"))
            vc_output2 = gr.Audio(
                label=i18n(
                    "Export audio (click on the three dots in the lower right corner to download)"
                )
            )
            but0.click(
                vc.vc_single,
                [
                    spk_item,
                    vc_input3,
                    vc_transform0,
                    f0_file,
                    f0method0,
                    file_index1,
                    file_index2,
                    # file_big_npy1,
                    index_rate1,
                    filter_radius0,
                    resample_sr0,
                    rms_mix_rate0,
                    protect0,
                ],
                [vc_output1, vc_output2],
            )


app.launch()
