<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
An easy-to-use Voice Conversion framework based on VITS.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>
  
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/github/license/RVC-Project/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

[**Changelog**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/Changelog_EN.md) | [**FAQ (Frequently Asked Questions)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/FAQ-(Frequently-Asked-Questions)) 

[**English**](../en/README.en.md) | [**中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Français**](../fr/README.fr.md) | [**Türkçe**](../tr/README.tr.md) | [**Português**](../pt/README.pt.md)

</div>

> Check out our [Demo Video](https://www.bilibili.com/video/BV1pm4y1z7Gm/) here!

<table>
   <tr>
		<td align="center">Training and inference Webui</td>
		<td align="center">Real-time voice changing GUI</td>
	</tr>
  <tr>
		<td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/092e5c12-0d49-4168-a590-0b0ef6a4f630"></td>
    <td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/730b4114-8805-44a1-ab1a-04668f3c30a6"></td>
	</tr>
	<tr>
		<td align="center">go-web.bat</td>
		<td align="center">go-realtime-gui.bat</td>
	</tr>
  <tr>
    <td align="center">You can freely choose the action you want to perform.</td>
		<td align="center">We have achieved an end-to-end latency of 170ms. With the use of ASIO input and output devices, we have managed to achieve an end-to-end latency of 90ms, but it is highly dependent on hardware driver support.</td>
	</tr>
</table>

> The dataset for the pre-training model uses nearly 50 hours of high quality audio from the VCTK open source dataset.

> High quality licensed song datasets will be added to the training-set often for your use, without having to worry about copyright infringement.

> Please look forward to the pretrained base model of RVCv3, which has larger parameters, more training data, better results, unchanged inference speed, and requires less training data for training.

## Features:
+ Reduce tone leakage by replacing the source feature to training-set feature using top1 retrieval;
+ Easy + fast training, even on poor graphics cards;
+ Training with a small amounts of data (>=10min low noise speech recommended);
+ Model fusion to change timbres (using ckpt processing tab->ckpt merge);
+ Easy-to-use WebUI;
+ UVR5 model to quickly separate vocals and instruments;
+ High-pitch Voice Extraction Algorithm [InterSpeech2023-RMVPE](#Credits) to prevent a muted sound problem. Provides the best results (significantly) and is faster with lower resource consumption than Crepe_full;
+ AMD/Intel graphics cards acceleration supported;
+ Intel ARC graphics cards acceleration with IPEX supported.

## Preparing the environment
The following commands need to be executed with Python 3.8 or higher.

(Windows/Linux)
First install the main dependencies through pip:
```bash
# Install PyTorch-related core dependencies, skip if installed
# Reference: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

#For Windows + Nvidia Ampere Architecture(RTX30xx), you need to specify the cuda version corresponding to pytorch according to the experience of https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/21
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

#For Linux + AMD Cards, you need to use the following pytorch versions:
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

Then can use poetry to install the other dependencies:
```bash
# Install the Poetry dependency management tool, skip if installed
# Reference: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# Install the project dependencies
poetry install
```

You can also use pip to install them:
```bash

for Nvidia graphics cards
  pip install -r requirements.txt

for AMD/Intel graphics cards on Windows (DirectML)：
  pip install -r requirements-dml.txt

for Intel ARC graphics cards on Linux / WSL using Python 3.10: 
  pip install -r requirements-ipex.txt

for AMD graphics cards on Linux (ROCm):
  pip install -r requirements-amd.txt
```

------
Mac users can install dependencies via `run.sh`:
```bash
sh ./run.sh
```

## Preparation of other Pre-models
RVC requires other pre-models to infer and train.

```bash
#Download all needed models from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/
python tools/download_models.py
```

Or just download them by yourself from our [Huggingface space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/).

Here's a list of Pre-models and other files that RVC needs:
```bash
./assets/hubert/hubert_base.pt

./assets/pretrained 

./assets/uvr5_weights

Additional downloads are required if you want to test the v2 version of the model.

./assets/pretrained_v2

If you want to test the v2 version model (the v2 version model has changed the input from the 256 dimensional feature of 9-layer Hubert+final_proj to the 768 dimensional feature of 12-layer Hubert, and has added 3 period discriminators), you will need to download additional features

./assets/pretrained_v2

If you want to use the latest SOTA RMVPE vocal pitch extraction algorithm, you need to download the RMVPE weights and place them in the RVC root directory

https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt

    For AMD/Intel graphics cards users you need download:

    https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx

```

### 2. Install FFmpeg
If you have FFmpeg and FFprobe installed on your computer, you can skip this step.

#### For Ubuntu/Debian users
```bash
sudo apt install ffmpeg
```
#### For MacOS users
```bash
brew install ffmpeg
```
#### For Windows users
Download these files and place them in the root folder:
- [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)

- [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)

## ROCm Support for AMD graphic cards (Linux only)
To use ROCm on Linux install all required drivers as described [here](https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html).

On Arch use pacman to install the driver:
````
pacman -S rocm-hip-sdk rocm-opencl-sdk
````

You might also need to set these environment variables (e.g. on a RX6700XT):
````
export ROCM_PATH=/opt/rocm #Set ROCM Executables Path
export HSA_OVERRIDE_GFX_VERSION=10.3.0 #Spoof GPU Model for ROCM
````

And overwrite PyTorch with its ROCM version after installing dependencies.
````
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
````

Make sure your user is part of the `render` and `video` group:
````
sudo usermod -aG render $USERNAME
sudo usermod -aG video $USERNAME
````

## Get started
### start up directly
Use the following command to start WebUI:
```bash
python infer-web.py
```
### Use the integration package
Download and extract file `RVC-beta.7z`, then follow the steps below according to your system:
#### For Windows users
Double click `go-web.bat`
#### For MacOS users
```bash
sh ./run.sh
```
### For Intel IPEX users (Linux Only)
```bash
source /opt/intel/oneapi/setvars.sh
```
## Credits
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  + The pretrained model is trained and tested by [yxlllc](https://github.com/yxlllc/RMVPE) and [RVC-Boss](https://github.com/RVC-Boss).
  
## Thanks to all contributors for their efforts
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>

