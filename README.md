<div align="center">

# Retrieval-based-Voice-Conversion-WebUI
An easy-to-use voice conversion framework based on VITS.



[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI)

![moe](https://counter.seku.su/cmoe?name=rvc&theme=r34)

[![Licence](https://img.shields.io/github/license/fumiama/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

[**FAQ (Frequently Asked Questions)**](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/wiki/FAQ-(Frequently-Asked-Questions)) 

[**English**](./README.md) | [**中文简体**](./docs/cn/README.cn.md) | [**日本語**](./docs/jp/README.ja.md) | [**한국어**](./docs/kr/README.ko.md) ([**韓國語**](./docs/kr/README.ko.han.md)) | [**Français**](./docs/fr/README.fr.md) | [**Türkçe**](./docs/tr/README.tr.md) | [**Português**](./docs/pt/README.pt.md)

</div>

> The base model is trained using nearly 50 hours of high-quality open-source VCTK training set. Therefore, there are no copyright concerns, please feel free to use.

> Please look forward to the base model of RVCv3 with larger parameters, larger dataset, better effects, basically flat inference speed, and less training data required.

> There's a [one-click downloader](https://github.com/fumiama/RVC-Models-Downloader) for models/integration packages/tools. Welcome to try.

| Training and inference Webui |
| :--------: |
| ![web](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/assets/41315874/17e48404-2627-4fad-a0ec-65f9065aeade) |

| Real-time voice changing GUI |
| :---------: |
| ![realtime-gui](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/assets/41315874/95b36866-b92d-40c7-b5db-6a35ca5caeac) |

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

Check out our [Demo Video](https://www.bilibili.com/video/BV1pm4y1z7Gm/) here!

## Environment Configuration
### Python Version Limitation
> It is recommended to use venv to manage the Python environment.

> For the reason of the version limitation, please refer to this [bug](https://github.com/facebookresearch/fairseq/issues/5012).

```bash
python --version # 3.8 <= Python < 3.11
```

### Linux/MacOS One-click Dependency Installation & Startup Script
By executing `run.sh` in the project root directory, you can configure the `venv` virtual environment, automatically install the required dependencies, and start the main program with one click.
```bash
sh ./run.sh
```

### Manual Installation of Dependencies
1. Install `pytorch` and its core dependencies, skip if already installed. Refer to: https://pytorch.org/get-started/locally/
	```bash
	pip install torch torchvision torchaudio
	```
2. If you are using Nvidia Ampere architecture (RTX30xx) in Windows, according to the experience of #21, you need to specify the cuda version corresponding to pytorch.
	```bash
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
	```

3. Install the corresponding dependencies according to your own graphics card.
- Nvidia GPU
	```bash
	pip install -r requirements/main.txt
	```
- AMD/Intel GPU
	```bash
	pip install -r requirements/dml.txt
	```
- AMD ROCM (Linux)
	```bash
	pip install -r requirements/amd.txt
	```
- Intel IPEX (Linux)
	```bash
	pip install -r requirements/ipex.txt
	```

4.If you are using an ROCM-capable AMD Radeon GPU, then you need to choose ROCM version of PyTorch.
	```bash
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
	```

## Preparation of Other Files
### 1. Assets
> RVC requires some models located in the `assets` folder for inference and training.
#### Check/Download Automatically (Default)
> By default, RVC can automatically check the integrity of the required resources when the main program starts.

> Even if the resources are not complete, the program will continue to start.

- If you want to download all resources, please add the `--update` parameter.
- If you want to skip the resource integrity check at startup, please add the `--nocheck` parameter.

#### Download Manually
> All resource files are located in [Hugging Face space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

> You can find some scripts to download them in the `tools` folder

> You can also use the [one-click downloader](https://github.com/fumiama/RVC-Models-Downloader) for models/integration packages/tools

Below is a list that includes the names of all pre-models and other files required by RVC.

- ./assets/hubert/hubert_base.pt
	```bash
	rvcmd assets/hubert # RVC-Models-Downloader command
	```
- ./assets/pretrained
	```bash
	rvcmd assets/v1 # RVC-Models-Downloader command
	```
- ./assets/uvr5_weights
	```bash
	rvcmd assets/uvr5 # RVC-Models-Downloader command
	```
If you want to use the v2 version of the model, you need to download additional resources in

- ./assets/pretrained_v2
	```bash
	rvcmd assets/v2 # RVC-Models-Downloader command
	```

### 2. Download the required files for the rmvpe vocal pitch extraction algorithm

If you want to use the latest RMVPE vocal pitch extraction algorithm, you need to download the pitch extraction model parameters and place them in `assets/rmvpe`.

- [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)
	```bash
	rvcmd assets/rmvpe # RVC-Models-Downloader command
	```

#### Download DML environment of RMVPE (optional, for AMD/Intel GPU)

- [rmvpe.onnx](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx)
	```bash
	rvcmd assets/rmvpe # RVC-Models-Downloader command
	```

### 3. AMD ROCM (optional, Linux only)

If you want to run RVC on a Linux system based on AMD's ROCM technology, please first install the required drivers [here](https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html).

If you are using Arch Linux, you can use pacman to install the required drivers.
````
pacman -S rocm-hip-sdk rocm-opencl-sdk
````
For some models of graphics cards, you may need to configure the following environment variables (such as: RX6700XT).
````
export ROCM_PATH=/opt/rocm #Set ROCM Executables Path
export HSA_OVERRIDE_GFX_VERSION=10.3.0 #Spoof GPU Model for ROCM
````
Also, make sure your current user is in the `render` and `video` user groups.
````
sudo usermod -aG render $USERNAME
sudo usermod -aG video $USERNAME
````
## Getting Started
### Direct Launch
Use the following command to start the WebUI.
```bash
python web.py
```
### Linux/MacOS
```bash
./run.sh
```
### For I-card users who need to use IPEX technology (Linux only)
```bash
source /opt/intel/oneapi/setvars.sh
./run.sh
```
### Using the Integration Package (Windows Users)
Download and unzip `RVC-beta.7z`. After unzipping, double-click `go-web.bat` to start the program with one click.
```bash
rvcmd packs/general/latest # RVC-Models-Downloader command
```

## Credits
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  + The pretrained model is trained and tested by [yxlllc](https://github.com/yxlllc/RMVPE) and [RVC-Boss](https://github.com/RVC-Boss).

## Thanks to all contributors for their efforts
[![contributors](https://contrib.rocks/image?repo=fumiama/Retrieval-based-Voice-Conversion-WebUI)](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors)
