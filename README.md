<div align="center">

# Retrieval-based Voice Conversion (WebUI)

An easy-to-use Voice Conversion framework based on VITS.
<br><br>

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/github/license/fumiama/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI)

</div>

## Features

- Minimize tone leakage with top1 retrieval replacing the source feature
- Quick and easy training, even on low-end GPUs
- Train with minimal data (recommend >=10min of low-noise speech)
- Model fusion for timbre variation (via ckpt processing tab->ckpt merge)
- User-friendly WebUI
- UVR5 model for efficient vocals and instrument separation
- High-pitch Voice Extraction Algorithm [InterSpeech2023-RMVPE](#Credits) for superior results with faster processing
- Support for AMD/Intel graphics card acceleration
- Intel ARC graphics card acceleration with IPEX support

## Environment Setup

Ensure Python 3.8 or higher is installed, then follow these steps:

1. Install main dependencies via pip:

```bash
pip install torch torchvision torchaudio
```

2. Install additional dependencies using Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

Or, use pip directly:

```bash
pip install -r requirements/main.txt
pip install -r requirements/dml.txt  # for AMD/Intel graphics cards on Windows (DirectML)
pip install -r requirements/ipex.txt  # for Intel ARC graphics cards on Linux / WSL using Python 3.10
pip install -r requirements/amd.txt  # for AMD graphics cards on Linux (ROCm)
```

For Mac users:

```bash
sh ./run.sh
```

3. Download required pre-models:

```bash
python tools/download_models.py
```

Or manually download from our [Hugging Face repository](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/).

4. Install FFmpeg:

- For Ubuntu/Debian:

```bash
sudo apt install ffmpeg
```

- For MacOS:

```bash
brew install ffmpeg
```

- For Windows, download and place `ffmpeg.exe` and `ffprobe.exe` in the root folder.

## ROCm Support for AMD Graphics Cards (Linux)

For Linux users with AMD graphics cards, follow these steps:

1. Install required drivers as described [here](https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html).
2. For Arch, use pacman to install the driver:

```bash
pacman -S rocm-hip-sdk rocm-opencl-sdk
```

3. Set environment variables (e.g., on a RX6700XT):

```bash
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

4. Ensure your user is part of the `render` and `video` group:

```bash
sudo usermod -aG render $USERNAME
sudo usermod -aG video $USERNAME
```

## Getting Started

### Direct Startup

Launch the WebUI with:

```bash
python infer-web.py
```

### Integration Package

1. Download and extract `RVC-beta.7z`.
2. Choose the appropriate script for your system:
   - For Windows, open `go-web.bat`.
   - For MacOS, run:

```bash
sh ./run.sh
```

### Intel IPEX Users (Linux Only)

```bash
source /opt/intel/oneapi/setvars.sh
```

## Credits

This project wouldn't be possible without the contributions of numerous individuals and open-source projects:

- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [VITS](https://github.com/jaywalnut310/vits)
- [HIFIGAN](https://github.com/jik876/hifi-gan)
- [Gradio](https://github.com/gradio-app/gradio)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [RMVPE](https://github.com/Dream-High/RMVPE) _Trained and tested by [yxlllc](https://github.com/yxlllc/RMVPE) & [RVC-Boss](https://github.com/RVC-Boss)._

## Acknowledgments

Special thanks to all contributors for their dedication and support. Check out our [contributors](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors).

<a href="https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
<img src="https://contrib.rocks/image?repo=fumiama/Retrieval-based-Voice-Conversion-WebUI" />
</a>
