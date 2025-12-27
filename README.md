![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.8-orange)

# **AGIN**

AGIN is a **Stream Diffusion–style** algorithm for **real-time video stream processing** using diffusion models. It focuses on **temporal consistency, low latency, and high throughput**, enabling stable, high-resolution video generation.

![Demo](assets/demo.gif)

[▶ Full-resolution video demo](assets/large_demo.mp4)

## Overview

AGIN builds upon the ideas of Stream Diffusion and improves them in several key directions:

* **Extended temporal attention**
  Inspired by TokenFlow and adapted for **sequential frame processing**.
  This is the **core feature**, significantly improving temporal consistency and reducing jitter.

* **Large diffusion models with very few steps**
  Uses **SDXL-Turbo** with **2–4 diffusion steps**.

* **Frame interpolation (RIFE)**
  Reduces the “slideshow” effect and further improves visual smoothness.

## Performance Optimizations

AGIN applies multiple system-level and model-level optimizations to reduce latency and increase throughput:

* TensorRT compilation for **all pipeline models**
* Cached text embeddings
* **FP16** computation
* Optional **TAESDXL** decoder
* Model inference running in a **separate process**
* **Shared memory buffers** for efficient IPC
* No CFG overhead (SDXL-Turbo does not require it)

All of this enables **up to 20 FPS** at **1024 × 1024** resolution with **SDXL-level quality** on an **RTX 5090**.
Other GPUs are also supported, with slightly lower performance.

## Additional Features

* ControlNet supported as the **default mode**
* Simple, non-blocking API:

  * Asynchronous frame sending and receiving
  * Pipeline configuration can be changed on the fly
* Support for **arbitrary SDXL-compatible resolutions**
* **LoRA** support for the UNet

## Usage Example

```python
from agin import StreamProcessor
from agin.utils import crop_maximal_rectangle
import cv2

def main():
    config_path = "configs/stream_processor_config.json"

    stream_processor = StreamProcessor(config_path)
    input_tensor = stream_processor.get_input_tensor()
    output_tensor = stream_processor.get_output_tensor()

    stream_processor.start()
    stream_processor.set_prompt(
        "A man in a cyberpunk street at night, neon lamps, colorful lighting"
    )

    resolution = stream_processor.get_resolution()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = crop_maximal_rectangle(
            frame, resolution["height"], resolution["width"]
        )
        input_tensor.copy_from(resized_frame)

        processed_frame = output_tensor.to_numpy()
        cv2.imshow("Processed Stream", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream_processor.stop()

if __name__ == "__main__":
    main()
```


## System Requirements

- OS: Linux/Windows (macOS not tested)
- GPU: NVIDIA RTX 4090+
- RAM: 32 GB recommended
- VRAM: 24+ GB recommended for 1024 × 1024 resolution

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/krasnoteh/AGIN
cd agin
```

### 2. Install Python dependencies

> **Python 3.9 is recommended** (most stable for TensorRT)

```bash
conda create -n agin python=3.9 pip -y
conda activate agin

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

*(Adjust the CUDA version if needed.)*


### 3. Download Models

All models should be placed under:

```text
agin/models/
```

Example structure:

```text
agin/models
├── controlnet
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── interpolation_model
│   └── flownet.pkl
├── loras
│   └── your_lora.safetensors
├── scheduler
│   └── scheduler_config.json
├── taesdxl
│   ├── config.json
│   └── weights.safetensors
├── text_encoder
│   ├── config.json
│   └── model.safetensors
├── text_encoder_2
│   ├── config.json
│   └── model.safetensors
├── tokenizer
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── tokenizer_2
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet
│   ├── config.json
│   └── diffusion_pytorch_model.fp16.safetensors
└── vae
    ├── config.json
    └── weights.safetensors
```

#### Option 1: Download all models as a ZIP archive (Google Drive)

**All models:**
[https://drive.google.com/file/d/1GHuP9GCHO87EjUA9QQLhG_P-CJwf4sOK/view?usp=sharing](https://drive.google.com/file/d/1GHuP9GCHO87EjUA9QQLhG_P-CJwf4sOK/view?usp=sharing)

#### Option 2: Download models from individual sources

* **ControlNet (Scribble, SDXL)**
  [https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0](https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0)

* **TAESDXL**
  [https://huggingface.co/madebyollin/taesdxl](https://huggingface.co/madebyollin/taesdxl)

* **RIFE (frame interpolation)**
  [https://drive.google.com/file/d/1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX/view](https://drive.google.com/file/d/1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX/view)
  *(Required file: `flownet.pkl`)*

* **RIFE repository (backup)**
  [https://github.com/hzwer/ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE)

* **SDXL-Turbo and related models**
  [https://huggingface.co/stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo)


### 4. Compile TensorRT Engines

```bash
cd agin
conda activate agin
python scripts/compile_all_engines.py
```

#### Notes

* Compiled engines are stored in:

  ```text
  engines/square_engines/
  ```

* To use a different resolution:

  * Edit `configs/engine_compiler_config.json`
  * Consider using a separate directory (e.g. `portrait_engines`)
  * Update `configs/stream_processor_config.json` accordingly

* TensorRT engines are:
  * GPU-architecture specific
  * Resolution specific
  * Not portable across machines

  You must recompile engines when changing GPU or resolution.

For advanced or custom compilation (e.g. LoRA support), see:

```text
agin/src/agin/engine_compilation_tools
```


### 5. Run the demo

```bash
cd agin
conda activate agin
python scripts/run_cv2_demo.py
```


## Contributing

This project is research-oriented and under active development.

- Bug reports are welcome
- Please include logs, GPU model, and config files
- Feature requests may not be prioritized

## About the Name

The name stands for:

* **A**synchronous frame processing
* **G**lobal generation context
* **I**nterpolation of intermediate frames
* **N**o computational overhead

**AGIN** is also similar to the Kazakh word **“ағын”**, meaning *stream*.


## Credits & License

Developed by **Krasnoteh** in collaboration with **Noise to Signal**.

**License:** MIT