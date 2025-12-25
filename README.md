Below is a cleaned-up and more polished version of your **README.md**, with improved Markdown structure, clearer wording, consistent terminology, and better readability—while preserving your original intent and technical depth.

You can copy-paste this directly as your new README.

---

# **AGIN**

> **Real-time video stream generation with diffusion models**

AGIN is a **Stream Diffusion–style** algorithm for **real-time video stream processing** using diffusion models. It focuses on **temporal consistency, low latency, and high throughput**, enabling stable video generation at high resolutions.

🎬 *(demo video placeholder: `demo.mp4`)*

---

## Overview

AGIN builds upon the ideas of Stream Diffusion and improves them in several key directions:

### Core Improvements

* **Extended temporal attention**
  Inspired by TokenFlow and adapted for **sequential frame processing**.
  This is the **core feature** that significantly improves temporal consistency and reduces jitter.

* **Large diffusion models with very few steps**
  Uses **SDXL-Turbo** with **2–4 diffusion steps**.

* **Frame interpolation (RIFE)**
  Reduces the “slide-show” effect and further improves visual smoothness.

---

## Performance Optimizations

AGIN applies multiple system-level and model-level optimizations to reduce latency and increase throughput:

* TensorRT compilation for **all pipeline models**
* Cached text embeddings
* **FP16** computation
* Optional **TAESDXL** decoder
* Model inference runs in a **separate process**
* **Shared memory buffers** for efficient IPC
* No CFG overhead (SDXL-Turbo does not require it)

### Performance

* **Up to 20 FPS**
* Resolution: **1024 × 1024**
* Hardware: **RTX 5090**

---

## Additional Features

* ControlNet supported as the **default mode**
* Simple, non-blocking API:

  * Sending/receiving frames is asynchronous
  * Pipeline configuration can be changed on the fly
* Supports **arbitrary SDXL-compatible resolutions**
* **LoRA** support for UNet

---

## Interface Example

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
        "A man in the cyberpunk street, night, neon lamps, colorful"
    )

    resolution = stream_processor.get_resolution()
    cap = cv2.VideoCapture(4)

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

---

## Setup

### 1. Install Python Dependencies

> **Python 3.9 is recommended** (most stable for TensorRT)

```bash
conda create -n agin python=3.9 pip -y
conda activate agin

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

*(Adjust CUDA version if needed.)*

---

### 2. Download Models

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

#### Recommended Sources

* **ControlNet (Scribble, SDXL)**
  [https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0](https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0)

* **TAESDXL**
  [https://huggingface.co/madebyollin/taesdxl](https://huggingface.co/madebyollin/taesdxl)

* **RIFE (frame interpolation)**
  [https://drive.google.com/file/d/1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX/view](https://drive.google.com/file/d/1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX/view)
  *(Required file: `flownet.pkl`)*

* **RIFE Repository (backup)**
  [https://github.com/hzwer/ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE)

* **SDXL-Turbo and related models**
  [https://huggingface.co/stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo)

---

### 3. Compile TensorRT Engines

```bash
cd agin
conda activate agin
python scripts/compile_all_engines.py
```

#### Notes

* Compiled engines will be stored in:

  ```text
  engines/square_engines/
  ```
* To use a different resolution:

  * Edit `configs/engine_compiler_config.json`
  * Consider using a separate directory (e.g. `portrait_engines`)
  * Update `configs/stream_processor_config.json` accordingly

For advanced or custom compilation (e.g. LoRA support), see:

```text
agin/src/agin/engine_compilation_tools
```

---

## Running the Demo

```bash
cd agin
conda activate agin
python scripts/run_cv3_demo.py
```

---

## About the Name

**“AGIN”** comes from the Kazakh word **“ағын”**, meaning *stream*.

It also stands for:

* **A**rbitrary model size
* **G**lobal generation context
* **I**ntermediate frame interpolation
* **N**ever-ending optimization

---

##  Credits & License

Developed by **CyberAY** in collaboration with **Noise2Signal**.

**License:** MIT
