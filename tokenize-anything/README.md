<!-- <div align="center"> -->

<h1>Tokenize Anything via Prompting</h1>
<br><br><image src="assets/model_overview.png"/>

## News
[Feb.25 2024] Referring to SAM (Meta), TapPredictor has been implemented to enable multiple prompt-based inference through a single image input.

[Feb.25 2024] Referring to SAM (Meta), the implementation of TapAutomaticMaskGenerator enables the automatic generation of diverse masks, each accompanied by its corresponding caption.

[Feb.25 2024] FlashAttention of flash-attn is not available on some devices, Attention implemented with pytorch is equivalent to replace FlashAttention of flash-attn, now flash-attn is not required to be installed.

## Installation

### Preliminaries

``torch``

``gradio-image-prompter`` (for GradioApp, Install from [URL](https://github.com/PhyscalX/gradio-image-prompter))

### Installing Package

Clone this repository to local disk and install:

```bash
git clone https://github.com/Youlixiya/tokenize-anything.git
cd tokenize-anything && pip install .
```

You can also install from the remote repository: 

```bash
pip install git+ssh://git@github.com/Youlixiya/tokenize-anything.git
```

## Quick Start

### Development

The **TAP** models can be used for diverse vision and language tasks. 

We adopt a modular design that decouples all components and predictors.

As a best practice, implement your custom predictor and asynchronous pipeline as follows:

```python
from tokenize_anything import model_registry

with <distributed_actor>:
    model = model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
    results = <custom_predictor>(model, *args, **kwargs)

server.collect_results()
```

See builtin examples (web-demo and evaluations) provided in [scripts](scripts/) for more details.

### Inference

See [Inference Guide](notebooks/inference.ipynb).

See [Concept Guide](notebooks/concept.ipynb).

### Evaluation

See [Evaluation Guide for TAP-L](notebooks/evaluation_tap_vit_l.ipynb).

See [Evaluation Guide for TAP-B](notebooks/evaluation_tap_vit_b.ipynb).

## Models

### Model weights

Two versions of the model are available with different image encoders.

| Model | Description | MD5 | Weights |
| ----- | ------------| ----| ------ |
| **tap_vit_l** | ViT-L TAP model | 03f8ec | [🤗 HF link](https://huggingface.co/BAAI/tokenize-anything/blob/main/models/tap_vit_l_03f8ec.pkl) |
| **tap_vit_b** | ViT-B TAP model | b45cbf | [🤗 HF link](https://huggingface.co/BAAI/tokenize-anything/blob/main/models/tap_vit_b_b45cbf.pkl) |

### Concept weights

***Note***: You can generate these weights following the [Concept Guide](notebooks/concept.ipynb).

| Concept | Description | Weights |
| ------- | ------------| ------ |
| **Merged-2560** | Merged concepts | [🤗 HF link](https://huggingface.co/BAAI/tokenize-anything/blob/main/concepts/merged_2560.pkl) |
| **LVIS-1203**   | LVIS concepts | [🤗 HF link](https://huggingface.co/BAAI/tokenize-anything/blob/main/concepts/lvis_1203.pkl) |
| **COCO-80**   | COCO concepts  | [🤗 HF link](https://huggingface.co/BAAI/tokenize-anything/blob/main/concepts/coco_80.pkl) |


## License
[Apache License 2.0](LICENSE)

## Acknowledgement

We thank the repositories: [TAP](https://github.com/baaivision/tokenize-anything), [SAM](https://github.com/facebookresearch/segment-anything), [EVA](https://github.com/baaivision/EVA), [LLaMA](https://github.com/facebookresearch/llama), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [Gradio](https://github.com/gradio-app/gradio), [Detectron2](https://github.com/facebookresearch/detectron2) and [CodeWithGPU](https://github.com/seetacloud/codewithgpu).
