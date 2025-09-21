<p align="center">
  <img src="assets/logo.png" height=200>
</p>
<hr>
<div align="center">
  
## 🎯 DICEPTION: A Generalist Diffusion Model for Vision Perception

<p align="center">
  <a href="https://aim-uofa.github.io/Diception/"><b>📖 Project Page</b></a> |
  <a href="https://arxiv.org/abs/2502.17157"><b>📄 Paper Link</b></a> |
  <a href="https://huggingface.co/spaces/Canyu/Diception-Demo"><b>🤗 Huggingface Demo</b></a>
</p>

</div>

> One single model solves multiple perception tasks, on par with SOTA!

## 📰 News

- 2025-09-21: 🚀 Model and inference code released
- 2025-09-19: 🌟 Accepted as NeurIPS 2025 Spotlight
- 2025-02-25: 📝 Paper released

## 🛠️ Installation
```
conda create -n diception python=3.10 -y

conda activate diception

pip install -r requirements.txt
```

## 👾 Inference

### ⚡ Quick Start

#### 🧩 Model Setup

1. **Download SD3 Base Model**:
   Download the Stable Diffusion 3 medium model from:
   https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers

2. **Download Trained Weights**:
   Please download the model from Hugging Face: https://huggingface.co/Canyu/DICEPTION

3. **Update Paths**:
   Set `--pretrained_model_path` to your SD3 path, and set `--diception_path` to the local path of the downloaded `DICEPTION_v1.pth`.

4. **Sample JSON for Batch Inference**:
   We provide several JSON examples for batch inference in the `DATA/jsons/evaluate` directory.


#### ▶️ Option 1: Simple Inference Script
For single image inference:

```bash
python inference.py \
    --image path/to/your/image.jpg \
    --prompt "[[image2depth]]" \
    --pretrained_model_path PATH_TO_SD3 \
    --diception_path PATH_TO_DICEPTION_v1.PTH \
    --output_dir ./outputs \
    --guidance_scale 2 \
    --num_inference_steps 28
```

**With coordinate points** (for interactive segmentation):

```bash
python inference.py \
    --image path/to/your/image.jpg \
    --prompt "[[image2segmentation]]" \
    --pretrained_model_path PATH_TO_SD3 \
    --diception_path PATH_TO_DICEPTION_v1.PTH \
    --output_dir ./outputs \
    --guidance_scale 2 \
    --num_inference_steps 28 \
    --points "0.3,0.5;0.7,0.2"
```

The `--points` parameter accepts coordinates in format `"y1,x1;y2,x2;y3,x3"` where:
- Coordinates are normalized to [0,1] range
- Format is (y,x) where y=height/image_height, x=width/image_width
- Multiple points are separated by semicolons
- Maximum 5 points are supported

#### 📦 Option 2: Batch Inference
For batch processing with a JSON dataset:

```bash
python batch_inference.py \
    --pretrained_model_path PATH_TO_SD3 \
    --diception_path PATH_TO_DICEPTION_v1.PTH \
    --input_path example_batch.json \
    --data_root_path ./ \
    --save_path ./batch_results \
    --batch_size 4 \
    --guidance_scale 2 \
    --num_inference_steps 28
    # --save_npy (for depth and normal value)
```

**JSON Format for Batch Inference**:
The input JSON file should contain a list of tasks in the following format:
```json
[
  {
    "input": "path/to/image1.jpg",
    "caption": "[[image2segmentation]]"
  },
  {
    "input": "path/to/image2.jpg", 
    "caption": "[[image2depth]]"
  },
  {
    "input": "path/to/image3.jpg",
    "caption": "[[image2segmentation]]",
    "target": {
      "path": "path/to/sa1b.json"   (For convenience, randomly select a region for point prompt from the GT json)
    }
  }
]
```

### 📋 Supported Tasks

DICEPTION supports various vision perception tasks:
- **Depth Estimation**: `[[image2depth]]` 
- **Surface Normal Estimation**: `[[image2normal]]`
- **Pose Estimation**: `[[image2pose]]`
- **Interactive Segmentation**: `[[image2segmentation]]`
- **Semantic Segmentation**: `[[image2semantic]] + (category in coco)`, e.g. `[[image2semantic]] person`
- **Entity Segmentation**: `[[image2entity]]`


### 💡 Inference Tips

- **General settings**: For best overall results, use `--num_inference_steps 28` and `--guidance_scale 2.0`.
- **1-step/few-step inference**: We found flow-matching diffusion models naturally support few-step inference, especially for tasks like depth and surface normals. DICEPTION can run with `--num_inference_steps 1` and `--guidance_scale 1.0` with barely quality loss. If you prioritize speed, consider this setting. We provide a detailed analysis in our NeurIPS paper.


### 🗺️ Plan
- [X] Release inference code and pretrained model v1
- [ ] Release training code
- [ ] Release few-shot finetuning code


## 🎫 License

For academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 
For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).

## 🖊️ Citation
```
@misc{zhao2025diceptiongeneralistdiffusionmodel,
      title={DICEPTION: A Generalist Diffusion Model for Visual Perceptual Tasks}, 
      author={Canyu Zhao and Mingyu Liu and Huanyi Zheng and Muzhi Zhu and Zhiyue Zhao and Hao Chen and Tong He and Chunhua Shen},
      year={2025},
      eprint={2502.17157},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.17157}, 
}
```
