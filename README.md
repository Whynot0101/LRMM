# 🏆 LRMM: Latent Reward Model with MMDiT Architecture

Welcome to the official repository for **LRMM**! This project introduces a Latent Reward Model built upon the MMDiT architecture. Our framework supports evaluating and training cutting-edge reward models for text-to-image generation.

---

## ⚙️ Installation

**1. Create and activate a new Conda environment:**
```bash
conda create -n lrmm python=3.10 -y
conda activate lrmm
```

**2. Install PyTorch:**  
*(Adjust to your system's CUDA version, e.g., below is for CUDA 11.3)*
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

**3. Install the remaining dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🗂️ Data Preparation

First, create a directory to store all your datasets:
```bash
mkdir -p data
```

### 📥 Download HPDv3
```bash
hf download ymhao/HPDv3 --repo-type dataset --local-dir data/HPDv3
python utils/generate_hpdv3_test.py
```

### 📥 Download ImageRewardDB
```bash
mkdir -p data/ImageRewardDB
hf download zai-org/ImageRewardDB --repo-type dataset --local-dir data/ImageRewardDB
seq 1 32 | xargs -I {} unzip data/ImageRewardDB/images/train/train_{}.zip -d data/ImageRewardDB/images/train/train_{}
seq 1 2 | xargs -I {} unzip data/ImageRewardDB/images/test/test_{}.zip -d data/ImageRewardDB/images/test/test_{}
python utils/generate_imagerewarddb_test.py 
```

### 📥 Download HPDv2
```bash
hf download ymhao/HPDv2 --include test.tar.gz test.json --repo-type dataset --local-dir data/HPDv2
tar -xvf data/HPDv2/test.tar.gz -C data/HPDv2
python utils/generate_hpdv2_test.py 
```

### 📥 Download Pick-a-Pic v2
```bash
hf download liuhuohuo2/pick-a-pic-v2 --repo-type dataset --local-dir data/Pickapic
python utils/generate_pickapic.py
```

### 🔄 Generate `train.json`
Consolidate the downloaded datasets into a unified format:
```bash
python utils/concat_datasets.py
```

---

## 🧠 Model Preparation

### 🌟 LRMM-SD3 (Pre-trained)
You can use our pre-trained model hosted on Hugging Face:
```bash
hf download whynot0128/LRMM-SD3 --local-dir ./output/SD3_Reward/transformer
```
> 💡 **Note:** The test/validation scripts will automatically download the model from Hugging Face if you skip this step.

### ⚖️ Reward Models
> 💡 **Note:** `PickScore`, `ImageReward`, `AestheticScorer`, and `HPSv2` models will be **automatically downloaded** from Hugging Face when you run the reward scripts for the first time.

#### 🔧 HPSv3 
Due to dependency conflicts, HPSv3 requires being deployed as an API service on a **separate server**:

```bash
# On your separate API server:
pip install hpsv3 gradio
git clone https://github.com/MizzenAI/HPSv3.git
cd HPSv3
python gradio_demo/demo.py 
```
*After starting the server, make sure to update `self.api_url` in `rewards/hpsv3_api.py` to match your deployed server's IP and port (e.g., `http://<YOUR_SERVER_IP>:8000/infer_score`).*

---

## 🚀 Training

To train the LRMM-SD3 reward model on the prepared datasets, use the provided bash script. This script uses `accelerate` to launch distributed training. 

First, configure your `accelerate` environment to use 8 GPUs:
```bash
accelerate config
```
*(When prompted, select "This machine", "multi-GPU", and specify `8` for the number of GPUs).*

We train the model using **8 GPUs** for approximately **50 hours** with a global batch size of **128** (`WORLD_SIZE=8`, `ACCUMULATION_STEPS=16`, `train_batch_size=1`).

```bash
bash run_sd3_hpdv3.sh
```
> 💡 **Note:** You can customize parameters such as `BATCH_SIZE`, `LR`, `MAX_TRAIN_STEPS`, and the `OUTPUT_DIR` directly inside `run_sd3_hpdv3.sh`.

---

## 🧪 Validation

To quickly evaluate the trained model's accuracy and tie rate on a specific test dataset (e.g., HPDv3), run the validation script:

```bash
python validate.py \
  --transformer_path whynot0128/LRMM-SD3 \
  --model_path stabilityai/stable-diffusion-3-medium-diffusers \
  --train_data_dir data/HPDv3 \
  --cache_dir ./huggingface_cache/datasets \
  --timestep 1
```
> 💡 **Note:** You can pass custom paths and evaluation parameters using the command-line arguments. The results will be printed in a beautiful table format and saved to `validation_results.json`.

---

## 📊 Benchmark

To comprehensively benchmark different reward models (including LRMM-SD3, PickScore, HPSv2, AestheticScorer, ImageReward, etc.) across various datasets (HPDv2, HPDv3, Pick-a-Pic, ImageRewardDB), use the `rewards.py` script. It supports multi-GPU evaluation via `accelerate`.

```bash
accelerate launch --num_processes 8 rewards.py \
  --dataset_name HPDv2 HPDv3 Pick ImageRewardDB \
  --scorers LRMM-SD3 pickscore hpsv2 aesthetic clip imagereward hpsv3 \
  --transformer_path whynot0128/LRMM-SD3 \
  --model_path stabilityai/stable-diffusion-3-medium-diffusers \
  --cache_dir ./huggingface_cache/datasets
```
> 💡 **Note:** You can pass multiple datasets (`HPDv2`, `HPDv3`, `Pick`, `ImageRewardDB`) and multiple scorers (`LRMM-SD3`, `pickscore`, `hpsv2`, `aesthetic`, `clip`, `imagereward`, `hpsv3`, `HPS`, `MPS`) via command-line arguments. The benchmark results will be displayed in a markdown-like table and automatically appended to `benchmark_results.json`.
