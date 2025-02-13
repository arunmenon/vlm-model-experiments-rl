# GRPO Trainer for Vision-Language Models

This repository implements a GRPO-based fine-tuning pipeline for Vision-Language Models (LLaVA, BLIP-2, etc.) to detect offensive content (Nudity, Weapons) with **high recall**.

## Features
- **Supervised Fine-Tuning (SFT)**
- **GRPO Training with Reinforcement Learning**
- **Custom Reward Function for High Recall**
- **Evaluation Metrics: Precision, Recall**

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the supervised fine-tuning first:
```bash
bash scripts/run_sft.sh
```

Then apply GRPO training:
```bash
bash scripts/run_grpo.sh
```

