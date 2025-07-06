# Multi-Token MoE Model Converter
### Fine-tune + Inference Guide

A model-agnostic Multi-Token Mixture of Experts (MoE) system that converts any Hugging Face causal language model into a multi-head expert architecture for parallel token generation.

## Overview

* **Model-agnostic**: Accepts any Hugging Face causal LM as frozen backbone
* **Multi-GPU support**: Backbone on `cuda:0`, Router + Heads on `cuda:1`
* **Three-phase curriculum** implements progressive training:
    1. **Stage 1 – Supervised Teacher-Forced CE**: Full teacher forcing with all heads active
    2. **Stage 2 – Annealed Teacher Forcing + k-sparsity**: Gradual transition to sparse routing with auxiliary losses
    3. **Stage 3 – PPO fine-tuning**: Reinforcement learning with composite rewards (accuracy, order, efficiency, clean stop)
* **Dataset format**: JSONL with `{"question": str, "response": str}`
    - Responses are token-split across H heads (round-robin by index)
* **Inference**: Greedy multi-token generation with automatic OFF token handling
* **Quantization support**: 8-bit and 4-bit quantization via BitsAndBytesConfig

## Architecture

```
Input → Backbone (frozen) → Router → [Head₁, Head₂, ..., Headₕ] → Multi-token output
                                 ↓
                            Gate logits → Top-k selection
```

- **Router**: 3-layer MLP that learns to route tokens to appropriate expert heads
- **Projection Heads**: Linear layers mapping hidden states to vocabulary
- **Gating**: Top-k sparse selection of active heads during inference

## Installation Requirements

```bash
pip install torch transformers trl bitsandbytes
```

## Dataset Preparation

Create a JSONL file where each line contains:
```json
{"question": "What is machine learning?", "response": "Machine learning is a subset of artificial intelligence..."}
{"question": "Explain neural networks", "response": "Neural networks are computational models..."}
```

## CLI Quick-start

### Training (All Stages)
```bash
python multi_moe_train.py train \
    --model microsoft/DialoGPT-medium \
    --data ./qa_dataset.jsonl \
    --out ./checkpoints \
    --heads 4 \
    --k_max 2 \
    --stage1_epochs 3 \
    --stage2_epochs 3 \
    --ppo_steps 1000 \
    --batch 4 \
    --lr 2e-4 \
    --quant none
```

### Inference
```bash
python multi_moe_train.py infer \
    --ckpt ./checkpoints/s2 \
    --prompt "What are the benefits of renewable energy?" \
    --max_tokens 100 \
    --quant none
```

## Training Parameters

### Core Model Parameters
- `--model`: Hugging Face model path/name (e.g., `microsoft/DialoGPT-medium`)
- `--heads`: Number of expert heads (default: 3)
- `--k_max`: Maximum active heads during sparse inference (default: 2)

### Training Configuration
- `--stage1_epochs`: Teacher-forced training epochs (default: 2)
- `--stage2_epochs`: Annealed training epochs (default: 2) 
- `--ppo_steps`: PPO fine-tuning steps (default: 0, set >0 to enable)
- `--batch`: Batch size (default: 3)
- `--lr`: Learning rate (default: 2e-4)

### Hardware Optimization
- `--quant`: Quantization mode [`none`, `8bit`, `4bit`] (default: `none`)

## Training Stages Explained

### Stage 1: Supervised Teacher-Forced CE
- All heads receive ground truth tokens via teacher forcing
- Tokens distributed round-robin across heads
- Full cross-entropy loss on all active heads
- Builds basic token-to-head associations

### Stage 2: Annealed Teacher Forcing + k-sparsity
- Gradually reduces teacher forcing probability (1.0 → 0.0)
- Progressively limits active heads (H → k_max)
- Additional losses:
  - **BCE loss**: Router prediction accuracy
  - **Load balancing**: Encourages equal head utilization
  - **Entropy regularization**: Prevents routing collapse
  - **Budget constraint**: Controls sparsity level

### Stage 3: PPO Fine-tuning (Optional)
- Reinforcement learning with generated responses
- Composite reward function based on:
  - Answer accuracy (substring matching)
  - Token ordering quality
  - Generation efficiency
  - Clean stopping behavior

## Advanced Usage

### Custom Loss Weights
Modify the loss coefficients in the `MultiTokenMoE` class:
```python
self.w_bce = 1.0      # Router BCE weight
self.w_lb = 0.2       # Load balancing weight  
self.w_ent = 0.01     # Entropy regularization weight
self.w_budget = 0.05  # Budget constraint weight
```

### Multi-GPU Setup
The system automatically uses:
- `cuda:0` for backbone model (inference only)
- `cuda:1` for router and projection heads (trainable)

### Resume Training
Training automatically resumes from the latest checkpoint in the output directory. Checkpoints are saved every 5000 steps.

### Custom Backbone Models
Any Hugging Face causal LM works:
```bash
# GPT-2 variants
--model gpt2-medium

# LLaMA models  
--model meta-llama/Llama-2-7b-hf

# Qwen models
--model Qwen/Qwen2-1.5B
```

## Output Structure

After training, the output directory contains:
```
checkpoints/
├── s1/              # Stage 1 outputs
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── router.pt
│   ├── heads.pt
│   └── moe_meta.json
├── s2/              # Stage 2 outputs (recommended for inference)
└── ppo_final/       # Stage 3 outputs (if PPO enabled)
```

## Performance Tips

1. **Memory optimization**: Use quantization for large models
2. **Batch size tuning**: Start small (2-4) and increase based on GPU memory
3. **Head count**: More heads = more parallelism but higher memory usage
4. **k_max selection**: Usually 50-70% of total heads works well

## Troubleshooting

**CUDA out of memory**: Reduce `--batch` size or enable `--quant 8bit`

**Poor routing**: Increase `--stage2_epochs` or adjust loss weights

**Slow convergence**: Increase learning rate or reduce teacher forcing annealing

**Generation stops early**: Check OFF token handling in dataset preprocessing

## Examples

### Small-scale experiment (1B parameters)
```bash
python multi_moe_train.py train \
    --model microsoft/DialoGPT-small \
    --data small_qa.jsonl \
    --out ./test_run \
    --heads 3 --k_max 2 \
    --stage1_epochs 1 --stage2_epochs 2 \
    --batch 2 --lr 1e-4
```

### Production training (7B+ parameters)
```bash
python multi_moe_train.py train \
    --model meta-llama/Llama-2-7b-hf \
    --data large_qa.jsonl \
    --out ./production \
    --heads 8 --k_max 4 \
    --stage1_epochs 3 --stage2_epochs 5 \
    --ppo_steps 2000 \
    --batch 8 --lr 5e-5 \
    --quant 4bit
```
