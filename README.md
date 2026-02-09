# LLM-KT: Knowledge Tracing with LLMs

Predict student performance using dual encoders + fine-tuned LLMs on the MOOCRadar dataset.

---

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd LLM-KT

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch transformers peft tqdm numpy scikit-learn gdown python-dotenv requests
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# WandB (optional - for experiment tracking)
WANDB_API_KEY=your_wandb_key_here

# Hugging Face (for downloading models)
HF_TOKEN=your_hf_token_here

# Proxy (optional - if behind corporate firewall)
HTTP_PROXY=http://your.proxy:port
HTTPS_PROXY=http://your.proxy:port
```

### 3. Prepare Data

The dataset will be **automatically downloaded** from Google Drive if not present:

```bash
# Quick test with small dataset
python prepare_data.py --preset small --output-dir processed/small

# Full dataset
python prepare_data.py --output-dir processed/full
```

**What this does:**
- Downloads `problem.json` and `student-problem-coarse-flattened.json` from Google Drive (if missing)
- Processes ~9.3K problems and ~9.9M student interactions
- Creates train/val/test splits
- Saves to `processed/` directory

### 4. Training (2-Stage Pipeline)

We split training into two independent steps for better control:

**Step 1: Train Encoder (AKT)**
Trains the Context + Sequence encoder to predict student performance (without LLM).

```bash
# Train on small preset
python train_encoder.py --preset small --processed-dir processed/small --epochs 10

# Output: checkpoints/akt/small/best_akt_model.pt
```

**Step 2: Fine-tune LLM**
Uses the pre-trained encoder to fine-tune the LLM for the final task.

```bash
# Fine-tune using the trained encoder
python finetune_llm.py \
    --preset small \
    --processed-dir processed/small \
    --encoder-path checkpoints/akt/small/best_akt_model.pt \
    --epochs 3
```

**Why this approach?**
1. **Efficiency**: Train the encoder quickly (lightweight) before the expensive LLM fine-tuning.
2. **Modularity**: Reuse the same good encoder for different LLMs (Phi-3, Llama 2, etc.).

### 5. Monitor Training

If you enabled WandB in `.env`:
```bash
python finetune_llm.py --preset small --processed-dir processed/small --wandb
```

---

## Project Structure

```
LLM-KT/
├── prepare_data.py      # Step 1: Data Prep
├── train_encoder.py     # Step 2: Encoder Training (AKT)
├── finetune_llm.py      # Step 3: LLM Fine-tuning
├── predict.py           # Manual Inference (CLI)
├── test.py              # Evaluation script
├── config.py            # Configuration
├── .env                 # API Keys
│
├── examples/            # Sample inputs
│   └── sample.json
│
├── models/
│   ├── encoders.py      # Context & Sequence Encoders
│   ├── embedding_model.py # KTEmbeddingModel Wrapper
│   └── kt_llm.py        # LLM Integration
│
├── checkpoints/
│   ├── akt/             # Encoder checkpoints
│   └── llm/             # LLM checkpoints
```

### 6. Complete Usage Guide

This section provides dedicated workflows for different use cases and hardware capabilities.

#### Available Presets

| Preset | Scenario | Model | Data Size | Recommended VRAM |
|--------|----------|-------|-----------|------------------|
| `small` | Debugging | TinyLlama-1.1B | 5 Users | < 4GB |
| `percent10` | Experimentation | TinyLlama-1.1B | ~10k Users | 8GB |
| `standard` | Production | Phi-2 (2.7B) | ~100 Users | 12GB |
| `phi3` | High Performance | Phi-3-Mini (3.8B) | ~200 Users | 16GB |
| `qwen` | Chinese Optimized | Qwen2-1.5B | ~200 Users | 12GB |

---

#### Scenario 1: Quick Debugging (`small`)

Use this to verify your setup is correct. Runs in minutes.

```bash
# 1. Prepare Data
python prepare_data.py --preset small --output-dir processed/small

# 2. Train Encoder (AKT)
python train_encoder.py --preset small --processed-dir processed/small --epochs 5

# 3. Fine-tune LLM
python finetune_llm.py \
    --preset small \
    --processed-dir processed/small \
    --encoder-path checkpoints/akt/small/best_akt_model.pt \
    --epochs 2

# 4. Test
python predict.py --preset small --checkpoint checkpoints/llm/small/best_model --input examples/sample.json
```

#### Scenario 2: Balanced Experiment (`percent10`) - **Recommended**

Use this for meaningful experiments on a logical subset (10%) of the data.

```bash
# 1. Prepare Data
python prepare_data.py --preset percent10 --output-dir processed/percent10

# 2. Train Encoder
python train_encoder.py --preset percent10 --processed-dir processed/percent10 --epochs 10

# 3. Fine-tune LLM
python finetune_llm.py \
    --preset percent10 \
    --processed-dir processed/percent10 \
    --encoder-path checkpoints/akt/percent10/best_akt_model.pt \
    --epochs 3 --save-every 1
```

#### Scenario 3: Production Training (`standard` / `phi3`)

For best results, use larger models and more data.

```bash
# 1. Prepare Data (replace 'phi3' with 'standard' or 'qwen' as needed)
python prepare_data.py --preset phi3 --output-dir processed/phi3

# 2. Train Encoder
python train_encoder.py --preset phi3 --processed-dir processed/phi3 --epochs 20

# 3. Fine-tune LLM
python finetune_llm.py \
    --preset phi3 \
    --processed-dir processed/phi3 \
    --encoder-path checkpoints/akt/phi3/best_akt_model.pt \
    --epochs 5 --save-every 1
```

---

#### Advanced Testing & Inference

After training any preset, you can evaluate the model.

**A. Manual Prediction**
Test with specific examples using `predict.py`.

```bash
python predict.py \
    --preset percent10 \
    --checkpoint checkpoints/llm/percent10/best_model \
    --processed-dir processed/percent10 \
    --input examples/sample.json
```

**B. Full Test Set Evaluation**
Calculate AUC and Accuracy on the held-out test set.

```bash
python test.py \
    --preset percent10 \
    --checkpoint checkpoints/llm/percent10/best_model \
    --split test
```

---

## Troubleshooting

### Network Issues

The project automatically handles proxy configuration:
1. Reads proxy from `.env`
2. Tests connectivity with proxy
3. Falls back to direct connection if proxy fails

Test your connection:
```bash
python test_proxy.py
```

### Missing Dataset Files

Dataset files are **automatically downloaded** from Google Drive when you run:
- `python prepare_data.py`
- `python train.py` (without `--processed-dir`)

If download fails:
1. Check your network connection
2. Verify proxy settings in `.env`
3. Or download manually from Google Drive and place in `dataset/MOOCRadar/`

### CUDA Out of Memory

```bash
# Reduce batch size
python train.py --preset small --batch-size 2

# Use gradient accumulation
python train.py --preset small --batch-size 2 --grad-accum 4

# Use smaller model
python train.py --preset small  # Uses TinyLlama
```

---

## Architecture

```
┌─────────────────────┐
│  Problem Text       │──┐
│  (Chinese)          │  │
└─────────────────────┘  │
                         ├──► Context Encoder
┌─────────────────────┐  │    (multilingual-e5-base)
│  Concept Names      │──┘
└─────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
┌─────────────────────┐  │                      │
│  Interaction        │──┤   Hybrid Encoder     │
│  History            │  │   (Combine Context   │
└─────────────────────┘  │    + Sequence)       │
                         └──────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   LLM (LoRA)         │
                         │   Phi-3 / Qwen /     │
                         │   Llama2             │
                         └──────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Prediction         │
                         │   Correct/Incorrect  │
                         └──────────────────────┘
```

---

## Dataset

**MOOCRadar** - Chinese MOOC learning data:
- **9,300+** problems with Chinese text
- **9.9M** student interaction logs
- **Multiple concepts** per problem
- **Temporal sequence** of student attempts

---

## Citation

If you use this code, please cite:

```bibtex
@software{llm_kt_2024,
  title={LLM-KT: Knowledge Tracing with Large Language Models},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/LLM-KT}
}
```

---

## License

MIT License - See LICENSE file for details
