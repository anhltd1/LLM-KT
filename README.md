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
├── test.py              # Evaluation script
├── config.py            # Configuration
├── .env                 # API Keys
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

---

## Common Commands

### Data Preparation

```bash
# Small dataset for testing
python prepare_data.py --preset small --output-dir processed/small

# Custom configuration
python prepare_data.py \
    --problems-path dataset/MOOCRadar/problem.json \
    --interactions-path dataset/MOOCRadar/student-problem-coarse-flattened.json \
    --output-dir processed/custom \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --max-seq-len 200 \
    --max-students 50
```

### Training

```bash
# Basic training
python train.py --preset small --processed-dir processed/small --epochs 5

# With custom hyperparameters
python train.py --preset phi3 \
    --processed-dir processed/full \
    --epochs 10 \
    --batch-size 4 \
    --lr 5e-5 \
    --wandb

# Resume from checkpoint
python train.py --preset small \
    --processed-dir processed/small \
    --resume checkpoints/llm/small/epoch_3
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
