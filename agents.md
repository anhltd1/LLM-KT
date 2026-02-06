# Agents Guide

Development guidelines for AI agents working on the LLM-KT codebase.

## Project Overview

**Goal**: Predict student performance (correct/incorrect) using dual encoders + LLM.

**Key Files**:
- `prepare_data.py` - Data preprocessing CLI
- `train.py` - Training CLI
- `config.py` - All configuration presets
- `models/encoders.py` - Context/Sequence/Hybrid encoders
- `models/kt_llm.py` - LLM integration with LoRA
- `utils/data_loader.py` - Dataset classes

## Conventions

### Code Style
- Type hints on all function signatures
- Docstrings with Args/Returns sections
- Use `Config` class for all hyperparameters

### Data Flow
```
Raw JSON → prepare_data.py → .pt files → train.py → model checkpoints
```

## Common Tasks

### Adding a New Model Preset

Edit `config.py`:
```python
MODEL_PRESETS["my_preset"] = {
    "name": "My Model",
    "LLM_MODEL_NAME": "microsoft/phi-3",
    "EMBED_DIM": 768,
    "MAX_SEQ_LEN": 200,
    # ... other params
}
```

### Adding a New Data Source

1. Create loader function in `utils/data_loader.py`
2. Update `prepare_data.py` to support new format
3. Ensure output matches existing sample structure:
```python
{
    'question_ids': [...],
    'concept_ids': [...],
    'responses': [...],
    'target_question_id': int,
    'target_response': int,
    # ...
}
```

### Modifying Encoder Architecture

Edit `models/encoders.py`:
- `ContextEncoder`: Text encoding (BERT-based)
- `SequenceEncoder`: History encoding (Transformer)
- `HybridEncoder`: Combining both

## Testing

```bash
# Quick test with small preset
python prepare_data.py --preset small --output-dir processed/test
python train.py --preset small --processed-dir processed/test --epochs 1
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `--batch-size` or use `--grad-accum` |
| Missing data | Run `prepare_data.py` first |
| Chinese text issues | Use `intfloat/multilingual-e5-base` encoder |
