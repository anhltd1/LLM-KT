"""
============================================================================
KNOWLEDGE TRACING CONFIGURATION
============================================================================
Centralized configuration for the entire LLM-KT training pipeline.
Modify these parameters to experiment with different settings.

Usage:
    from config import Config
    
    # Use a preset
    Config.use_preset("small")   # For quick testing
    Config.use_preset("standard")  # For production training
    
    # Or customize individual parameters
    Config.MAX_USERS = 100
    
    Config.print_config()
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# MODEL PRESETS - Quick configuration switching
# ============================================================================

MODEL_PRESETS = {
    # -------------------------------------------------------------------------
    # SMALL PRESET - For quick testing and debugging
    # Uses smallest models, minimal data, fast training
    # -------------------------------------------------------------------------
    "small": {
        "name": "Small (Testing)",
        "description": "Minimal configuration for quick testing and debugging",
        
        # Data (minimal)
        "MAX_INTERACTIONS": 100,
        "MAX_USERS": 5,
        "MAX_SEQ_LEN": 50,
        
        # AKT Model (small)
        "EMBED_DIM": 768,          # Match small LLM hidden size
        "NUM_HEADS": 4,
        "NUM_LAYERS": 2,
        "AKT_BATCH_SIZE": 32,
        "AKT_EPOCHS": 5,
        "AKT_LR": 1e-4,
        
        # LLM (smallest available)
        "LLM_MODEL_NAME": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "LLM_HIDDEN_SIZE": 2048,   # TinyLlama hidden size
        "LLM_BATCH_SIZE": 8,
        "LLM_EPOCHS": 2,
        "LLM_LR": 2e-4,
        
        # LoRA (lightweight)
        "LORA_R": 8,
        "LORA_ALPHA": 16,
        "LORA_DROPOUT": 0.1,
        
        # Soft prompts
        "NUM_SOFT_TOKENS": 4,
        "MAX_PROMPT_LENGTH": 256,
    },
    
    # -------------------------------------------------------------------------
    # STANDARD PRESET - Balanced for good results
    # Uses Phi-2 or similar mid-size models
    # -------------------------------------------------------------------------
    "standard": {
        "name": "Standard (Balanced)",
        "description": "Balanced configuration for good results with reasonable resources",
        
        # Data (moderate)
        "MAX_INTERACTIONS": 5000,
        "MAX_USERS": 100,
        "MAX_SEQ_LEN": 100,
        
        # AKT Model
        "EMBED_DIM": 2560,         # Match Phi-2 hidden size
        "NUM_HEADS": 8,
        "NUM_LAYERS": 4,
        "AKT_BATCH_SIZE": 64,
        "AKT_EPOCHS": 20,
        "AKT_LR": 5e-5,
        
        # LLM
        "LLM_MODEL_NAME": "microsoft/phi-2",
        "LLM_HIDDEN_SIZE": 2560,   # Phi-2 hidden size
        "LLM_BATCH_SIZE": 4,
        "LLM_EPOCHS": 5,
        "LLM_LR": 1e-4,
        
        # LoRA
        "LORA_R": 16,
        "LORA_ALPHA": 32,
        "LORA_DROPOUT": 0.1,
        
        # Soft prompts
        "NUM_SOFT_TOKENS": 8,
        "MAX_PROMPT_LENGTH": 512,
    },
    
    # -------------------------------------------------------------------------
    # PHI3 PRESET - Using Microsoft Phi-3 Mini
    # Good balance of performance and efficiency
    # -------------------------------------------------------------------------
    "phi3": {
        "name": "Phi-3 Mini",
        "description": "Microsoft Phi-3 Mini - excellent performance for its size",
        
        # Data
        "MAX_INTERACTIONS": 10000,
        "MAX_USERS": 200,
        "MAX_SEQ_LEN": 100,
        
        # AKT Model
        "EMBED_DIM": 3072,         # Match Phi-3 hidden size
        "NUM_HEADS": 8,
        "NUM_LAYERS": 4,
        "AKT_BATCH_SIZE": 64,
        "AKT_EPOCHS": 20,
        "AKT_LR": 5e-5,
        
        # LLM
        "LLM_MODEL_NAME": "microsoft/Phi-3-mini-4k-instruct",
        "LLM_HIDDEN_SIZE": 3072,   # Phi-3 Mini hidden size
        "LLM_BATCH_SIZE": 2,
        "LLM_EPOCHS": 5,
        "LLM_LR": 5e-5,
        
        # LoRA
        "LORA_R": 16,
        "LORA_ALPHA": 32,
        "LORA_DROPOUT": 0.05,
        
        # Soft prompts
        "NUM_SOFT_TOKENS": 8,
        "MAX_PROMPT_LENGTH": 512,
    },
    
    # -------------------------------------------------------------------------
    # QWEN PRESET - Using Qwen2-1.5B
    # Good multilingual support
    # -------------------------------------------------------------------------
    "qwen": {
        "name": "Qwen2-1.5B",
        "description": "Alibaba Qwen2 - good for multilingual tasks",
        
        # Data
        "MAX_INTERACTIONS": 10000,
        "MAX_USERS": 200,
        "MAX_SEQ_LEN": 100,
        
        # AKT Model
        "EMBED_DIM": 1536,         # Match Qwen2-1.5B hidden size
        "NUM_HEADS": 8,
        "NUM_LAYERS": 4,
        "AKT_BATCH_SIZE": 64,
        "AKT_EPOCHS": 20,
        "AKT_LR": 5e-5,
        
        # LLM
        "LLM_MODEL_NAME": "Qwen/Qwen2-1.5B",
        "LLM_HIDDEN_SIZE": 1536,   # Qwen2-1.5B hidden size
        "LLM_BATCH_SIZE": 4,
        "LLM_EPOCHS": 5,
        "LLM_LR": 1e-4,
        
        # LoRA
        "LORA_R": 16,
        "LORA_ALPHA": 32,
        "LORA_DROPOUT": 0.1,
        
        # Soft prompts
        "NUM_SOFT_TOKENS": 8,
        "MAX_PROMPT_LENGTH": 512,
    },
    
    # -------------------------------------------------------------------------
    # LLAMA2 PRESET - Using Llama-2-7B (requires more GPU memory)
    # -------------------------------------------------------------------------
    "llama2": {
        "name": "Llama-2-7B",
        "description": "Meta Llama 2 - powerful but requires more resources",
        
        # Data
        "MAX_INTERACTIONS": 20000,
        "MAX_USERS": 500,
        "MAX_SEQ_LEN": 100,
        
        # AKT Model
        "EMBED_DIM": 4096,         # Match Llama-2 hidden size
        "NUM_HEADS": 8,
        "NUM_LAYERS": 4,
        "AKT_BATCH_SIZE": 64,
        "AKT_EPOCHS": 20,
        "AKT_LR": 5e-5,
        
        # LLM
        "LLM_MODEL_NAME": "meta-llama/Llama-2-7b-hf",
        "LLM_HIDDEN_SIZE": 4096,   # Llama-2 hidden size
        "LLM_BATCH_SIZE": 2,
        "LLM_EPOCHS": 3,
        "LLM_LR": 5e-5,
        
        # LoRA
        "LORA_R": 16,
        "LORA_ALPHA": 32,
        "LORA_DROPOUT": 0.05,
        
        # Soft prompts
        "NUM_SOFT_TOKENS": 8,
        "MAX_PROMPT_LENGTH": 512,
    },
}


class Config:
    """
    Centralized configuration for the entire training pipeline.
    Modify these parameters to experiment with different settings.
    """
    
    # ==================== ACTIVE PRESET ====================
    ACTIVE_PRESET = "small"  # Change this to switch presets: "small", "standard", "phi3", "qwen", "llama2"
    current_preset = "small"  # Alias for convenience
    
    # ==================== DATA PARAMETERS ====================
    MAX_INTERACTIONS = 500
    MAX_USERS = 10
    MAX_SEQ_LEN = 100
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # ==================== AKT MODEL PARAMETERS ====================
    EMBED_DIM = 768            # Will be adapted to match LLM hidden size
    NUM_HEADS = 8
    NUM_LAYERS = 2
    AKT_BATCH_SIZE = 64
    AKT_EPOCHS = 10
    AKT_LR = 5e-5
    AKT_WEIGHT_DECAY = 1e-4
    AKT_WARMUP_EPOCHS = 5
    AKT_LABEL_SMOOTHING = 0.1
    AKT_DROPOUT = 0.15
    AKT_EARLY_STOP_PATIENCE = 5
    
    # ==================== LLM PARAMETERS ====================
    LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LLM_HIDDEN_SIZE = 2048     # Hidden size of the LLM (for embedding adaptation)
    NUM_SOFT_TOKENS = 8
    MAX_PROMPT_LENGTH = 512
    LLM_BATCH_SIZE = 4
    LLM_EPOCHS = 3
    LLM_LR = 1e-4
    LLM_WEIGHT_DECAY = 1e-5
    
    # ==================== LORA CONFIGURATION ====================
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # ==================== FILE PATHS ====================
    DATA_DIR = "dataset/MOOCRadar"
    PROBLEM_JSON = f"{DATA_DIR}/problem.json"
    STUDENT_JSON = f"{DATA_DIR}/student-problem-coarse-flattened.json"
    
    METADATA_FILE = "students_metadata_enriched.json"
    AKT_CHECKPOINT_DIR = "checkpoints/akt"
    LLM_CHECKPOINT_DIR = "checkpoints/llm"
    AKT_LOG_DIR = "logs"
    AKT_BEST_MODEL = "checkpoints/akt/best_akt_model.ckpt"
    LLM_BEST_MODEL = "checkpoints/llm/best_llm_model.pth"
    
    TRAIN_EMBEDDINGS = "embeddings/student_embeddings_train.npy"
    VAL_EMBEDDINGS = "embeddings/student_embeddings_val.npy"
    ALL_EMBEDDINGS = "embeddings/student_embeddings_all.npy"
    
    # Convenience aliases for notebook
    CHECKPOINT_DIR = "checkpoints"
    EMBEDDING_DIR = "embeddings"
    OUTPUT_DIR = "outputs"
    
    # ==================== TRAINING OPTIONS ====================
    USE_GPU = True
    MIXED_PRECISION = True
    GRADIENT_ACCUMULATION = 2
    
    # ==================== WANDB CONFIGURATION ====================
    USE_WANDB = True
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")  # Load from .env file
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "letrongducanh456-viettel")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "LLM-KT")
    WANDB_RUN_NAME = None
    WANDB_SAVE_MODEL = True    # Save model checkpoints to wandb
    
    @classmethod
    def use_preset(cls, preset_name: str):
        """
        Apply a preset configuration.
        
        Args:
            preset_name: One of "small", "standard", "phi3", "qwen", "llama2"
        """
        if preset_name not in MODEL_PRESETS:
            available = ", ".join(MODEL_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        
        preset = MODEL_PRESETS[preset_name]
        cls.ACTIVE_PRESET = preset_name
        cls.current_preset = preset_name  # Update alias
        
        # Apply all preset values
        for key, value in preset.items():
            if key not in ["name", "description"] and hasattr(cls, key):
                setattr(cls, key, value)
        
        # Update file paths to include preset name
        cls.AKT_CHECKPOINT_DIR = f"checkpoints/akt/{preset_name}"
        cls.LLM_CHECKPOINT_DIR = f"checkpoints/llm/{preset_name}"
        cls.AKT_BEST_MODEL = f"checkpoints/akt/{preset_name}/best_akt_model.ckpt"
        cls.LLM_BEST_MODEL = f"checkpoints/llm/{preset_name}/best_llm_model.pth"
        cls.TRAIN_EMBEDDINGS = f"embeddings/{preset_name}/student_embeddings_train.npy"
        cls.VAL_EMBEDDINGS = f"embeddings/{preset_name}/student_embeddings_val.npy"
        cls.ALL_EMBEDDINGS = f"embeddings/{preset_name}/student_embeddings_all.npy"
        
        # Update convenience directories
        cls.CHECKPOINT_DIR = f"checkpoints/{preset_name}"
        cls.EMBEDDING_DIR = f"embeddings/{preset_name}"
        cls.OUTPUT_DIR = f"outputs/{preset_name}"
        
        print(f"✓ Applied preset: {preset['name']}")
        print(f"  {preset['description']}")
        print(f"  LLM: {cls.LLM_MODEL_NAME}")
        print(f"  Embedding dim: {cls.EMBED_DIM} → LLM hidden: {cls.LLM_HIDDEN_SIZE}")
    
    @classmethod
    def list_presets(cls):
        """List all available presets"""
        print("\n" + "=" * 70)
        print("AVAILABLE MODEL PRESETS")
        print("=" * 70)
        for key, preset in MODEL_PRESETS.items():
            active = " ← ACTIVE" if key == cls.ACTIVE_PRESET else ""
            print(f"\n📦 {key}{active}")
            print(f"   Name: {preset['name']}")
            print(f"   Description: {preset['description']}")
            print(f"   LLM: {preset['LLM_MODEL_NAME']}")
            print(f"   Hidden Size: {preset['LLM_HIDDEN_SIZE']}")
            print(f"   Data: {preset['MAX_USERS']} users, {preset['MAX_INTERACTIONS']} interactions")
        print("\n" + "=" * 70)
        print("Usage: Config.use_preset('preset_name')")
        print("=" * 70)
    
    @classmethod
    def get_embedding_adapter_config(cls):
        """
        Get configuration for the embedding adapter.
        This adapts AKT embeddings to LLM hidden size.
        """
        return {
            "akt_embed_dim": cls.EMBED_DIM,
            "llm_hidden_size": cls.LLM_HIDDEN_SIZE,
            "num_soft_tokens": cls.NUM_SOFT_TOKENS,
            "adapter_hidden_dim": (cls.EMBED_DIM + cls.LLM_HIDDEN_SIZE) // 2,
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        preset_info = MODEL_PRESETS.get(cls.ACTIVE_PRESET, {})
        
        print("\n" + "=" * 70)
        print("TRAINING CONFIGURATION")
        print("=" * 70)
        
        print(f"\n🎯 ACTIVE PRESET: {cls.ACTIVE_PRESET.upper()}")
        if preset_info:
            print(f"   {preset_info.get('name', '')} - {preset_info.get('description', '')}")
        
        print("\n📊 DATA:")
        print(f"  Max Interactions: {cls.MAX_INTERACTIONS:,}")
        print(f"  Max Users: {cls.MAX_USERS}")
        print(f"  Max Sequence Length: {cls.MAX_SEQ_LEN}")
        print(f"  Test Split: {cls.TEST_SIZE*100:.0f}%")
        
        print("\n🧠 AKT MODEL:")
        print(f"  Embedding Dim: {cls.EMBED_DIM}")
        print(f"  Attention Heads: {cls.NUM_HEADS}")
        print(f"  Transformer Layers: {cls.NUM_LAYERS}")
        print(f"  Batch Size: {cls.AKT_BATCH_SIZE}")
        print(f"  Epochs: {cls.AKT_EPOCHS}")
        print(f"  Learning Rate: {cls.AKT_LR}")
        print(f"  Dropout: {cls.AKT_DROPOUT}")
        
        print("\n🤖 LLM FINE-TUNING:")
        print(f"  Model: {cls.LLM_MODEL_NAME}")
        print(f"  LLM Hidden Size: {cls.LLM_HIDDEN_SIZE}")
        print(f"  Soft Tokens: {cls.NUM_SOFT_TOKENS}")
        print(f"  Batch Size: {cls.LLM_BATCH_SIZE}")
        print(f"  Epochs: {cls.LLM_EPOCHS}")
        print(f"  Learning Rate: {cls.LLM_LR}")
        
        print("\n🔗 EMBEDDING ADAPTER:")
        adapter_cfg = cls.get_embedding_adapter_config()
        print(f"  AKT → LLM: {adapter_cfg['akt_embed_dim']} → {adapter_cfg['llm_hidden_size']}")
        print(f"  Adapter Hidden: {adapter_cfg['adapter_hidden_dim']}")
        print(f"  Soft Tokens: {adapter_cfg['num_soft_tokens']}")
        
        print("\n🔧 LORA:")
        print(f"  Rank: {cls.LORA_R}")
        print(f"  Alpha: {cls.LORA_ALPHA}")
        print(f"  Dropout: {cls.LORA_DROPOUT}")
        
        print("\n⚙️  OPTIMIZATION:")
        print(f"  GPU: {'✓ Enabled' if cls.USE_GPU else '✗ Disabled'}")
        print(f"  Mixed Precision: {'✓ Enabled' if cls.MIXED_PRECISION else '✗ Disabled'}")
        print(f"  Gradient Accumulation: {cls.GRADIENT_ACCUMULATION} steps")
        
        print("\n📊 EXPERIMENT TRACKING:")
        print(f"  Wandb: {'✓ Enabled' if cls.USE_WANDB else '✗ Disabled'}")
        if cls.USE_WANDB:
            print(f"  Entity: {cls.WANDB_ENTITY}")
            print(f"  Project: {cls.WANDB_PROJECT}")
            print(f"  Save Model: {'✓ Yes' if cls.WANDB_SAVE_MODEL else '✗ No'}")
        
        print("\n📁 OUTPUT PATHS:")
        print(f"  AKT Checkpoints: {cls.AKT_CHECKPOINT_DIR}")
        print(f"  LLM Checkpoints: {cls.LLM_CHECKPOINT_DIR}")
        print(f"  Embeddings: embeddings/{cls.ACTIVE_PRESET}/")
        
        print("=" * 70)


# Auto-apply the default preset on import
Config.use_preset(Config.ACTIVE_PRESET)


if __name__ == "__main__":
    # Demo: List presets and show current config
    Config.list_presets()
    print("\n")
    Config.print_config()
