"""
Knowledge Tracing LLM Model

Integrates the dual encoder embedding system with an LLM for knowledge tracing.
Uses soft prompts with [QuesEmbed] and [ConcEmbed] tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
import os

from .embedding_model import KTEmbeddingModel


class PromptFormatter:
    """
    Formats prompts for the Knowledge Tracing LLM.
    
    Creates prompts with embedded question and concept representations:
    "The student has previously answered {question QID=38 [QuesEmbed38] 
    involving concept CID=219 [ConcEmbed219] correctly, ...}.
    Please predict whether the student will answer {question QID=55 [QuesEmbed55] 
    involving concept CID=245 [ConcEmbed245] correctly}.
    Response with 'Yes' or 'No'."
    """
    
    # Special tokens for embeddings
    QUES_EMBED_TOKEN = "[QuesEmbed{}]"
    CONC_EMBED_TOKEN = "[ConcEmbed{}]"
    
    # Prompt templates
    HISTORY_ITEM_TEMPLATE = "question QID={} {} involving concept CID={} {} {}"
    PREDICTION_TEMPLATE = (
        "The student has previously answered {}. "
        "Please predict whether the student will answer {{question QID={} {} "
        "involving concept CID={} {}}} correctly. "
        "Response with 'Yes' or 'No'."
    )
    
    @classmethod
    def format_history_item(
        cls,
        question_id: int,
        concept_id: int,
        correct: bool
    ) -> str:
        """Format a single history item."""
        ques_token = cls.QUES_EMBED_TOKEN.format(question_id)
        conc_token = cls.CONC_EMBED_TOKEN.format(concept_id)
        result = "correctly" if correct else "incorrectly"
        
        return cls.HISTORY_ITEM_TEMPLATE.format(
            question_id, ques_token, 
            concept_id, conc_token, 
            result
        )
    
    @classmethod
    def format_prompt(
        cls,
        history: List[Tuple[int, int, bool]],  # (qid, cid, correct)
        target_question_id: int,
        target_concept_id: int,
        max_history: int = 10
    ) -> str:
        """
        Format the full prediction prompt.
        
        Args:
            history: List of (question_id, concept_id, correct) tuples
            target_question_id: Target question ID
            target_concept_id: Target concept ID
            max_history: Maximum history items to include
            
        Returns:
            Formatted prompt string
        """
        # Format history items
        history_items = []
        for qid, cid, correct in history[-max_history:]:
            item = cls.format_history_item(qid, cid, correct)
            history_items.append(item)
        
        history_str = ", ".join(history_items) if history_items else "no questions"
        
        # Format target
        target_ques_token = cls.QUES_EMBED_TOKEN.format(target_question_id)
        target_conc_token = cls.CONC_EMBED_TOKEN.format(target_concept_id)
        
        prompt = cls.PREDICTION_TEMPLATE.format(
            history_str,
            target_question_id, target_ques_token,
            target_concept_id, target_conc_token
        )
        
        return prompt
    
    @classmethod
    def get_embedding_tokens(cls, prompt: str) -> List[Tuple[str, int]]:
        """
        Extract embedding token positions from a prompt.
        
        Args:
            prompt: The formatted prompt
            
        Returns:
            List of (token_type, id) tuples where token_type is 'question' or 'concept'
        """
        import re
        
        tokens = []
        
        # Find question embed tokens
        ques_pattern = r'\[QuesEmbed(\d+)\]'
        for match in re.finditer(ques_pattern, prompt):
            tokens.append(('question', int(match.group(1))))
        
        # Find concept embed tokens
        conc_pattern = r'\[ConcEmbed(\d+)\]'
        for match in re.finditer(conc_pattern, prompt):
            tokens.append(('concept', int(match.group(1))))
        
        return tokens


class KnowledgeTracingLLM(nn.Module):
    """
    Knowledge Tracing LLM with Dual Encoder Embeddings.
    
    Combines:
    - KTEmbeddingModel: Produces question and concept embeddings
    - LLM: Generates predictions based on embedded prompts
    - LoRA: Efficient fine-tuning
    """
    
    def __init__(
        self,
        llm_model_name: str,
        embedding_model: KTEmbeddingModel,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: List[str] = None,
        use_4bit: bool = True,
        max_prompt_length: int = 512,
        num_soft_tokens: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Knowledge Tracing LLM.
        
        Args:
            llm_model_name: Name of the LLM to use
            embedding_model: Pre-initialized KTEmbeddingModel
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            lora_target_modules: Modules to apply LoRA to
            use_4bit: Whether to use 4-bit quantization
            max_prompt_length: Maximum prompt length
            num_soft_tokens: Number of soft prompt tokens per embedding
            device: Device to use
        """
        super().__init__()
        
        self.llm_model_name = llm_model_name
        self.embedding_model = embedding_model
        self.max_prompt_length = max_prompt_length
        self.num_soft_tokens = num_soft_tokens
        self.device = device
        
        print(f"\n[KnowledgeTracingLLM] Initializing...")
        print(f"  → LLM: {llm_model_name}")
        print(f"  → Device: {device}")
        
        # Default target modules for LoRA
        if lora_target_modules is None:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        # Load LLM configuration
        self._load_llm(
            llm_model_name, use_4bit,
            lora_r, lora_alpha, lora_dropout, lora_target_modules
        )
        
        # Set up embedding projection to match LLM hidden size
        self.embedding_model.set_llm_projection(self.llm_hidden_size)
        
        # Soft prompt tokens for embedding injection
        self.soft_prompt_tokens = nn.Parameter(
            torch.randn(num_soft_tokens, self.llm_hidden_size)
        )
        nn.init.normal_(self.soft_prompt_tokens, mean=0, std=0.02)
        
        # Register special tokens
        self._add_special_tokens()
        
        self.prompt_formatter = PromptFormatter()
        
        print(f"  → KnowledgeTracingLLM ready!")
    
    def _load_llm(
        self,
        model_name: str,
        use_4bit: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: List[str]
    ):
        """Load and configure the LLM with LoRA."""
        print(f"  → Loading LLM: {model_name}")
        
        # Load config first
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Handle Phi-3 specific config
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            if isinstance(config.rope_scaling, dict):
                config.rope_scaling['type'] = config.rope_scaling.get(
                    'type', 
                    config.rope_scaling.get('rope_type', 'dynamic')
                )
        
        # Quantization config
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        # Use device_map to place model on single GPU (avoid multi-GPU issues)
        # For 4-bit quantization, we need device_map, but force single device
        if use_4bit:
            # Force to cuda:0 to avoid multi-GPU splitting
            device_map = {"": 0}
        else:
            device_map = None
            
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="eager"  # For compatibility
        )
        
        self.llm_hidden_size = self.llm.config.hidden_size
        print(f"  → LLM hidden size: {self.llm_hidden_size}")
        
        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            inference_mode=False
        )
        
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()
    
    def _add_special_tokens(self):
        """Add special tokens for embeddings."""
        # We'll handle embedding tokens dynamically rather than adding them
        # to the vocabulary, since we inject embeddings directly
        pass
    
    def prepare_inputs_with_embeddings(
        self,
        prompt: str,
        question_embeddings: Dict[int, torch.Tensor],
        concept_embeddings: Dict[int, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs by replacing embedding tokens with actual embeddings.
        
        Args:
            prompt: Formatted prompt with [QuesEmbed] and [ConcEmbed] tokens
            question_embeddings: Dict mapping question IDs to embeddings
            concept_embeddings: Dict mapping concept IDs to embeddings
            
        Returns:
            Dictionary with input_ids, attention_mask, and inputs_embeds
        """
        import re
        
        # Tokenize the prompt
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_prompt_length,
            truncation=True,
            padding=True
        )
        
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        # Get base embeddings from LLM
        word_embeddings = self.llm.get_input_embeddings()
        inputs_embeds = word_embeddings(input_ids)
        
        # Find and replace embedding tokens
        # This is a simplified version - for production, we'd need
        # more sophisticated token position matching
        
        # Get token positions for special tokens
        prompt_tokens = self.tokenizer.tokenize(prompt)
        
        # For each embedding token in the prompt, replace with actual embedding
        # Note: This is a simplified implementation
        # In practice, you'd need careful token position tracking
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'inputs_embeds': inputs_embeds
        }
    
    def forward(
        self,
        batch: Dict[str, Any],
        compute_embeddings: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            batch: Batch dictionary from KTDataset
            compute_embeddings: Whether to compute embeddings (False for cached)
            
        Returns:
            Dictionary with loss and logits
        """
        device = self.device
        
        # Move tensors to device
        question_ids = batch['question_ids'].to(device)
        concept_ids = batch['concept_ids'].to(device)
        responses = batch['responses'].to(device)
        mask = batch['mask'].to(device)
        target_response = batch['target_response'].to(device)
        
        # Also move target tensors to device
        target_question_id = batch['target_question_id'].to(device)
        target_concept_ids = batch['target_concept_ids'].to(device)
        
        batch_size = question_ids.size(0)
        
        # Compute embeddings for each sample
        all_embeddings = []
        all_prompts = []
        
        for i in range(batch_size):
            # Get target info (use already moved tensors)
            target_qid = target_question_id[i].item()
            target_q_text = batch['target_question_text'][i]
            target_c_ids = target_concept_ids[i]
            target_c_texts = batch['target_concept_texts'][i]
            
            # Use first concept
            target_cid = target_c_ids[0].item()
            target_c_text = target_c_texts[0] if target_c_texts else ""
            
            if compute_embeddings:
                # Get embeddings from the embedding model
                q_emb, c_emb = self.embedding_model.get_embeddings_for_prompt(
                    question_text=target_q_text,
                    question_id=str(target_qid),
                    concept_text=target_c_text,
                    concept_id=str(target_cid),
                    history_question_ids=question_ids[i:i+1],
                    history_concept_ids=concept_ids[i:i+1],
                    history_responses=responses[i:i+1],
                    mask=mask[i:i+1]
                )
                
                all_embeddings.append((q_emb, c_emb))
            
            # Build history for prompt
            seq_len = mask[i].sum().item()
            history = []
            for j in range(seq_len):
                h_qid = question_ids[i, j].item()
                h_cid = concept_ids[i, j].item()
                h_correct = responses[i, j].item() == 1
                history.append((h_qid, h_cid, h_correct))
            
            # Format prompt
            prompt = self.prompt_formatter.format_prompt(
                history=history,
                target_question_id=target_qid,
                target_concept_id=target_cid
            )
            all_prompts.append(prompt)
        
        # Tokenize prompts
        encodings = self.tokenizer(
            all_prompts,
            return_tensors="pt",
            max_length=self.max_prompt_length,
            truncation=True,
            padding=True
        ).to(device)
        
        # Create labels (Yes=correct, No=incorrect)
        # We'll compute loss based on next token prediction
        labels = encodings['input_ids'].clone()
        labels[encodings['attention_mask'] == 0] = -100
        
        # Forward through LLM
        outputs = self.llm(
            input_ids=encodings['input_ids'],
            attention_mask=encodings['attention_mask'],
            labels=labels
        )
        
        # Binary classification loss for Yes/No
        yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
        
        # Get logits for Yes/No tokens
        last_logits = outputs.logits[:, -1, :]  # [batch, vocab]
        yes_no_logits = torch.stack([
            last_logits[:, no_token_id],
            last_logits[:, yes_token_id]
        ], dim=1)  # [batch, 2]
        
        # Classification loss
        ce_loss = F.cross_entropy(yes_no_logits, target_response)
        
        # Combine with LM loss
        total_loss = outputs.loss + ce_loss
        
        # Predictions
        predictions = yes_no_logits.argmax(dim=1)
        accuracy = (predictions == target_response).float().mean()
        
        return {
            'loss': total_loss,
            'lm_loss': outputs.loss,
            'ce_loss': ce_loss,
            'logits': yes_no_logits,
            'predictions': predictions,
            'accuracy': accuracy
        }
    
    def predict(
        self,
        history: List[Tuple[int, int, bool]],
        target_question_text: str,
        target_question_id: int,
        target_concept_text: str,
        target_concept_id: int,
        history_question_ids: torch.Tensor,
        history_concept_ids: torch.Tensor,
        history_responses: torch.Tensor,
        history_mask: torch.Tensor
    ) -> Tuple[int, float]:
        """
        Make a single prediction.
        
        Args:
            history: List of (qid, cid, correct) tuples
            target_question_text: Target question text
            target_question_id: Target question ID
            target_concept_text: Target concept text
            target_concept_id: Target concept ID
            history_question_ids: History question ID tensor
            history_concept_ids: History concept ID tensor
            history_responses: History response tensor
            history_mask: History mask tensor
            
        Returns:
            Tuple of (prediction, probability)
        """
        self.eval()
        device = self.device
        
        with torch.no_grad():
            # Get embeddings
            q_emb, c_emb = self.embedding_model.get_embeddings_for_prompt(
                question_text=target_question_text,
                question_id=str(target_question_id),
                concept_text=target_concept_text,
                concept_id=str(target_concept_id),
                history_question_ids=history_question_ids.unsqueeze(0).to(device),
                history_concept_ids=history_concept_ids.unsqueeze(0).to(device),
                history_responses=history_responses.unsqueeze(0).to(device),
                mask=history_mask.unsqueeze(0).to(device)
            )
            
            # Format prompt
            prompt = self.prompt_formatter.format_prompt(
                history=history,
                target_question_id=target_question_id,
                target_concept_id=target_concept_id
            )
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_prompt_length,
                truncation=True
            ).to(device)
            
            # Generate
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].size(1):],
                skip_special_tokens=True
            ).strip().lower()
            
            prediction = 1 if response.startswith('yes') else 0
            
            # Get probability from logits
            with torch.no_grad():
                logits = self.llm(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                ).logits[:, -1, :]
                
                yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
                no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
                
                probs = F.softmax(logits[:, [no_token_id, yes_token_id]], dim=-1)
                probability = probs[0, 1].item()
            
            return prediction, probability
    
    def save(self, save_dir: str):
        """Save model and embeddings."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save LoRA adapter
        self.llm.save_pretrained(os.path.join(save_dir, "lora_adapter"))
        
        # Save embedding model
        torch.save(
            self.embedding_model.state_dict(),
            os.path.join(save_dir, "embedding_model.pt")
        )
        
        # Save soft prompt tokens
        torch.save(
            self.soft_prompt_tokens,
            os.path.join(save_dir, "soft_prompt_tokens.pt")
        )
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        
        print(f"  → Model saved to {save_dir}")
    
    def load(self, save_dir: str):
        """Load saved model components."""
        # Load embedding model
        self.embedding_model.load_state_dict(
            torch.load(os.path.join(save_dir, "embedding_model.pt"))
        )
        
        # Load soft prompt tokens
        self.soft_prompt_tokens = nn.Parameter(
            torch.load(os.path.join(save_dir, "soft_prompt_tokens.pt"))
        )
        
        print(f"  → Model loaded from {save_dir}")
