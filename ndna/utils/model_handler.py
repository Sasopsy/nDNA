"""
Generic model handler that works with various HuggingFace transformer architectures.
Provides unified interface for accessing model components across different architectures.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel, 
    GPT2LMHeadModel, LlamaForCausalLM, T5ForConditionalGeneration,
    BertModel, RobertaModel
)
from typing import Dict, List, Tuple, Optional, Union
import warnings


class ModelHandler:
    """
    Unified handler for different HuggingFace model architectures.
    Automatically detects model type and provides consistent access to components.
    """
    
    SUPPORTED_ARCHITECTURES = {
        'gpt2': ['transformer', 'h', 'ln_f', 'lm_head'],
        'llama': ['model', 'layers', 'norm', 'lm_head'], 
        'phi': ['model', 'layers', 'final_layernorm', 'lm_head'],  # Microsoft Phi models
        'bert': ['bert', 'encoder', 'layer', None],  # No lm_head for base BERT
        'roberta': ['roberta', 'encoder', 'layer', None],
        't5': ['encoder', 'block', 'final_layer_norm', 'lm_head'],
    }
    
    def __init__(self, model_name: str, device: str = "auto", torch_dtype: torch.dtype = torch.float32):
        """
        Initialize model handler.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ('cuda', 'cpu', or 'auto')
            torch_dtype: Data type for model parameters
        """
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.tokenizer = None
        self.model = None
        self.architecture = None
        self.is_causal = None
        
        self._load_model()
        self._detect_architecture()
    
    def _load_model(self):
        """Load tokenizer and model from HuggingFace."""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        
        # Set pad token if missing (common for GPT-style models)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Try to load as causal LM first, then fall back to base model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=self.torch_dtype
            )
            self.is_causal = True
        except Exception as e:
            print(f"Failed to load as causal LM: {e}")
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype
                )
                self.is_causal = False
            except Exception as e2:
                raise ValueError(f"Failed to load model: {e2}")
        
        self.model.eval().to(self.device)
        print(f"Model loaded on {self.device}, causal: {self.is_causal}")
    
    def _detect_architecture(self):
        """Detect model architecture and set component paths."""
        model_class = self.model.__class__.__name__.lower()
        
        if 'gpt2' in model_class or 'dialogpt' in model_class:
            self.architecture = 'gpt2'
        elif 'llama' in model_class:
            self.architecture = 'llama'
        elif 'phi' in model_class:
            self.architecture = 'phi'
        elif 'bert' in model_class and 'roberta' not in model_class:
            self.architecture = 'bert'
        elif 'roberta' in model_class:
            self.architecture = 'roberta'
        elif 't5' in model_class:
            self.architecture = 't5'
        else:
            # Try to infer from model structure
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                self.architecture = 'gpt2'
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                # Check if it's Phi or Llama based on normalization layer
                if hasattr(self.model.model, 'final_layernorm'):
                    self.architecture = 'phi'
                elif hasattr(self.model.model, 'norm'):
                    self.architecture = 'llama'
                else:
                    self.architecture = 'llama'  # Default to llama for similar structures
            else:
                warnings.warn(f"Unknown architecture for {model_class}. Attempting generic handling.")
                self.architecture = 'generic'
    
    def get_transformer_blocks(self) -> List[torch.nn.Module]:
        """Get list of transformer blocks/layers."""
        if self.architecture == 'gpt2':
            return list(self.model.transformer.h)
        elif self.architecture in ['llama', 'phi']:
            return list(self.model.model.layers)
        elif self.architecture in ['bert', 'roberta']:
            encoder = getattr(self.model, self.architecture)
            return list(encoder.encoder.layer)
        elif self.architecture == 't5':
            return list(self.model.encoder.block)
        else:
            # Generic fallback - try common patterns
            for attr_name in ['transformer.h', 'model.layers', 'encoder.layer', 'encoder.block']:
                try:
                    obj = self.model
                    for part in attr_name.split('.'):
                        obj = getattr(obj, part)
                    return list(obj)
                except AttributeError:
                    continue
            raise ValueError(f"Could not find transformer blocks for architecture: {self.architecture}")
    
    def get_final_layer_norm(self) -> Optional[torch.nn.Module]:
        """Get final layer normalization."""
        if self.architecture == 'gpt2':
            return self.model.transformer.ln_f
        elif self.architecture == 'llama':
            return self.model.model.norm
        elif self.architecture == 'phi':
            return self.model.model.final_layernorm
        elif self.architecture == 't5':
            return self.model.encoder.final_layer_norm
        elif self.architecture in ['bert', 'roberta']:
            return None  # BERT/RoBERTa don't have a final layer norm before classification
        return None
    
    def get_lm_head(self) -> Optional[torch.nn.Module]:
        """Get language modeling head."""
        if not self.is_causal:
            return None
        
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head
        elif hasattr(self.model, 'cls') and hasattr(self.model.cls, 'predictions'):
            return self.model.cls.predictions  # BERT-style
        return None
    
    def get_embeddings(self) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
        """Get word and position embeddings."""
        if self.architecture == 'gpt2':
            return self.model.transformer.wte, self.model.transformer.wpe
        elif self.architecture in ['llama', 'phi']:
            return self.model.model.embed_tokens, None  # Llama/Phi use RoPE, no learned pos embeddings
        elif self.architecture in ['bert', 'roberta']:
            base_model = getattr(self.model, self.architecture)
            return base_model.embeddings.word_embeddings, base_model.embeddings.position_embeddings
        elif self.architecture == 't5':
            return self.model.shared, None  # T5 uses relative position encoding
        return None, None
    
    def forward_with_hidden_states(self, input_ids: torch.Tensor, **kwargs) -> Dict:
        """
        Forward pass returning all hidden states.
        
        Returns:
            Dict with 'hidden_states' key containing tuple of all layer outputs
        """
        with torch.no_grad():
            return self.model(input_ids, output_hidden_states=True, **kwargs)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.model.config.vocab_size
    
    @property
    def num_layers(self) -> int:
        """Get number of transformer layers."""
        return len(self.get_transformer_blocks())
    
    @property
    def hidden_size(self) -> int:
        """Get hidden dimension size."""
        return self.model.config.hidden_size
