"""
Thermodynamic Length Calculator

Implements two methods for calculating thermodynamic length:
1. Parameter strain method: Based on gradient norms per layer
2. Fisher-Rao method: Based on prediction changes between layers

References:
- The original notebook implementations
- Information geometry and Fisher-Rao metrics
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union

from .utils.model_handler import ModelHandler


class ThermodynamicCalculator:
    """
    Calculator for thermodynamic length metrics in neural networks.
    
    Supports both parameter strain (gradient-based) and Fisher-Rao (prediction-based) methods.
    """
    
    def __init__(self, 
                 model_handler: ModelHandler,
                 amp_dtype: torch.dtype = torch.bfloat16,
                 eps_numerical: float = 1e-9):
        """
        Initialize thermodynamic calculator.
        
        Args:
            model_handler: Initialized ModelHandler instance
            amp_dtype: Data type for automatic mixed precision
            eps_numerical: Small epsilon for numerical stability
        """
        self.model_handler = model_handler
        self.device = model_handler.device
        self.amp_dtype = amp_dtype if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        self.eps = eps_numerical
        
        # Get model components
        self.transformer_blocks = model_handler.get_transformer_blocks()
        self.final_ln = model_handler.get_final_layer_norm()
        self.lm_head = model_handler.get_lm_head()
        self.num_layers = len(self.transformer_blocks)
        
        # Verify we have what we need for causal LM calculations
        if model_handler.is_causal and (self.final_ln is None or self.lm_head is None):
            raise ValueError("Causal LM model must have final layer norm and lm_head for thermodynamic calculations")
    
    def calculate_parameter_strain(self, dataloader: DataLoader) -> Dict[str, Union[np.ndarray, float]]:
        """
        Calculate parameter strain (gradient-based thermodynamic length).
        
        Computes per-layer squared gradient norms using observed Fisher information approach.
        
        Args:
            dataloader: DataLoader with batched text data
            
        Returns:
            Dictionary containing:
            - 'layer_grad_norms': Per-layer mean squared gradient norms
            - 'total_strain': Sum of all layer strains
            - 'num_batches': Number of batches processed
        """
        if not self.model_handler.is_causal:
            raise ValueError("Parameter strain calculation requires causal language model")
        
        print(f"Computing parameter strain across {self.num_layers} layers...")
        
        # Accumulators
        layer_grad_sums = torch.zeros(self.num_layers, device=self.device)
        num_batches = 0
        
        # Get model components
        model = self.model_handler.model
        transformer = getattr(model, 'transformer', getattr(model, 'model', model))
        
        for batch in tqdm(dataloader, desc="Processing batches"):
            num_batches += 1
            
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            B, S = input_ids.shape
            
            # Build initial embeddings
            # Disable autocast temporarily to debug
            # with torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu", 
            #                    dtype=self.amp_dtype, enabled=(self.device == "cuda")):
            # Get embeddings
            wte, wpe = self.model_handler.get_embeddings()
            h = wte(input_ids)
            
            if wpe is not None:
                pos_ids = torch.arange(S, dtype=torch.long, device=self.device).unsqueeze(0).expand(B, S)
                h = h + wpe(pos_ids)
            
            # Apply dropout if available
            if hasattr(transformer, 'drop'):
                h = transformer.drop(h)
            
            # Per-layer losses
            per_layer_losses = []
            
            for ell in range(self.num_layers):
                h_prev = h.detach()  # Isolate gradients to current block
                
                # Disable autocast temporarily to debug
                # with torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu",
                #                    dtype=self.amp_dtype, enabled=(self.device == "cuda")):
                # Forward through one block with proper arguments for different architectures
                if self.model_handler.architecture in ['llama', 'phi']:
                    # Llama/Phi models need position_embeddings (RoPE)
                    position_ids = torch.arange(S, dtype=torch.long, device=self.device).unsqueeze(0).expand(B, S)
                    
                    # Get RoPE embeddings from the model
                    if hasattr(self.model_handler.model.model, 'rotary_emb'):
                        rotary_emb = self.model_handler.model.model.rotary_emb
                        cos, sin = rotary_emb(h_prev, position_ids)
                        position_embeddings = (cos, sin)
                    else:
                        position_embeddings = None
                    
                    block_output = self.transformer_blocks[ell](
                        h_prev,
                        attention_mask=None,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        use_cache=False
                    )
                else:
                    # GPT-2 style models
                    block_output = self.transformer_blocks[ell](h_prev)
                
                h_ell = block_output[0] if isinstance(block_output, tuple) else block_output
                
                # Apply logit lens: final_ln + lm_head
                if self.final_ln is not None:
                    h_normed = self.final_ln(h_ell)
                else:
                    h_normed = h_ell
                
                logits_ell = self.lm_head(h_normed)
                
                # Cross-entropy loss for next-token prediction
                loss_ell = F.cross_entropy(
                    logits_ell.view(-1, logits_ell.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="mean"
                )
                
                per_layer_losses.append(loss_ell)
                h = h_ell  # Update hidden state for next layer
            
            # Single backward pass for all layers
            model.zero_grad(set_to_none=True)
            total_loss = torch.stack(per_layer_losses).sum()
            total_loss.backward()
            
            # Accumulate squared gradient norms for each block
            with torch.no_grad():
                for ell in range(self.num_layers):
                    grad_norm_sq = torch.tensor(0.0, device=self.device)
                    for param in self.transformer_blocks[ell].parameters():
                        if param.grad is not None:
                            grad_norm_sq += (param.grad.detach() ** 2).sum()
                    layer_grad_sums[ell] += grad_norm_sq
        
        # Compute mean over batches
        mean_layer_grad_sq = (layer_grad_sums / max(1, num_batches)).detach().cpu().numpy()
        
        results = {
            'layer_grad_norms': mean_layer_grad_sq,
            'total_strain': float(mean_layer_grad_sq.sum()),
            'num_batches': num_batches,
            'layer_indices': np.arange(1, self.num_layers + 1)
        }
        
        print(f"Parameter strain calculation complete. Total strain: {results['total_strain']:.6e}")
        return results
    
    def calculate_fisher_rao_length(self, dataloader: DataLoader) -> Dict[str, Union[np.ndarray, float]]:
        """
        Calculate Fisher-Rao thermodynamic length between layer predictions.
        
        Computes exact Fisher-Rao distances between probability distributions
        from consecutive layers using the logit lens approach.
        
        Args:
            dataloader: DataLoader with batched text data
            
        Returns:
            Dictionary containing:
            - 'step_lengths': Mean FR distances between consecutive layers
            - 'total_length': Sum of all step lengths
            - 'num_steps': Number of inter-layer steps
        """
        if not self.model_handler.is_causal:
            raise ValueError("Fisher-Rao calculation requires causal language model")
        
        print(f"Computing Fisher-Rao thermodynamic length across {self.num_layers-1} inter-layer steps...")
        
        num_steps = self.num_layers - 1
        fr_step_sums = torch.zeros(num_steps, device=self.device)
        fr_step_counts = torch.zeros(num_steps, device=self.device)
        
        model = self.model_handler.model
        transformer = getattr(model, 'transformer', getattr(model, 'model', model))
        
        @torch.no_grad()
        def fisher_rao_distance(logp_prev: torch.Tensor, 
                               logp_next: torch.Tensor, 
                               valid_mask: torch.Tensor) -> torch.Tensor:
            """
            Compute exact Fisher-Rao distance between two log-probability distributions.
            
            Args:
                logp_prev: Log probabilities from previous layer (B, S, V)
                logp_next: Log probabilities from next layer (B, S, V)  
                valid_mask: Valid positions mask (B, S)
                
            Returns:
                FR distances at each position (B, S)
            """
            # Bhattacharyya coefficient via log-sum-exp
            s = 0.5 * (logp_prev + logp_next)  # (B, S, V)
            log_bc = torch.logsumexp(s, dim=-1)  # (B, S)
            bc = torch.exp(log_bc).clamp_(0.0, 1.0)  # Numerical stability
            
            # Exact Fisher-Rao distance: 2 * arccos(BC)
            fr_dist = 2.0 * torch.acos(bc)  # (B, S)
            
            if valid_mask is not None:
                fr_dist = fr_dist.masked_fill(~valid_mask, 0.0)
            
            return fr_dist
        
        for batch in tqdm(dataloader, desc="Computing FR distances"):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            B, S = input_ids.shape
            
            # Build initial embeddings
            # Disable autocast temporarily to debug
            # with torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu",
            #                    dtype=self.amp_dtype, enabled=(self.device == "cuda")):
            wte, wpe = self.model_handler.get_embeddings()
            h = wte(input_ids)
            
            if wpe is not None:
                pos_ids = torch.arange(S, dtype=torch.long, device=self.device).unsqueeze(0).expand(B, S)
                h = h + wpe(pos_ids)
            
            if hasattr(transformer, 'drop'):
                h = transformer.drop(h)
            
            logp_prev = None
            
            for ell in range(self.num_layers):
                # Forward through current layer with proper arguments for different architectures
                if self.model_handler.architecture in ['llama', 'phi']:
                    # Llama/Phi models need position_embeddings (RoPE)
                    position_ids = torch.arange(S, dtype=torch.long, device=self.device).unsqueeze(0).expand(B, S)
                    
                    # Get RoPE embeddings from the model
                    if hasattr(self.model_handler.model.model, 'rotary_emb'):
                        rotary_emb = self.model_handler.model.model.rotary_emb
                        cos, sin = rotary_emb(h, position_ids)
                        position_embeddings = (cos, sin)
                    else:
                        position_embeddings = None
                    
                    block_output = self.transformer_blocks[ell](
                        h,
                        attention_mask=None,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        use_cache=False
                    )
                else:
                    # GPT-2 style models
                    block_output = self.transformer_blocks[ell](h)
                
                h = block_output[0] if isinstance(block_output, tuple) else block_output
                
                with torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu",
                                   dtype=self.amp_dtype, enabled=(self.device == "cuda")):
                    # Apply logit lens
                    if self.final_ln is not None:
                        h_normed = self.final_ln(h)
                    else:
                        h_normed = h
                    
                    logits = self.lm_head(h_normed)  # (B, S, V)
                    
                # Convert to log probabilities (in float32 for stability)
                logp = F.log_softmax(logits.float(), dim=-1)  # (B, S, V)
                
                if logp_prev is not None:
                    # Valid positions (where we have labels != -100)
                    valid_mask = (labels != -100)  # (B, S)
                    
                    # Compute Fisher-Rao step length
                    fr_distances = fisher_rao_distance(logp_prev, logp, valid_mask)  # (B, S)
                    
                    # Accumulate for this inter-layer step
                    step_idx = ell - 1  # Between layer ell-1 and ell
                    fr_step_sums[step_idx] += fr_distances.sum()
                    fr_step_counts[step_idx] += valid_mask.sum()
                
                logp_prev = logp
        
        # Compute mean FR distances per step
        fr_step_means = (fr_step_sums / fr_step_counts.clamp_min(1)).detach().cpu().numpy()
        
        results = {
            'step_lengths': fr_step_means,
            'total_length': float(fr_step_means.sum()),
            'num_steps': num_steps,
            'step_indices': np.arange(1, num_steps + 1)
        }
        
        print(f"Fisher-Rao calculation complete. Total length: {results['total_length']:.6f} radians")
        return results
    
    def calculate_semantic_efficiency(self, 
                                    param_strain_results: Dict, 
                                    fisher_rao_results: Dict,
                                    eps_efficiency: float = 1e-9) -> Dict[str, Union[np.ndarray, float]]:
        """
        Calculate semantic efficiency by combining parameter strain and Fisher-Rao metrics.
        
        Efficiency = log(1 + normalized_belief_change / (normalized_param_strain + eps))
        
        Args:
            param_strain_results: Results from calculate_parameter_strain
            fisher_rao_results: Results from calculate_fisher_rao_length  
            eps_efficiency: Small epsilon for numerical stability
            
        Returns:
            Dictionary with semantic efficiency metrics
        """
        param_strain = param_strain_results['layer_grad_norms'][:fisher_rao_results['num_steps']]  # Match lengths
        belief_change = fisher_rao_results['step_lengths']
        
        # Normalize both metrics to [0, 1]
        def min_max_normalize(x):
            x = np.array(x)
            return (x - x.min()) / (x.max() - x.min() + self.eps)
        
        norm_param_strain = min_max_normalize(param_strain)
        norm_belief_change = min_max_normalize(belief_change)
        
        # Calculate semantic efficiency
        semantic_efficiency = np.log(1 + norm_belief_change / (norm_param_strain + eps_efficiency))
        
        results = {
            'semantic_efficiency': semantic_efficiency,
            'normalized_param_strain': norm_param_strain,
            'normalized_belief_change': norm_belief_change,
            'mean_efficiency': float(semantic_efficiency.mean()),
            'layer_indices': np.arange(1, len(semantic_efficiency) + 1)
        }
        
        print(f"Semantic efficiency calculated. Mean efficiency: {results['mean_efficiency']:.4f}")
        return results
    
    def describe_results(self, 
                        param_strain_results: Optional[Dict] = None,
                        fisher_rao_results: Optional[Dict] = None, 
                        semantic_efficiency_results: Optional[Dict] = None) -> Dict[str, str]:
        """
        Generate detailed descriptions of thermodynamic length results.
        
        Args:
            param_strain_results: Parameter strain results
            fisher_rao_results: Fisher-Rao results
            semantic_efficiency_results: Semantic efficiency results
            
        Returns:
            Dictionary with detailed descriptions and interpretation guides
        """
        descriptions = {}
        
        if param_strain_results is not None:
            descriptions["parameter_strain"] = {
                "description": "Measures the effort (gradient magnitude) each layer exerts during next-token prediction",
                "formula": "E_ℓ = ||∇_θ_ℓ L||² (mean squared gradient norm per layer)",
                "interpretation": {
                    "high_values": "Layer is working hard to transform representations", 
                    "low_values": "Layer may be operating in saturation or be redundant",
                    "typical_patterns": "Often decreases in middle layers, increases near output"
                },
                "data_format": {
                    "layer_grad_norms": "Array of gradient norms per layer (length = num_layers)",
                    "total_strain": "Sum of all layer strains (scalar)",
                    "layer_indices": "Layer numbers (1-indexed)",
                    "num_batches": "Number of batches processed"
                }
            }
            
        if fisher_rao_results is not None:
            descriptions["fisher_rao"] = {
                "description": "Measures the geometric distance between probability distributions of consecutive layers",
                "formula": "d_FR(p,q) = 2 * arccos(∑ᵢ √(pᵢ * qᵢ)) (Fisher-Rao distance in radians)",
                "interpretation": {
                    "high_values": "Significant changes in next-token predictions between layers",
                    "low_values": "Minimal changes, layers may be similar in function", 
                    "typical_patterns": "Often high in early layers, stabilizes in later layers"
                },
                "data_format": {
                    "step_lengths": "Array of FR distances between consecutive layers (length = num_layers-1)",
                    "total_length": "Sum of all step lengths (scalar)",
                    "step_indices": "Inter-layer step numbers (1-indexed)",
                    "num_steps": "Number of inter-layer transitions"
                }
            }
            
        if semantic_efficiency_results is not None:
            descriptions["semantic_efficiency"] = {
                "description": "Combines parameter strain and belief change into unified efficiency metric", 
                "formula": "L_eff(ℓ) = log(1 + norm(Δp_ℓ) / (norm(E_ℓ) + ε))",
                "interpretation": {
                    "high_values": "Layer efficiently converts computational effort into meaningful output changes",
                    "low_values": "Layer is either not working hard or changes are not reflected in outputs",
                    "optimal_range": "Values typically range from 0-20, with 5-15 being efficient"
                },
                "data_format": {
                    "semantic_efficiency": "Array of efficiency scores per layer",
                    "normalized_param_strain": "Min-max normalized parameter strain values",
                    "normalized_belief_change": "Min-max normalized Fisher-Rao distances", 
                    "mean_efficiency": "Average efficiency across all layers",
                    "layer_indices": "Layer numbers corresponding to efficiency scores"
                }
            }
            
        return descriptions
