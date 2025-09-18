"""
High-level API for easy calculation of all metrics.

Provides convenient functions and classes that combine thermodynamic length
and spectral curvature calculations with minimal setup.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

from .utils import ModelHandler, DatasetHandler
from .thermodynamic import ThermodynamicCalculator
from .spectral import SpectralCalculator


class nDNA:
    """
    High-level interface for calculating all nDNA metrics.
    
    This class provides a simple way to calculate both thermodynamic length
    and spectral curvature metrics for any HuggingFace model and dataset.
    """
    
    def __init__(self, 
                 model_name: str,
                 device: str = "auto",
                 max_samples: Optional[int] = 200,
                 batch_size: int = 4,
                 max_length: int = 512):
        """
        Initialize LLM metrics calculator.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or 'auto')
            max_samples: Maximum samples to process from dataset
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Initialize model handler
        print(f"Initializing model handler for: {model_name}")
        self.model_handler = ModelHandler(model_name, device=device)
        
        # Initialize calculators
        self.thermo_calc = ThermodynamicCalculator(self.model_handler)
        self.spectral_calc = SpectralCalculator(self.model_handler)
        
        # Initialize dataset handler
        self.dataset_handler = DatasetHandler(
            self.model_handler.tokenizer,
            max_length=max_length,
            batch_size=batch_size
        )
        
        print(f"nDNA initialized for {model_name}")
        print(f"Architecture: {self.model_handler.architecture}")
        print(f"Device: {self.model_handler.device}")
        print(f"Is causal: {self.model_handler.is_causal}")
    
    def calculate_for_dataset(self, 
                            dataset_name: str,
                            split: str = "train",
                            text_column: str = "text",
                            calculate_thermodynamic: bool = True,
                            calculate_spectral: bool = True,
                            sample_texts_for_spectral: int = 10,
                            config_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate metrics for a HuggingFace dataset.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to use
            text_column: Column containing text data
            calculate_thermodynamic: Whether to calculate thermodynamic metrics
            calculate_spectral: Whether to calculate spectral curvature
            sample_texts_for_spectral: Number of texts to sample for spectral analysis
            config_name: Optional config name for datasets with multiple configs
            
        Returns:
            Dictionary containing all calculated metrics
        """
        print(f"\n=== Processing dataset: {dataset_name} ===")
        
        results = {"model_name": self.model_name, "dataset_name": dataset_name}
        
        # Load dataset
        try:
            self.dataset_handler.load_from_huggingface(
                dataset_name, split, text_column, self.max_samples, 
                text_processor=None, config_name=config_name
            )
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            print("Falling back to sample prompts...")
            return self.calculate_for_prompts()
        
        # Calculate thermodynamic metrics (requires dataloader)
        if calculate_thermodynamic and self.model_handler.is_causal:
            print("\n--- Calculating Thermodynamic Metrics ---")
            dataloader = self.dataset_handler.create_dataloader(
                shuffle=False, for_causal_lm=True
            )
            
            try:
                # Parameter strain
                param_strain = self.thermo_calc.calculate_parameter_strain(dataloader)
                results["parameter_strain"] = param_strain
                
                # Fisher-Rao length
                fisher_rao = self.thermo_calc.calculate_fisher_rao_length(dataloader)
                results["fisher_rao"] = fisher_rao
                
                # Semantic efficiency
                semantic_eff = self.thermo_calc.calculate_semantic_efficiency(
                    param_strain, fisher_rao
                )
                results["semantic_efficiency"] = semantic_eff
                
            except Exception as e:
                warnings.warn(f"Failed to calculate thermodynamic metrics: {e}")
                results["thermodynamic_error"] = str(e)
        
        # Calculate spectral curvature (sample texts)
        if calculate_spectral:
            print("\n--- Calculating Spectral Curvature ---")
            sample_texts = self.dataset_handler.get_sample_texts(sample_texts_for_spectral)
            
            try:
                if len(sample_texts) > 1:
                    spectral_results = self.spectral_calc.calculate_curvature_for_texts(sample_texts)
                else:
                    spectral_results = self.spectral_calc.calculate_curvature_for_text(sample_texts[0])
                
                results["spectral_curvature"] = spectral_results
                
            except Exception as e:
                warnings.warn(f"Failed to calculate spectral curvature: {e}")
                results["spectral_error"] = str(e)
        
        return results
    
    def calculate_for_texts(self, 
                          texts: List[str],
                          calculate_thermodynamic: bool = True, 
                          calculate_spectral: bool = True) -> Dict[str, Any]:
        """
        Calculate metrics for a list of text strings.
        
        Args:
            texts: List of text strings to analyze
            calculate_thermodynamic: Whether to calculate thermodynamic metrics
            calculate_spectral: Whether to calculate spectral curvature
            
        Returns:
            Dictionary containing all calculated metrics
        """
        print(f"\n=== Processing {len(texts)} custom texts ===")
        
        results = {"model_name": self.model_name, "dataset_name": "custom_texts"}
        
        # Load texts
        self.dataset_handler.load_from_texts(texts)
        
        # Calculate thermodynamic metrics
        if calculate_thermodynamic and self.model_handler.is_causal:
            print("\n--- Calculating Thermodynamic Metrics ---")
            dataloader = self.dataset_handler.create_dataloader(
                shuffle=False, for_causal_lm=True
            )
            
            try:
                param_strain = self.thermo_calc.calculate_parameter_strain(dataloader)
                fisher_rao = self.thermo_calc.calculate_fisher_rao_length(dataloader)
                semantic_eff = self.thermo_calc.calculate_semantic_efficiency(
                    param_strain, fisher_rao
                )
                
                results.update({
                    "parameter_strain": param_strain,
                    "fisher_rao": fisher_rao,
                    "semantic_efficiency": semantic_eff
                })
                
            except Exception as e:
                warnings.warn(f"Failed to calculate thermodynamic metrics: {e}")
                results["thermodynamic_error"] = str(e)
        
        # Calculate spectral curvature
        if calculate_spectral:
            print("\n--- Calculating Spectral Curvature ---")
            
            try:
                if len(texts) > 1:
                    spectral_results = self.spectral_calc.calculate_curvature_for_texts(texts)
                else:
                    spectral_results = self.spectral_calc.calculate_curvature_for_text(texts[0])
                
                results["spectral_curvature"] = spectral_results
                
            except Exception as e:
                warnings.warn(f"Failed to calculate spectral curvature: {e}")
                results["spectral_error"] = str(e)
        
        return results
    
    def calculate_for_prompts(self, 
                            prompts: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
        """
        Calculate spectral curvature for sample prompts of different types.
        
        Args:
            prompts: List of (name, text) tuples. If None, uses default sample prompts.
            
        Returns:
            Dictionary containing spectral curvature results for each prompt
        """
        if prompts is None:
            prompts = self.spectral_calc.get_sample_prompts()
        
        print(f"\n=== Processing {len(prompts)} sample prompts ===")
        
        results = {
            "model_name": self.model_name,
            "dataset_name": "sample_prompts",
            "spectral_curvature": self.spectral_calc.calculate_curvature_for_prompts(prompts)
        }
        
        return results


def calculate_all_metrics(model_name: str,
                         dataset_source: Union[str, List[str]],
                         max_samples: int = 200,
                         batch_size: int = 4,
                         device: str = "auto") -> Dict[str, Any]:
    """
    Convenience function to calculate all metrics for a model and dataset.
    
    Args:
        model_name: HuggingFace model identifier
        dataset_source: Either dataset name (str) or list of texts
        max_samples: Maximum samples to process
        batch_size: Batch size for processing
        device: Device to use
        
    Returns:
        Dictionary containing all calculated metrics
    """
    # Initialize metrics calculator
    calculator = nDNA(
        model_name=model_name,
        device=device,
        max_samples=max_samples,
        batch_size=batch_size
    )
    
    # Calculate metrics based on source type
    if isinstance(dataset_source, str):
        if dataset_source.lower() == "prompts":
            results = calculator.calculate_for_prompts()
        else:
            results = calculator.calculate_for_dataset(dataset_source)
    elif isinstance(dataset_source, list):
        results = calculator.calculate_for_texts(dataset_source)
    else:
        raise ValueError("dataset_source must be a string (dataset name) or list of texts")
    
    return results
