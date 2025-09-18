"""
Dataset handler for processing various dataset formats and sources.
Supports HuggingFace datasets, custom datasets, and text prompts.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Dataset as HFDataset
from typing import List, Dict, Union, Optional, Callable, Any
import pandas as pd
import json


class TextDataset(Dataset):
    """Simple dataset wrapper for text data."""
    
    def __init__(self, texts: List[str]):
        self.texts = texts
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {"text": self.texts[idx]}


class DatasetHandler:
    """
    Flexible dataset handler that can process various data sources and formats.
    
    Recommended datasets (no config required):
    - "imdb": Movie reviews  
    - "ag_news": News categorization
    - "yelp_review_full": Restaurant reviews
    - "amazon_polarity": Amazon product reviews
    
    Datasets requiring configs:
    - "wikitext": Use config='wikitext-2-raw-v1' or 'wikitext-103-raw-v1'
    - "glue": Use config='sst2', 'cola', 'mrpc', etc.
    - "super_glue": Use config='boolq', 'cb', 'copa', etc.
    """
    
    def __init__(self, 
                 tokenizer,
                 max_length: int = 512,
                 batch_size: int = 4,
                 num_workers: int = 2):
        """
        Initialize dataset handler.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = None
        self.dataloader = None
    
    def load_from_huggingface(self, 
                              dataset_name: str, 
                              split: str = "train",
                              text_column: str = "text",
                              max_samples: Optional[int] = None,
                              text_processor: Optional[Callable] = None,
                              config_name: Optional[str] = None) -> 'DatasetHandler':
        """
        Load dataset from HuggingFace datasets.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to use
            text_column: Column name containing text data
            max_samples: Maximum number of samples to use
            text_processor: Optional function to process text examples
            config_name: Optional config name for datasets with multiple configs
            
        Returns:
            Self for method chaining
        """
        print(f"Loading dataset: {dataset_name} (split: {split})")
        
        # Load dataset with better error handling
        try:
            if config_name:
                ds = load_dataset(dataset_name, config_name, split=split)
            else:
                ds = load_dataset(dataset_name, split=split)
        except Exception as e:
            # If dataset loading fails, try some common fallbacks
            if "Config name is missing" in str(e) or "available configs" in str(e):
                print(f"Dataset requires config. Trying common configs...")
                
                # Try some common configs based on dataset name
                common_configs = {
                    'wikitext': ['wikitext-2-raw-v1', 'wikitext-103-raw-v1'],
                    'glue': ['sst2', 'cola', 'mrpc'],
                    'super_glue': ['boolq', 'cb', 'copa'],
                }
                
                if dataset_name in common_configs:
                    for config in common_configs[dataset_name]:
                        try:
                            print(f"  Trying config: {config}")
                            ds = load_dataset(dataset_name, config, split=split)
                            print(f"  ✓ Successfully loaded with config: {config}")
                            break
                        except Exception:
                            continue
                    else:
                        raise ValueError(f"Could not load dataset {dataset_name}. "
                                       f"Please specify a config_name parameter. "
                                       f"Error: {e}")
                else:
                    raise ValueError(f"Dataset {dataset_name} requires a config_name. "
                                   f"Please check the dataset documentation. Error: {e}")
            else:
                raise e
        
        # Apply text processor if provided
        if text_processor:
            ds = ds.map(text_processor, remove_columns=ds.column_names)
            text_column = "text"  # Assume processor outputs 'text' column
        
        # Extract text data
        if text_column not in ds.column_names:
            raise ValueError(f"Column '{text_column}' not found in dataset. Available: {ds.column_names}")
        
        texts = ds[text_column]
        
        # Filter short texts
        texts = [text for text in texts if len(text.strip()) > 10]
        
        # Limit samples if specified
        if max_samples and len(texts) > max_samples:
            texts = texts[:max_samples]
        
        print(f"Dataset loaded: {len(texts)} samples")
        
        self.dataset = TextDataset(texts)
        return self
    
    
    def load_from_texts(self, texts: List[str]) -> 'DatasetHandler':
        """
        Load from list of text strings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Self for method chaining
        """
        # Filter short texts
        texts = [text for text in texts if len(text.strip()) > 10]
        print(f"Loaded {len(texts)} text samples")
        
        self.dataset = TextDataset(texts)
        return self
    
    def load_from_file(self, 
                       file_path: str, 
                       file_format: str = "txt",
                       text_column: Optional[str] = None) -> 'DatasetHandler':
        """
        Load dataset from file.
        
        Args:
            file_path: Path to data file
            file_format: File format ('txt', 'json', 'csv', 'jsonl')
            text_column: Column name for CSV/JSON files
            
        Returns:
            Self for method chaining
        """
        if file_format == "txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        
        elif file_format == "json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    if text_column:
                        texts = [item[text_column] for item in data]
                    else:
                        texts = [str(item) for item in data]
                else:
                    raise ValueError("JSON file should contain a list of items")
        
        elif file_format == "jsonl":
            texts = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    if text_column:
                        texts.append(item[text_column])
                    else:
                        texts.append(str(item))
        
        elif file_format == "csv":
            df = pd.read_csv(file_path)
            if text_column and text_column in df.columns:
                texts = df[text_column].astype(str).tolist()
            else:
                raise ValueError(f"Column '{text_column}' not found in CSV. Available: {df.columns.tolist()}")
        
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        return self.load_from_texts(texts)
    
    def create_dataloader(self, 
                          shuffle: bool = False,
                          for_causal_lm: bool = True) -> DataLoader:
        """
        Create PyTorch DataLoader with appropriate collate function.
        
        Args:
            shuffle: Whether to shuffle data
            for_causal_lm: Whether to prepare data for causal language modeling
            
        Returns:
            PyTorch DataLoader
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Use one of the load_* methods first.")
        
        if for_causal_lm:
            collate_fn = self._causal_collate
        else:
            collate_fn = self._simple_collate
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return self.dataloader
    
    def _simple_collate(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Simple collate function for classification/encoding tasks."""
        texts = [item["text"] for item in batch]
        
        encoded = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return encoded
    
    def _causal_collate(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for causal language modeling with teacher forcing."""
        texts = [item["text"] for item in batch]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding="longest", 
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        
        # Teacher forcing: predict next token
        # Input: tokens[:-1], Labels: tokens[1:]
        input_ids_shifted = input_ids[:, :-1].contiguous()
        attention_shifted = attention_mask[:, :-1].contiguous() 
        labels_shifted = input_ids[:, 1:].contiguous()
        
        # Mask padding tokens in labels (-100 = ignore in loss)
        labels_shifted = labels_shifted.masked_fill(attention_mask[:, 1:] == 0, -100)
        
        return {
            "input_ids": input_ids_shifted,
            "labels": labels_shifted,
            "attention_mask": attention_shifted,
        }
    
    def get_sample_texts(self, n: int = 5) -> List[str]:
        """Get sample texts from the dataset."""
        if self.dataset is None:
            return []
        
        samples = []
        for i in range(min(n, len(self.dataset))):
            samples.append(self.dataset[i]["text"])
        return samples
