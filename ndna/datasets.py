"""
Recommended datasets for nDNA analysis.

This module provides lists of datasets that work well with the nDNA library,
organized by type and complexity.
"""

# Datasets that work without additional configuration
SIMPLE_DATASETS = {
    "imdb": {
        "description": "Movie reviews with binary sentiment",
        "text_column": "text",
        "size": "large",
        "domain": "entertainment"
    },
    "ag_news": {
        "description": "News categorization dataset", 
        "text_column": "text",
        "size": "medium",
        "domain": "news"
    },
    "yelp_review_full": {
        "description": "Restaurant reviews with ratings",
        "text_column": "text", 
        "size": "large",
        "domain": "reviews"
    },
    "amazon_polarity": {
        "description": "Amazon product reviews with sentiment",
        "text_column": "content",
        "size": "large", 
        "domain": "ecommerce"
    }
}

# Datasets requiring specific configs
CONFIGURED_DATASETS = {
    "wikitext": {
        "configs": {
            "wikitext-2-raw-v1": "Smaller Wikipedia text, raw version",
            "wikitext-2-v1": "Smaller Wikipedia text, processed version",
            "wikitext-103-raw-v1": "Larger Wikipedia text, raw version", 
            "wikitext-103-v1": "Larger Wikipedia text, processed version"
        },
        "text_column": "text",
        "domain": "encyclopedia",
        "recommended_config": "wikitext-2-raw-v1"
    },
    "glue": {
        "configs": {
            "sst2": "Stanford Sentiment Treebank",
            "cola": "Corpus of Linguistic Acceptability", 
            "mrpc": "Microsoft Research Paraphrase Corpus",
            "qqp": "Quora Question Pairs",
            "mnli": "Multi-Genre Natural Language Inference"
        },
        "text_column": "sentence",  # varies by config
        "domain": "nlp_benchmarks",
        "recommended_config": "sst2"
    }
}

def get_recommended_datasets(domain: str = None, size: str = None):
    """
    Get recommended datasets based on domain and size preferences.
    
    Args:
        domain: Filter by domain ('entertainment', 'news', 'reviews', etc.)
        size: Filter by size ('small', 'medium', 'large')
        
    Returns:
        Dictionary of recommended datasets
    """
    recommendations = {}
    
    # Filter simple datasets
    for name, info in SIMPLE_DATASETS.items():
        if domain and info['domain'] != domain:
            continue
        if size and info['size'] != size:
            continue
        recommendations[name] = info
    
    # Add configured datasets with their recommended configs
    for name, info in CONFIGURED_DATASETS.items():
        if domain and info['domain'] != domain:
            continue
        
        config_info = info.copy()
        config_info['config_name'] = info['recommended_config']
        config_info['description'] = info['configs'][info['recommended_config']]
        recommendations[name] = config_info
    
    return recommendations

def get_dataset_info(dataset_name: str):
    """
    Get information about a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with dataset information or None if not found
    """
    if dataset_name in SIMPLE_DATASETS:
        return SIMPLE_DATASETS[dataset_name]
    elif dataset_name in CONFIGURED_DATASETS:
        return CONFIGURED_DATASETS[dataset_name]
    else:
        return None
