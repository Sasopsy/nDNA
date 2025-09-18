# nDNA Library

A comprehensive mini library for calculating **thermodynamic length** and **spectral curvature** metrics for any Large Language Model from Hugging Face transformers.

## 🚀 Overview

This library implements two key geometric and information-theoretic metrics for analyzing the internal dynamics of transformer language models:

1. **Thermodynamic Length**: Measures the "effort" required for information processing through the network
   - **Parameter Strain Method**: Based on gradient norms per layer  
   - **Fisher-Rao Method**: Based on prediction changes between layers using Fisher-Rao distance
   - **Semantic Efficiency**: Combines both methods into a unified efficiency metric

2. **Spectral Curvature**: Measures the geometric curvature of the probability simplex manifold
   - Uses logit lens to extract layer-wise probability distributions
   - Computes discrete curvature using square-root embeddings on the unit sphere
   - Provides insights into the smoothness of representation transitions

## 📦 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd nDNA

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- NumPy, tqdm, pandas
- CUDA (optional but recommended for larger models)

## 🎯 Quick Start

### Simple Usage

```python
from ndna import calculate_all_metrics

# Calculate all metrics for GPT-2 with custom texts
custom_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models process information through layers.",
    "The integral of x squared from 0 to 1 equals one third."
]

results = calculate_all_metrics(
    model_name="gpt2",
    dataset_source=custom_texts,
    max_samples=200,
    batch_size=4
)

print(f"Total thermodynamic length: {results['fisher_rao']['total_length']:.4f} radians")
print(f"Mean spectral curvature: {results['spectral_curvature']['overall_mean']:.6e}")
```

### Using HuggingFace Datasets

```python
from ndna import calculate_all_metrics
from ndna import datasets

# Check recommended datasets
recommended = datasets.get_recommended_datasets(domain="reviews")
print(recommended)

# Use a simple dataset (no config needed)
results = calculate_all_metrics(
    model_name="microsoft/DialoGPT-medium", 
    dataset_source="imdb",  # Movie reviews
    max_samples=200
)

# Or use a configured dataset
results = calculate_all_metrics(
    model_name="gpt2",
    dataset_source="wikitext",  # Automatically tries common configs
    max_samples=200
)
```

### FLAN 2021 + Phi-2 Example

```python
from ndna import nDNA

# Analyze FLAN 2021 summarization tasks with Phi-2
analyzer = nDNA(
    model_name="microsoft/phi-2",
    max_samples=30,
    batch_size=2
)

# Focus on Gigaword summarization subset
results = analyzer.calculate_for_dataset(
    dataset_name="DataProvenanceInitiative/flan2021_submix_original",
    calculate_thermodynamic=True,
    calculate_spectral=True
)

print(f"Thermodynamic Length: {results['fisher_rao']['total_length']:.4f} radians")
print(f"Spectral Curvature: {results['spectral_curvature']['overall_mean']:.6e}")
```

### Advanced Usage

```python
from ndna import ModelHandler, DatasetHandler, ThermodynamicCalculator, SpectralCalculator

# Initialize components
model_handler = ModelHandler("gpt2", device="cuda")
dataset_handler = DatasetHandler(model_handler.tokenizer, batch_size=8)

# Load your dataset
dataset_handler.load_from_huggingface("wikitext", max_samples=500)
dataloader = dataset_handler.create_dataloader(for_causal_lm=True)

# Calculate thermodynamic metrics
thermo_calc = ThermodynamicCalculator(model_handler)
param_strain = thermo_calc.calculate_parameter_strain(dataloader)
fisher_rao = thermo_calc.calculate_fisher_rao_length(dataloader) 
semantic_eff = thermo_calc.calculate_semantic_efficiency(param_strain, fisher_rao)

# Calculate spectral curvature
spectral_calc = SpectralCalculator(model_handler)
sample_texts = dataset_handler.get_sample_texts(20)
curvature_results = spectral_calc.calculate_curvature_for_texts(sample_texts)

# Get detailed result descriptions
thermo_descriptions = thermo_calc.describe_results(param_strain, fisher_rao, semantic_eff)
spectral_descriptions = spectral_calc.describe_results(curvature_results)
```

## 🏗️ Architecture

The library is organized into several key modules:

```
ndna/
├── __init__.py           # Main API exports
├── api.py               # High-level convenience functions  
├── thermodynamic.py     # Thermodynamic length calculations
├── spectral.py          # Spectral curvature calculations
└── utils/
    ├── model_handler.py     # Unified model interface
    └── dataset_handler.py   # Flexible dataset processing
```

### Supported Model Architectures

- **GPT-2 / DialoGPT**: Full support for all metrics
- **LLaMA**: Full support for all metrics  
- **BERT / RoBERTa**: Spectral curvature only (no causal LM head)
- **T5**: Partial support (encoder only)
- **Generic**: Automatic detection for other architectures

### Supported Datasets

**Simple Datasets (no config needed):**
- **imdb**: Movie reviews with sentiment
- **ag_news**: News article categorization  
- **yelp_review_full**: Restaurant reviews
- **amazon_polarity**: Product reviews

**Configured Datasets (auto-handled):**
- **wikitext**: Wikipedia text (auto-tries wikitext-2-raw-v1)
- **glue**: NLP benchmark tasks (auto-tries sst2)

**Custom Data:**
- **Custom Texts**: Direct list of strings
- **File Formats**: TXT, JSON, JSONL, CSV

**Dataset Utilities:**
```python
from ndna import datasets

# Get recommendations
recommended = datasets.get_recommended_datasets(size="medium")
info = datasets.get_dataset_info("imdb")
```

## 📊 Metrics Explained

### Thermodynamic Length

**Concept**: Measures the "distance" traveled in parameter space or prediction space as information flows through the network layers.

1. **Parameter Strain (E_ℓ)**
   ```
   E_ℓ = ||∇_θ_ℓ L||²
   ```
   - Computed via observed Fisher information
   - Measures gradient magnitude per layer during next-token prediction
   - Higher values indicate layers working "harder"

2. **Fisher-Rao Distance (Δp_ℓ)**
   ```
   d_FR(p,q) = 2 * arccos(∑ᵢ √(pᵢ * qᵢ))
   ```
   - Measures distance between consecutive layer predictions 
   - Uses logit lens to extract layer-wise distributions
   - Natural metric on probability simplex

3. **Semantic Efficiency (L_eff)**
   ```
   L_eff(ℓ) = log(1 + norm(Δp_ℓ) / (norm(E_ℓ) + ε))
   ```
   - Combines belief change and parameter strain
   - Higher values indicate efficient information processing

### Spectral Curvature

**Concept**: Measures the geometric curvature of the probability simplex manifold as representations evolve through layers.

1. **Square-root Embedding**
   ```
   u = √p / ||√p||
   ```
   - Maps probability distributions to unit sphere
   - Enables Riemannian geometry calculations

2. **Discrete Curvature (κ_simp)**
   ```
   κ = ||Δ²u|| / ||Δu||^(3/2)
   ```
   - Computed using finite differences on tangent space
   - Higher values indicate rapid changes in representation geometry
   - Reveals layer specialization and processing dynamics

## 📈 Interpretation Guide

### Thermodynamic Length Results

- **High Parameter Strain**: Layer is working hard to transform representations
- **High Fisher-Rao Distance**: Significant changes in next-token predictions  
- **High Semantic Efficiency**: Layer efficiently converts effort into useful changes
- **Low values**: Layers may be redundant or operating in saturation

### Spectral Curvature Results

- **High Curvature**: Rapid changes in representational geometry
- **Low Curvature**: Smooth, gradual transformations
- **Curvature Peaks**: Often correspond to critical processing layers
- **Language-dependent patterns**: Different text types show different curvature profiles

## 🧪 Examples

### Example 1: Model Comparison

```python
models = ["gpt2", "distilgpt2", "microsoft/DialoGPT-medium"]
results = {}

custom_texts = ["The cat sat on the mat.", "Neural networks are computational models."]

for model in models:
    results[model] = calculate_all_metrics(
        model_name=model,
        dataset_source=custom_texts, 
        max_samples=100
    )
    
# Compare semantic efficiency
for model, result in results.items():
    if "semantic_efficiency" in result:
        mean_eff = result["semantic_efficiency"]["mean_efficiency"]
        print(f"{model}: {mean_eff:.4f}")
```

### Example 2: Text Type Analysis

```python
from ndna import nDNA

calculator = nDNA("gpt2")

# Different text types
prompts = [
    ("Simple English", "The cat sat on the mat."),
    ("Technical", "The mitochondrion is the powerhouse of the cell."), 
    ("Mathematical", "The integral of x squared equals x cubed over 3."),
    ("Programming", "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)")
]

results = calculator.calculate_for_prompts(prompts)

# Analyze curvature differences
for name, result in results["spectral_curvature"].items():
    if result:
        print(f"{name}: {result['mean_curvature']:.6e}")
```

## 🔧 Validation and Testing

Run the validation suite:

```bash
python test_validation.py
```

Run examples:

```bash
python examples/basic_usage.py
python examples/detailed_usage.py
```

## 📚 References

This library implements methods from:

1. **Information Geometry**: Fisher-Rao metrics on probability manifolds
2. **Thermodynamic Computing**: Parameter strain and information-theoretic measures  
3. **Differential Geometry**: Curvature analysis on Riemannian manifolds
4. **Interpretability Research**: Logit lens and layer-wise analysis techniques

## ⚠️ Limitations

1. **Computational Cost**: Thermodynamic calculations require gradient computation and can be memory-intensive
2. **Model Support**: Some architectures may require manual adaptation
3. **Batch Dependencies**: Fisher-Rao calculations benefit from larger batch sizes for stability
4. **Causality Requirement**: Thermodynamic metrics only work with causal language models

## 🛠️ Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Run validation tests
5. Submit a pull request

## 📄 License

This project is released under MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

Based on the original research notebooks implementing thermodynamic length and spectral curvature analysis for transformer language models. Special thanks to the nDNA research team for the foundational mathematical framework.

---

**Quick Links:**
- [Basic Usage Example](examples/basic_usage.py)
- [Detailed Usage Example](examples/detailed_usage.py)  
- [API Documentation](#-architecture)
- [Validation Tests](test_validation.py)