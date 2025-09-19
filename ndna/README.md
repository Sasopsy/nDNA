# nDNA Output Dictionary Reference

This document provides a comprehensive explanation of all dictionary keys returned by the nDNA thermodynamic and spectral calculations, based on the actual implementation in `thermodynamic.py` and `spectral.py`.

## Thermodynamic Length Calculator Outputs

The thermodynamic calculator implements three main methods that return different sets of dictionary keys:

### 1. Parameter Strain Results (`calculate_parameter_strain`)

**Dictionary Keys:**
```python
{
    'layer_grad_norms': np.ndarray,      # Per-layer gradient norms
    'total_strain': float,               # Sum of all layer strains  
    'num_batches': int,                  # Number of batches processed
    'layer_indices': np.ndarray          # Layer numbers (1-indexed)
}
```

**Detailed Explanation:**

- **`layer_grad_norms`**: Array of mean squared gradient norms for each transformer layer
  - **Shape**: `(num_layers,)` where `num_layers` is the total number of transformer blocks
  - **Calculation**: `||∇_θ_ℓ L||²` - squared L2 norm of gradients for parameters in layer ℓ
  - **Interpretation**: Measures how much "effort" each layer exerts during next-token prediction
  - **Units**: Squared gradient magnitude (dimensionless)
  - **Typical Range**: 10⁰ to 10⁴, with early layers often showing higher values

- **`total_strain`**: Scalar sum of all layer gradient norms
  - **Calculation**: `sum(layer_grad_norms)`
  - **Interpretation**: Overall computational effort across the entire model
  - **Usage**: Model-level comparison metric for different inputs or architectures

- **`num_batches`**: Number of data batches processed during calculation
  - **Purpose**: Indicates the statistical reliability of the gradient norm estimates
  - **Note**: Gradient norms are averaged across all processed batches

- **`layer_indices`**: Array of layer numbers for indexing
  - **Values**: `[1, 2, 3, ..., num_layers]` (1-indexed for human readability)
  - **Purpose**: Corresponds to the indices in `layer_grad_norms`

### 2. Fisher-Rao Distance Results (`calculate_fisher_rao_length`)

**Dictionary Keys:**
```python
{
    'step_lengths': np.ndarray,          # FR distances between consecutive layers
    'total_length': float,               # Sum of all step lengths
    'num_steps': int,                    # Number of inter-layer transitions
    'step_indices': np.ndarray           # Step numbers (1-indexed)
}
```

**Detailed Explanation:**

- **`step_lengths`**: Array of Fisher-Rao distances between consecutive layer predictions
  - **Shape**: `(num_layers-1,)` - one distance for each inter-layer transition
  - **Calculation**: `d_FR(p_ℓ, p_ℓ₊₁) = 2 * arccos(∑ᵢ √(pᵢ * qᵢ))` where p, q are probability distributions
  - **Units**: Radians (geometric distance on probability simplex)
  - **Range**: [0, π] radians, where π represents maximum possible distance
  - **Interpretation**: How much the model's next-token predictions change between layers

- **`total_length`**: Scalar sum of all Fisher-Rao step lengths
  - **Calculation**: `sum(step_lengths)`
  - **Units**: Radians
  - **Interpretation**: Total "thermodynamic length" - cumulative belief change through the model
  - **Typical Range**: 5-50 radians for most language models

- **`num_steps`**: Number of inter-layer transitions
  - **Value**: Always `num_layers - 1`
  - **Purpose**: Validation and array dimension checking

- **`step_indices`**: Array of step numbers for indexing
  - **Values**: `[1, 2, 3, ..., num_steps]` (1-indexed)
  - **Interpretation**: Step i represents the transition from layer i to layer i+1

### 3. Semantic Efficiency Results (`calculate_semantic_efficiency`)

**Dictionary Keys:**
```python
{
    'semantic_efficiency': np.ndarray,           # Efficiency scores per layer
    'normalized_param_strain': np.ndarray,      # Min-max normalized parameter strain
    'normalized_belief_change': np.ndarray,     # Min-max normalized Fisher-Rao distances
    'mean_efficiency': float,                   # Average efficiency across layers
    'layer_indices': np.ndarray                 # Layer numbers for efficiency scores
}
```

**Detailed Explanation:**

- **`semantic_efficiency`**: Array of efficiency scores combining parameter strain and belief change
  - **Shape**: `(num_steps,)` - matches Fisher-Rao step lengths
  - **Calculation**: `log(1 + norm_belief_change / (norm_param_strain + ε))`
  - **Interpretation**: How efficiently each layer converts computational effort into meaningful output changes
  - **Range**: [0, ∞) with typical values 0-20
  - **High values**: Layer efficiently transforms representations with minimal computational cost
  - **Low values**: Layer works hard but produces little change in outputs

- **`normalized_param_strain`**: Min-max normalized parameter strain values
  - **Normalization**: `(x - min(x)) / (max(x) - min(x))`
  - **Range**: [0, 1]
  - **Purpose**: Enables fair comparison with normalized belief change

- **`normalized_belief_change`**: Min-max normalized Fisher-Rao distances
  - **Normalization**: Same as parameter strain
  - **Range**: [0, 1]
  - **Purpose**: Standardized measure of prediction changes for efficiency calculation

- **`mean_efficiency`**: Scalar average of all efficiency scores
  - **Calculation**: `mean(semantic_efficiency)`
  - **Usage**: Single number summary of model's overall semantic efficiency

- **`layer_indices`**: Array of layer numbers corresponding to efficiency scores
  - **Length**: Matches `semantic_efficiency` array length
  - **Note**: May be shorter than total layers due to Fisher-Rao step matching

## Spectral Curvature Calculator Outputs

The spectral calculator has different output formats depending on the method used:

### 1. Single Text Analysis (`calculate_curvature_for_text`)

**Dictionary Keys:**
```python
{
    'curvatures': List[float],           # Curvature values for interior layers
    'speeds': List[float],               # First-order difference magnitudes  
    'num_layers': int,                   # Total layers including embeddings
    'num_interior_points': int,          # Number of interior layers
    'layer_indices': List[int],          # Interior layer indices
    'mean_curvature': float,             # Average curvature
    'max_curvature': float,              # Maximum curvature value
    'min_curvature': float               # Minimum curvature value
}
```

**Detailed Explanation:**

- **`curvatures`**: List of discrete curvature values for interior layers only
  - **Length**: `num_layers - 2` (excludes first and last layers)
  - **Calculation**: `κ = ||Δ²u|| / ||Δu||^(3/2)` where u is square-root embedding
  - **Units**: Inverse length (geometric curvature on unit sphere)
  - **Range**: [0, ∞) with typical values 10⁻⁶ to 10²
  - **Interpretation**: Rate of change of direction in probability space

- **`speeds`**: List of first-order difference magnitudes
  - **Length**: `num_layers - 1` (between consecutive layers)
  - **Calculation**: `||Δu||` where Δu is tangent-projected difference
  - **Purpose**: Denominators in curvature calculation, indicate transformation speed

- **`num_layers`**: Total number of layers processed
  - **Includes**: Embedding layer + all transformer blocks
  - **Note**: Different from transformer-only counts in thermodynamic calculations

- **`num_interior_points`**: Number of layers with computable curvature
  - **Value**: Always `num_layers - 2`
  - **Reason**: Curvature requires second derivatives, so boundary layers are excluded

- **`layer_indices`**: List of interior layer numbers
  - **Values**: `[1, 2, ..., num_interior_points]`
  - **Purpose**: Maps curvature values to specific layers

- **`mean_curvature`**: Average curvature across all interior layers
  - **Calculation**: `mean(curvatures)`
  - **Usage**: Single-number summary of geometric complexity

- **`max_curvature`** / **`min_curvature`**: Extreme curvature values
  - **Purpose**: Identify layers with highest/lowest geometric activity

### 2. Multiple Text Analysis (`calculate_curvature_for_texts`)

**Dictionary Keys:**
```python
{
    'mean_curvatures': np.ndarray,       # Mean curvature per layer across texts
    'std_curvatures': np.ndarray,        # Standard deviation per layer
    'all_curvatures': List[List[float]], # Raw curvatures for each text
    'num_texts': int,                    # Number of texts processed
    'layer_indices': np.ndarray,         # Interior layer indices
    'overall_mean': float,               # Grand mean across all texts/layers
    'overall_std': float                 # Standard deviation of mean curvatures
}
```

**Detailed Explanation:**

- **`mean_curvatures`**: Array of mean curvature values per layer
  - **Shape**: `(num_interior_points,)`
  - **Calculation**: Average of curvatures across all texts for each layer
  - **Purpose**: Identifies which layers show consistent geometric complexity

- **`std_curvatures`**: Array of standard deviations per layer
  - **Shape**: Same as `mean_curvatures`
  - **Interpretation**: Variability of curvature across different input texts
  - **High std**: Layer behavior varies significantly with input content
  - **Low std**: Layer shows consistent geometric properties

- **`all_curvatures`**: List containing raw curvature arrays for each processed text
  - **Structure**: `[[text1_curvatures], [text2_curvatures], ...]`
  - **Purpose**: Enables detailed per-text analysis and custom aggregations

- **`num_texts`**: Number of successfully processed texts
  - **Note**: May be less than input if some texts failed processing

- **`layer_indices`**: Array of interior layer numbers
  - **Shape**: `(num_interior_points,)`
  - **Values**: `[1, 2, ..., num_interior_points]`

- **`overall_mean`**: Grand mean curvature across all texts and layers
  - **Calculation**: `mean(mean_curvatures)`
  - **Usage**: Model-level geometric complexity metric

- **`overall_std`**: Standard deviation of layer-wise mean curvatures
  - **Calculation**: `std(mean_curvatures)`
  - **Interpretation**: How much geometric complexity varies across layers

### 3. Named Prompts Analysis (`calculate_curvature_for_prompts`)

**Dictionary Structure:**
```python
{
    'prompt_name_1': {
        # Same structure as single text analysis
        'curvatures': [...],
        'mean_curvature': float,
        # ... etc
    },
    'prompt_name_2': { ... },
    # ... more prompts
}
```

**Purpose**: Organized results for comparing curvature across different prompt categories or text types.

## Mathematical Background

### Square-Root Embedding
- **Formula**: `u = √p / ||√p||` where p is probability distribution
- **Purpose**: Maps probability simplex to unit sphere for geometric analysis
- **Benefit**: Enables use of spherical geometry tools

### Discrete Curvature Formula
- **Second-order difference**: `Δ²u[ℓ] = u[ℓ+1] - 2u[ℓ] + u[ℓ-1]`
- **Tangent projection**: `d²u = Δ²u - (u · Δ²u)u` (projects onto tangent space)
- **Curvature**: `κ = ||d²u|| / ||du||^(3/2)` where `du` is first-order difference

### Fisher-Rao Distance
- **Bhattacharyya coefficient**: `BC(p,q) = ∑ᵢ √(pᵢ × qᵢ)`
- **Fisher-Rao distance**: `d_FR = 2 × arccos(BC)`
- **Range**: [0, π] radians on probability simplex

## Usage Examples

### Accessing Raw Arrays
```python
# Thermodynamic results
param_strain = results['parameter_strain']['layer_grad_norms']
fisher_rao_steps = results['fisher_rao']['step_lengths']
efficiency_scores = results['semantic_efficiency']['semantic_efficiency']

# Spectral results  
layer_curvatures = results['spectral_curvature']['mean_curvatures']
curvature_variability = results['spectral_curvature']['std_curvatures']
```

### Interpretation Guidelines

**High Parameter Strain**: Layer is computationally active, potentially learning complex patterns
**High Fisher-Rao Distance**: Significant changes in model predictions between layers
**High Semantic Efficiency**: Layer efficiently converts computation into meaningful output changes
**High Spectral Curvature**: Rapid changes in representational geometry, indicates specialization

**Low values** generally indicate the opposite: computational stability, minimal prediction changes, inefficient processing, or smooth geometric transformations.

## File Output Formats

When using the `save_raw_arrays()` function, arrays are saved as:

- `parameter_strain_per_layer.npy`: Contains `layer_grad_norms` array
- `fisher_rao_step_lengths.npy`: Contains `step_lengths` array  
- `semantic_efficiency_per_text.npy`: Contains `semantic_efficiency` array
- `spectral_mean_curvatures.npy`: Contains `mean_curvatures` array
- `spectral_std_curvatures.npy`: Contains `std_curvatures` array
- `metadata.json`: Contains analysis metadata and array descriptions

These files can be loaded with `numpy.load()` for further analysis or visualization.
