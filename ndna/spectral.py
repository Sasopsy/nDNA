"""
Spectral Curvature Calculator

Implements spectral curvature analysis using geometric methods on the probability simplex.
Uses logit lens to extract layerwise distributions and computes discrete curvature 
on the Fisher-Rao manifold.

References:
- The original notebook implementation  
- Information geometry and Fisher-Rao metrics
- Spectral curvature analysis methods
"""

import torch
# import torch.nn.functional as F  # Not used directly
import numpy as np
import math
from typing import Dict, List, Tuple, Union
from tqdm import tqdm

from .utils.model_handler import ModelHandler


class SpectralCalculator:
    """
    Calculator for spectral curvature metrics using geometric analysis.
    
    Computes discrete curvature on the probability simplex using square-root embeddings
    and tangent space projections.
    """
    
    def __init__(self, 
                 model_handler: ModelHandler,
                 eps_dist: float = 1e-12,
                 eps_curv: float = 1e-12,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize spectral curvature calculator.
        
        Args:
            model_handler: Initialized ModelHandler instance
            eps_dist: Epsilon for probability distribution stability
            eps_curv: Epsilon for curvature calculation stability
            dtype: Data type for calculations
        """
        self.model_handler = model_handler
        self.device = model_handler.device
        self.eps_dist = eps_dist
        self.eps_curv = eps_curv
        self.dtype = dtype
        
        # Get model components
        self.final_ln = model_handler.get_final_layer_norm()
        self.lm_head = model_handler.get_lm_head()
        self.num_layers = model_handler.num_layers
        self.vocab_size = model_handler.vocab_size
        
        if model_handler.is_causal and (self.final_ln is None or self.lm_head is None):
            raise ValueError("Causal LM model must have final layer norm and lm_head for spectral calculations")
    
    def _sqrt_embedding(self, q: torch.Tensor) -> torch.Tensor:
        """
        Convert probability distribution to unit-normalized square-root embedding.
        
        Args:
            q: Probability distribution (V,)
            
        Returns:
            Unit-normalized square-root embedding (V,)
        """
        q = torch.clamp(q, min=self.eps_dist)
        q = q / q.sum()  # Ensure normalization
        u = torch.sqrt(q)
        u = u / (torch.norm(u, p=2) + 1e-30)  # Unit normalize
        return u
    
    def _project_tangent(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project vector v onto the tangent space at u on the unit sphere.
        
        Args:
            u: Point on unit sphere (V,)
            v: Vector to project (V,)
            
        Returns:
            Tangent vector (V,)
        """
        return v - torch.dot(u, v) * u
    
    def _discrete_curvature(self, u_list: List[torch.Tensor]) -> Tuple[List[float], List[float]]:
        """
        Compute discrete curvature for interior points using finite differences.
        
        Args:
            u_list: List of square-root embeddings for each layer
            
        Returns:
            Tuple of (curvatures, speeds) for interior layers
        """
        m = len(u_list)
        if m < 3:
            raise ValueError("Need at least 3 layers for curvature computation")
        
        # First-order differences and speeds
        delta_u = []
        speeds = []
        
        for ell in range(m - 1):
            u = u_list[ell]
            v = u_list[ell + 1] - u
            du = self._project_tangent(u, v)
            delta_u.append(du)
            speeds.append(torch.norm(du, p=2).item())
        
        # Second-order differences and curvatures (interior points only)
        curvatures = []
        
        for ell in range(1, m - 1):  # Interior points: ell = 1 to m-2
            u = u_list[ell]
            # Second difference: u[ell+1] - 2*u[ell] + u[ell-1]
            v2 = (u_list[ell + 1] - 2 * u_list[ell] + u_list[ell - 1])
            d2u = self._project_tangent(u, v2)
            
            # Curvature formula: ||d2u|| / ||du||^(3/2)
            numerator = torch.norm(d2u, p=2)
            speed = torch.norm(delta_u[ell], p=2)  
            denominator = (speed * speed + self.eps_curv) ** 1.5
            
            curvature = (numerator / denominator).item()
            curvatures.append(curvature)
        
        return curvatures, speeds
    
    @torch.no_grad()
    def _extract_layerwise_distributions(self, text: str) -> List[torch.Tensor]:
        """
        Extract next-token probability distributions from each layer using logit lens.
        
        Args:
            text: Input text string
            
        Returns:
            List of probability distributions (one per layer + embedding)
        """
        # Tokenize input
        tokenizer = self.model_handler.tokenizer
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        
        if encoded.input_ids.shape[1] == 0:
            raise ValueError("Empty tokenized input")
        
        # Forward pass with hidden states
        model_output = self.model_handler.forward_with_hidden_states(encoded.input_ids)
        hidden_states = model_output["hidden_states"] if isinstance(model_output, dict) else model_output.hidden_states
        
        # Extract distributions at the last token position (for next-token prediction)
        last_pos = encoded.input_ids.shape[1] - 1
        distributions = []
        
        for hidden_state in hidden_states:
            # Get hidden state at last position
            h = hidden_state[0, last_pos, :]  # (hidden_size,)
            
            # Apply final layer norm if available
            if self.final_ln is not None:
                h = self.final_ln(h)
            
            # Apply language modeling head
            lm_head = self.lm_head
            if lm_head is not None:
                logits = lm_head(h)  # (vocab_size,)
            else:
                # For models without LM head, use a dummy uniform distribution
                logits = torch.zeros(self.vocab_size, device=self.device)
            
            # Convert to probability distribution
            q = torch.softmax(logits.to(self.dtype), dim=-1)
            distributions.append(q)
        
        return distributions
    
    def calculate_curvature_for_text(self, text: str) -> Dict[str, Union[List[float], np.ndarray]]:
        """
        Calculate spectral curvature for a single text input.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary containing curvature metrics
        """
        # Extract layerwise probability distributions
        distributions = self._extract_layerwise_distributions(text)
        
        # Convert to square-root embeddings
        embeddings = [self._sqrt_embedding(q) for q in distributions]
        
        # Compute discrete curvatures
        curvatures, speeds = self._discrete_curvature(embeddings)
        
        results = {
            'curvatures': curvatures,
            'speeds': speeds,
            'num_layers': len(distributions),
            'num_interior_points': len(curvatures),
            'layer_indices': list(range(1, len(curvatures) + 1)),
            'mean_curvature': float(np.mean(curvatures)) if curvatures else 0.0,
            'max_curvature': float(np.max(curvatures)) if curvatures else 0.0,
            'min_curvature': float(np.min(curvatures)) if curvatures else 0.0
        }
        
        return results
    
    def calculate_curvature_for_texts(self, texts: List[str]) -> Dict[str, Union[np.ndarray, float]]:
        """
        Calculate spectral curvature for multiple text inputs.
        
        Args:
            texts: List of input text strings
            
        Returns:
            Dictionary containing aggregated curvature metrics
        """
        all_curvatures = []
        all_speeds = []
        
        for text in tqdm(texts, desc="Processing texts"):
            try:
                result = self.calculate_curvature_for_text(text)
                all_curvatures.append(result['curvatures'])
                all_speeds.append(result['speeds'])
            except Exception as e:
                print(f"Warning: Failed to process text '{text[:50]}...': {e}")
                continue
        
        if not all_curvatures:
            raise ValueError("No texts were successfully processed")
        
        # Aggregate results
        # Assume all texts have same number of layers (take first valid result)
        num_interior = len(all_curvatures[0])
        
        # Average curvatures across all texts
        mean_curvatures = np.zeros(num_interior)
        std_curvatures = np.zeros(num_interior)
        
        for layer_idx in range(num_interior):
            layer_curvatures = [curves[layer_idx] for curves in all_curvatures if len(curves) > layer_idx]
            if layer_curvatures:
                mean_curvatures[layer_idx] = np.mean(layer_curvatures)
                std_curvatures[layer_idx] = np.std(layer_curvatures)
        
        results = {
            'mean_curvatures': mean_curvatures,
            'std_curvatures': std_curvatures,
            'all_curvatures': all_curvatures,
            'num_texts': len(all_curvatures),
            'layer_indices': np.arange(1, num_interior + 1),
            'overall_mean': float(mean_curvatures.mean()),
            'overall_std': float(mean_curvatures.std())
        }
        
        print(f"Spectral curvature calculated for {results['num_texts']} texts. "
              f"Overall mean curvature: {results['overall_mean']:.6f}")
        
        return results
    
    def calculate_curvature_for_prompts(self, 
                                      prompts: List[Tuple[str, str]]) -> Dict[str, Dict]:
        """
        Calculate spectral curvature for named prompt categories.
        
        Args:
            prompts: List of (name, text) tuples
            
        Returns:
            Dictionary mapping prompt names to curvature results
        """
        results = {}
        
        for name, text in prompts:
            print(f"\n=== Processing: {name} ===")
            print(f"Text: {text}")
            
            try:
                result = self.calculate_curvature_for_text(text)
                results[name] = result
                
                print(f"Layers: {result['num_layers']}")
                print(f"Mean curvature: {result['mean_curvature']:.6e}")
                print("Per-layer curvatures:")
                for i, k in enumerate(result['curvatures'], 1):
                    print(f"  Layer {i:3d}: {k:.6e}")
                    
            except Exception as e:
                print(f"Error processing {name}: {e}")
                results[name] = None
        
        return results
    
    def describe_results(self, 
                        results: Union[Dict[str, Dict], Dict[str, Union[np.ndarray, float]]]) -> Dict[str, str]:
        """
        Generate detailed descriptions of spectral curvature results.
        
        Args:
            results: Either single result dict or dict of named results
            
        Returns:
            Dictionary with detailed descriptions and interpretation guides  
        """
        descriptions = {
            "spectral_curvature": {
                "description": "Measures the geometric curvature of probability simplex manifold as representations evolve through layers",
                "formula": "κ = ||Δ²u|| / ||Δu||^(3/2) where u = √p/||√p|| (square-root embedding on unit sphere)",
                "interpretation": {
                    "high_curvature": "Rapid changes in representational geometry, indicates layer specialization",
                    "low_curvature": "Smooth, gradual transformations between layers",
                    "curvature_peaks": "Often correspond to critical processing layers or architectural boundaries",
                    "language_patterns": "Different text types (technical, natural, mathematical) show distinct curvature profiles"
                },
                "mathematical_background": {
                    "square_root_embedding": "Maps probability distributions to unit sphere: u = √p/||√p||",
                    "tangent_projection": "Projects finite differences onto tangent space to handle sphere geometry",
                    "discrete_curvature": "Uses second-order finite differences to approximate continuous curvature",
                    "numerical_stability": "Uses epsilon values to prevent division by zero in low-curvature regions"
                }
            }
        }
        
        # Add specific format information based on result type
        if 'curvatures' in results:
            # Single text result
            descriptions["data_format"] = {
                "curvatures": "Array of curvature values for interior layers (length = num_layers-2)",
                "speeds": "Array of first-order difference magnitudes (length = num_layers-1)",
                "num_layers": "Total number of layers including embeddings",
                "layer_indices": "Interior layer indices (excludes first and last layers)",
                "mean_curvature": "Average curvature across all interior layers",
                "max_curvature": "Maximum curvature value found",
                "min_curvature": "Minimum curvature value found"
            }
        elif 'mean_curvatures' in results:
            # Multiple texts result
            descriptions["data_format"] = {
                "mean_curvatures": "Array of mean curvature values across texts per layer",
                "std_curvatures": "Array of standard deviation values per layer",
                "all_curvatures": "List of curvature arrays, one per input text",
                "num_texts": "Number of texts processed",
                "layer_indices": "Interior layer indices",
                "overall_mean": "Grand mean curvature across all texts and layers",
                "overall_std": "Standard deviation of mean curvatures across layers"
            }
        else:
            # Named results (multiple prompts)
            descriptions["data_format"] = {
                "structure": "Dictionary mapping prompt names to individual curvature results",
                "each_result_contains": "Same fields as single text result (curvatures, mean_curvature, etc.)",
                "failed_prompts": "Entries with None value indicate processing failures"
            }
            
        descriptions["usage_tips"] = {
            "comparison": "Compare mean_curvature values across different models or text types",
            "analysis": "Look for curvature peaks to identify specialized layers",
            "filtering": "Very high curvatures (>100) may indicate numerical instabilities",
            "aggregation": "Use overall_mean for model-level comparisons, layer-wise for detailed analysis"
        }
        
        return descriptions
        
    def get_sample_prompts(self) -> List[Tuple[str, str]]:
        """
        Get a set of sample prompts for testing spectral curvature across different text types.
        
        Returns:
            List of (name, text) prompt tuples
        """
        prompts = [
            ("Simple English", 
             "The artist drew a landscape with a river flowing towards the mountains."),
            
            ("Simple Hindi", 
             "इसे आज़माने के लिए, नीचे अपनी भाषा और इनपुट उपकरण चुनें और लिखना आरंभ करें|"),
            
            ("Arabic", 
             "أنا ملك عالمي الخاص ولا أجرؤ على إجباري على تلبية احتياجات من اختيارك"),
            
            ("Sanskrit", 
             "श्वः अतीव द्रुतं धावति"),
            
            ("Complex Math", 
             r"""Z = \int \mathcal{D}\phi \, \exp \left( i \int d^4x \, \sqrt{-g(x)}
\left[ \frac{1}{2} g^{\mu\nu}(x) \partial_\mu \phi(x) \, \partial_\nu \phi(x)
- \frac{1}{2} m^2 \phi^2(x) - \frac{\lambda}{4!} \phi^4(x)
+ \frac{1}{16\pi G} (R(x) - 2\Lambda) \right] \right)"""),
            
            ("Programming", 
             "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
             
            ("Scientific", 
             "The mitochondrion is the powerhouse of the cell, responsible for ATP synthesis through oxidative phosphorylation.")
        ]
        
        return prompts
