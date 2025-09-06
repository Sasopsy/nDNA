import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def setup_model_general(model_name: str, use_quantization: bool = True):
    """
    Setup any transformer model with optional quantization for Colab
    """
    print(f"Loading model: {model_name}")

    # Configure quantization for memory efficiency
    if use_quantization and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load as CausalLM, trying AutoModel: {e}")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

    return tokenizer, model

class ThermodynamicLengthCalculator:
    """
    Thermodynamic Length calculator integrated with existing architecture
    """
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if torch.cuda.is_available() else "cpu"
        
    def get_lm_head(self):
        """Extract the language modeling head from the model"""
        # Try different common attribute names for the LM head
        head_attrs = ['lm_head', 'head', 'classifier', 'score']
        
        for attr in head_attrs:
            if hasattr(self.model, attr):
                head = getattr(self.model, attr)
                if isinstance(head, nn.Linear):
                    return head
        
        # If no direct head found, try to find it in nested modules
        for name, module in self.model.named_modules():
            if any(head_name in name.lower() for head_name in head_attrs):
                if isinstance(module, nn.Linear):
                    return module
        
        print("Warning: Could not find LM head, creating dummy head")
        # Create a dummy head if we can't find one
        vocab_size = len(self.tokenizer)
        hidden_size = self.model.config.hidden_size if hasattr(self.model, 'config') else 768
        return nn.Linear(hidden_size, vocab_size)
    
    def apply_logit_lens(self, hidden_states_sequence: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply logit lens to convert hidden states to probability distributions
        """
        lm_head = self.get_lm_head()
        lm_head = lm_head.to(next(self.model.parameters()).device)
        
        probability_distributions = []
        
        with torch.no_grad():
            for hidden_state in hidden_states_sequence:
                # Move to same device as model
                hidden_state = hidden_state.to(next(self.model.parameters()).device)
                
                # Apply language modeling head
                logits = lm_head(hidden_state)  # (batch, seq_len, vocab_size)
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Move back to CPU to save memory
                probability_distributions.append(probs.cpu())
        
        return probability_distributions
    
    def kl_divergence(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Calculate KL divergence KL(p || q) with numerical stability"""
        p = torch.clamp(p, eps, 1.0)
        q = torch.clamp(q, eps, 1.0)
        return torch.sum(p * torch.log(p / q), dim=-1)
    
    def calculate_thermodynamic_length(self, hidden_states_sequence: List[torch.Tensor]) -> Dict[str, any]:
        """
        Calculate thermodynamic length using prediction-based method
        """
        # Apply logit lens to get probability distributions
        prob_distributions = self.apply_logit_lens(hidden_states_sequence)
        
        if len(prob_distributions) < 2:
            return {
                'total_thermodynamic_length': 0.0,
                'layer_step_lengths': {},
                'step_lengths_array': []
            }
        
        batch_size, seq_len, vocab_size = prob_distributions[0].shape
        
        # Calculate step lengths between adjacent layers
        step_lengths = {}
        step_lengths_array = []
        total_length = 0.0
        
        for layer_idx in range(len(prob_distributions) - 1):
            p_curr = prob_distributions[layer_idx]     # (batch, seq_len, vocab_size)
            p_next = prob_distributions[layer_idx + 1] # (batch, seq_len, vocab_size)
            
            # Calculate KL divergence for each position and batch element
            kl_divs = self.kl_divergence(p_curr, p_next)  # (batch, seq_len)
            
            # Average over batch and sequence
            avg_kl = torch.mean(kl_divs).item()
            
            # Calculate step length
            step_length = np.sqrt(2 * avg_kl) if avg_kl > 0 else 0.0
            
            step_lengths[f'layer_{layer_idx}_to_{layer_idx+1}'] = step_length
            step_lengths_array.append(step_length)
            total_length += step_length
        
        return {
            'total_thermodynamic_length': total_length,
            'layer_step_lengths': step_lengths,
            'step_lengths_array': step_lengths_array
        }

class EnhancedAnalysisCalculator:
    def __init__(self, tokenizer, model, device="cuda"):
        """
        Initialize calculator with both spectral curvature and thermodynamic length
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        
        # Initialize thermodynamic length calculator
        self.tl_calculator = ThermodynamicLengthCalculator(model, tokenizer, device)

        print(f"Using device: {self.device}")

    def get_activations_optimized(self, text: str, max_length: int = 256) -> Dict[str, torch.Tensor]:
        """
        Memory-optimized activation extraction using output_hidden_states
        """
        # Tokenize with shorter max_length to save memory
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        # Move inputs to the same device as model
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        activations = {}

        with torch.no_grad():
            # Use gradient checkpointing if available to save memory
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

            outputs = self.model(**inputs, output_hidden_states=True)

            # Extract hidden states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                for i, hidden_state in enumerate(outputs.hidden_states):
                    # Move to CPU immediately to save GPU memory
                    activations[f'layer_{i}'] = hidden_state.cpu().detach()

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return activations

    def compute_first_order_difference(self, h_t: torch.Tensor, h_t_minus_1: torch.Tensor) -> torch.Tensor:
        """Compute first-order difference: Δh_t = h_t - h_{t-1}"""
        return h_t - h_t_minus_1

    def compute_second_order_difference(self, h_t_plus_1: torch.Tensor, h_t: torch.Tensor, h_t_minus_1: torch.Tensor) -> torch.Tensor:
        """Compute second-order difference: Δ²h_t = Δh_{t+1} - Δh_t"""
        delta_h_t_plus_1 = h_t_plus_1 - h_t
        delta_h_t = h_t - h_t_minus_1
        return delta_h_t_plus_1 - delta_h_t

    def compute_spectral_curvature(self, activations_sequence: List[torch.Tensor],
                                 handle_boundaries: str = 'exclude') -> List[float]:
        """Compute spectral curvature for activation sequence"""
        curvatures = []
        n_layers = len(activations_sequence)

        if handle_boundaries == 'exclude':
            # Skip first and last layers
            for i in range(1, n_layers - 1):
                h_t_minus_1 = activations_sequence[i - 1]
                h_t = activations_sequence[i]
                h_t_plus_1 = activations_sequence[i + 1]

                # Compute second-order difference
                delta_2_h_t = self.compute_second_order_difference(h_t_plus_1, h_t, h_t_minus_1)

                # Compute first-order differences for normalization
                delta_h_t = self.compute_first_order_difference(h_t, h_t_minus_1)
                delta_h_t_plus_1 = self.compute_first_order_difference(h_t_plus_1, h_t)

                # Compute spectral curvature
                numerator = torch.norm(delta_2_h_t, dim=-1)
                denominator = torch.norm(delta_h_t_plus_1 + delta_h_t, dim=-1) + 1e-8

                curvature = numerator / denominator
                curvature_mean = curvature.mean().item()
                curvatures.append(curvature_mean)

        return curvatures

    def process_sample_enhanced(self, sample: Dict, max_length: int = 256) -> Dict:
        """Process single sample with both spectral curvature and thermodynamic length"""
        # Combine context and question (truncated)
        context = sample['context'][:500]  # Limit context length
        question = sample['question'][:100]  # Limit question length
        text = f"Context: {context} Question: {question}"

        # Get activations
        activations = self.get_activations_optimized(text, max_length)

        # Sort by layer number
        sorted_layers = sorted(activations.keys(),
                             key=lambda x: int(x.split('_')[1]) if '_' in x else 0)

        # Extract activation sequence
        activation_sequence = [activations[layer] for layer in sorted_layers]

        # Compute spectral curvature
        curvatures = self.compute_spectral_curvature(activation_sequence)

        # Compute thermodynamic length
        tl_results = self.tl_calculator.calculate_thermodynamic_length(activation_sequence)

        return {
            'sample_id': sample.get('id', ''),
            'layer_names': sorted_layers[1:-1] if len(sorted_layers) > 2 else sorted_layers,
            'spectral_curvatures': curvatures,
            'mean_curvature': np.mean(curvatures) if curvatures else 0,
            'max_curvature': np.max(curvatures) if curvatures else 0,
            'thermodynamic_length': tl_results['total_thermodynamic_length'],
            'tl_step_lengths': tl_results['step_lengths_array'],
            'tl_layer_contributions': tl_results['layer_step_lengths']
        }

def create_3d_visualizations(results: List[Dict], save_path: str = '/content/'):
    """
    Create 3D visualizations and animated GIFs
    """
    print("Creating 3D visualizations...")
    
    if not results:
        print("No results to visualize")
        return
    
    # Extract data
    sample_indices = list(range(len(results)))
    
    # Get maximum layer count for consistent plotting
    max_layers = max(len(r['spectral_curvatures']) for r in results if r['spectral_curvatures'])
    
    # Prepare data matrices
    curvature_matrix = np.zeros((len(results), max_layers))
    tl_matrix = np.zeros((len(results), max_layers))
    
    for i, result in enumerate(results):
        curvatures = result['spectral_curvatures']
        tl_steps = result['tl_step_lengths']
        
        # Fill matrices (pad with zeros if necessary)
        if curvatures:
            curvature_matrix[i, :len(curvatures)] = curvatures
        if tl_steps:
            tl_matrix[i, :len(tl_steps)] = tl_steps
    
    # Create meshgrids for 3D plotting
    X, Y = np.meshgrid(range(max_layers), sample_indices)
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 3D Surface plot for Spectral Curvature
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, curvature_matrix, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Sample Index')
    ax1.set_zlabel('Spectral Curvature')
    ax1.set_title('3D Surface: Spectral Curvature')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # 2. 3D Surface plot for Thermodynamic Length
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, tl_matrix, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Sample Index')
    ax2.set_zlabel('TL Step Length')
    ax2.set_title('3D Surface: Thermodynamic Length Steps')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # 3. Combined 3D Scatter plot
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    
    # Flatten matrices for scatter plot
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_curv = curvature_matrix.flatten()
    z_tl = tl_matrix.flatten()
    
    # Color points by thermodynamic length
    scatter = ax3.scatter(x_flat, y_flat, z_curv, c=z_tl, cmap='coolwarm', alpha=0.6)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Sample Index')
    ax3.set_zlabel('Spectral Curvature')
    ax3.set_title('3D Scatter: Curvature colored by TL')
    fig.colorbar(scatter, ax=ax3, shrink=0.5)
    
    # 4. Heatmap of Spectral Curvature
    ax4 = fig.add_subplot(2, 3, 4)
    im1 = ax4.imshow(curvature_matrix, cmap='viridis', aspect='auto')
    ax4.set_xlabel('Layer Index')
    ax4.set_ylabel('Sample Index')
    ax4.set_title('Heatmap: Spectral Curvature')
    fig.colorbar(im1, ax=ax4)
    
    # 5. Heatmap of Thermodynamic Length
    ax5 = fig.add_subplot(2, 3, 5)
    im2 = ax5.imshow(tl_matrix, cmap='plasma', aspect='auto')
    ax5.set_xlabel('Layer Index')
    ax5.set_ylabel('Sample Index')
    ax5.set_title('Heatmap: Thermodynamic Length Steps')
    fig.colorbar(im2, ax=ax5)
    
    # 6. Correlation plot
    ax6 = fig.add_subplot(2, 3, 6)
    # Average across samples for each layer
    avg_curvature = np.mean(curvature_matrix, axis=0)
    avg_tl = np.mean(tl_matrix, axis=0)
    
    ax6.scatter(avg_curvature, avg_tl, alpha=0.7)
    ax6.set_xlabel('Average Spectral Curvature')
    ax6.set_ylabel('Average TL Step Length')
    ax6.set_title('Layer-wise Correlation')
    
    # Add trend line
    if len(avg_curvature) > 1:
        z = np.polyfit(avg_curvature, avg_tl, 1)
        p = np.poly1d(z)
        ax6.plot(avg_curvature, p(avg_curvature), "r--", alpha=0.8)
        
        # Calculate correlation coefficient
        corr = np.corrcoef(avg_curvature, avg_tl)[0, 1]
        ax6.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}3d_analysis_static.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create animated GIF for 3D surface rotation
    create_animated_gif(curvature_matrix, tl_matrix, save_path)

def create_animated_gif(curvature_matrix: np.ndarray, tl_matrix: np.ndarray, save_path: str):
    """
    Create animated GIF of rotating 3D surfaces
    """
    print("Creating animated GIF...")
    
    # Prepare data
    max_layers = curvature_matrix.shape[1]
    sample_indices = list(range(curvature_matrix.shape[0]))
    X, Y = np.meshgrid(range(max_layers), sample_indices)
    
    # Create figure for animation
    fig = plt.figure(figsize=(16, 8))
    
    # Animation function
    def animate(frame):
        fig.clear()
        
        # Spectral Curvature surface
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf1 = ax1.plot_surface(X, Y, curvature_matrix, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Sample Index')
        ax1.set_zlabel('Spectral Curvature')
        ax1.set_title('Spectral Curvature Evolution')
        ax1.view_init(elev=30, azim=frame * 4)  # Rotate view
        
        # Thermodynamic Length surface
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(X, Y, tl_matrix, cmap='plasma', alpha=0.8)
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Sample Index')
        ax2.set_zlabel('TL Step Length')
        ax2.set_title('Thermodynamic Length Evolution')
        ax2.view_init(elev=30, azim=frame * 4)  # Rotate view
        
        plt.tight_layout()
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=90, interval=100, blit=False)
    
    # Save as GIF
    try:
        anim.save(f'{save_path}3d_analysis_animation.gif', writer='pillow', fps=10)
        print(f"Animated GIF saved to {save_path}3d_analysis_animation.gif")
    except Exception as e:
        print(f"Could not save GIF: {e}")
        print("You may need to install pillow: pip install pillow")

def create_advanced_visualizations(results: List[Dict], save_path: str = '/content/'):
    """
    Create additional advanced visualizations
    """
    print("Creating advanced visualizations...")
    
    if not results:
        return
    
    # Extract comprehensive data
    all_curvatures = []
    all_tl_steps = []
    total_tl_lengths = []
    
    for result in results:
        all_curvatures.extend(result['spectral_curvatures'])
        all_tl_steps.extend(result['tl_step_lengths'])
        total_tl_lengths.append(result['thermodynamic_length'])
    
    # Create comprehensive analysis figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution comparison
    axes[0, 0].hist(all_curvatures, bins=30, alpha=0.7, label='Spectral Curvature', color='blue')
    ax_twin = axes[0, 0].twinx()
    ax_twin.hist(all_tl_steps, bins=30, alpha=0.7, label='TL Steps', color='red')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency (Curvature)', color='blue')
    ax_twin.set_ylabel('Frequency (TL Steps)', color='red')
    axes[0, 0].set_title('Distribution Comparison')
    
    # 2. Sample-wise total thermodynamic length
    axes[0, 1].plot(total_tl_lengths, 'o-', alpha=0.7)
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Total Thermodynamic Length')
    axes[0, 1].set_title('Total TL per Sample')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Layer-wise analysis
    max_layers = max(len(r['spectral_curvatures']) for r in results if r['spectral_curvatures'])
    layer_curvatures = [[] for _ in range(max_layers)]
    layer_tl_steps = [[] for _ in range(max_layers)]
    
    for result in results:
        curvatures = result['spectral_curvatures']
        tl_steps = result['tl_step_lengths']
        
        for i, curv in enumerate(curvatures):
            if i < max_layers:
                layer_curvatures[i].append(curv)
        
        for i, tl_step in enumerate(tl_steps):
            if i < max_layers:
                layer_tl_steps[i].append(tl_step)
    
    # Calculate means and stds
    layer_curv_means = [np.mean(layer) if layer else 0 for layer in layer_curvatures]
    layer_curv_stds = [np.std(layer) if layer else 0 for layer in layer_curvatures]
    layer_tl_means = [np.mean(layer) if layer else 0 for layer in layer_tl_steps]
    layer_tl_stds = [np.std(layer) if layer else 0 for layer in layer_tl_steps]
    
    x_layers = list(range(max_layers))
    
    axes[0, 2].errorbar(x_layers, layer_curv_means, yerr=layer_curv_stds, 
                       label='Spectral Curvature', marker='o', capsize=5)
    ax_twin2 = axes[0, 2].twinx()
    ax_twin2.errorbar(x_layers, layer_tl_means, yerr=layer_tl_stds, 
                     label='TL Steps', marker='s', color='red', capsize=5)
    axes[0, 2].set_xlabel('Layer Index')
    axes[0, 2].set_ylabel('Mean Curvature', color='blue')
    ax_twin2.set_ylabel('Mean TL Step', color='red')
    axes[0, 2].set_title('Layer-wise Mean ± Std')
    
    # 4. Correlation heatmap
    if len(results) > 1:
        # Create correlation matrix
        max_len = max(len(r['spectral_curvatures']) for r in results if r['spectral_curvatures'])
        curv_matrix = np.zeros((len(results), max_len))
        tl_matrix = np.zeros((len(results), max_len))
        
        for i, result in enumerate(results):
            curvatures = result['spectral_curvatures']
            tl_steps = result['tl_step_lengths']
            
            if curvatures:
                curv_matrix[i, :len(curvatures)] = curvatures
            if tl_steps:
                tl_matrix[i, :len(tl_steps)] = tl_steps
        
        # Calculate correlation for each layer
        layer_correlations = []
        for layer in range(max_len):
            curv_layer = curv_matrix[:, layer]
            tl_layer = tl_matrix[:, layer]
            
            # Only calculate correlation if both have non-zero values
            if np.any(curv_layer) and np.any(tl_layer):
                corr = np.corrcoef(curv_layer, tl_layer)[0, 1]
                layer_correlations.append(corr if not np.isnan(corr) else 0)
            else:
                layer_correlations.append(0)
        
        axes[1, 0].plot(layer_correlations, 'o-', linewidth=2, markersize=6)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Correlation (Curvature vs TL)')
        axes[1, 0].set_title('Layer-wise Correlation')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Cumulative analysis
    if results:
        sample_curv_sums = [sum(r['spectral_curvatures']) for r in results]
        sample_tl_sums = [r['thermodynamic_length'] for r in results]
        
        axes[1, 1].scatter(sample_curv_sums, sample_tl_sums, alpha=0.7)
        axes[1, 1].set_xlabel('Total Spectral Curvature')
        axes[1, 1].set_ylabel('Total Thermodynamic Length')
        axes[1, 1].set_title('Sample-wise Total Comparison')
        
        # Add trend line
        if len(sample_curv_sums) > 1:
            z = np.polyfit(sample_curv_sums, sample_tl_sums, 1)
            p = np.poly1d(z)
            axes[1, 1].plot(sample_curv_sums, p(sample_curv_sums), "r--", alpha=0.8)
    
    # 6. Summary statistics
    axes[1, 2].axis('off')
    summary_text = f"""
    Analysis Summary:
    
    Samples processed: {len(results)}
    
    Spectral Curvature:
    • Mean: {np.mean(all_curvatures):.4f}
    • Std: {np.std(all_curvatures):.4f}
    • Range: [{np.min(all_curvatures):.4f}, {np.max(all_curvatures):.4f}]
    
    Thermodynamic Length:
    • Mean step: {np.mean(all_tl_steps):.4f}
    • Std step: {np.std(all_tl_steps):.4f}
    • Mean total: {np.mean(total_tl_lengths):.4f}
    
    Overall Correlation: {np.corrcoef(all_curvatures[:len(all_tl_steps)], all_tl_steps)[0,1]:.4f}
    """
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}advanced_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("GPU not available - using CPU (will be slow)")

    # Setup model - Using an open alternative to Llama
    print("Setting up model...")
    # Options for open models:
    # - "microsoft/DialoGPT-medium" (smaller, good for testing)
    # - "EleutherAI/gpt-neo-1.3B" (GPT-style model)
    # - "distilbert-base-uncased" (BERT-style, smaller)
    # - "gpt2" (classic GPT-2)

    model_name = "microsoft/DialoGPT-medium"  # Change this to your preferred model
    print(f"Using model: {model_name}")

    tokenizer, model = setup_model_general(model_name, use_quantization=True)

    # Initialize enhanced calculator
    calculator = EnhancedAnalysisCalculator(tokenizer, model)

    # Load SQuAD dataset (smaller subset for Colab)
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split="validation[:25]")  # Small subset for demo

    # Process samples with enhanced analysis
    print("Processing samples with both spectral curvature and thermodynamic length...")
    results = []

    for i, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        try:
            result = calculator.process_sample_enhanced(sample, max_length=128)
            results.append(result)

            # Clear memory more frequently for Colab
            if i % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Analyze results
    if results:
        print(f"\n📊 Enhanced Analysis Results:")
        print(f"Processed {len(results)} samples")
        
        # Spectral Curvature Analysis
        all_curvatures = []
        for result in results:
            all_curvatures.extend(result['spectral_curvatures'])
        
        if all_curvatures:
            print(f"\n🌊 Spectral Curvature:")
            print(f"  Mean: {np.mean(all_curvatures):.4f}")
            print(f"  Std: {np.std(all_curvatures):.4f}")
            print(f"  Min: {np.min(all_curvatures):.4f}")
            print(f"  Max: {np.max(all_curvatures):.4f}")
        
        # Thermodynamic Length Analysis
        all_tl_steps = []
        total_tl_lengths = []
        for result in results:
            all_tl_steps.extend(result['tl_step_lengths'])
            total_tl_lengths.append(result['thermodynamic_length'])
        
        if all_tl_steps:
            print(f"\n🔥 Thermodynamic Length:")
            print(f"  Mean step length: {np.mean(all_tl_steps):.4f}")
            print(f"  Std step length: {np.std(all_tl_steps):.4f}")
            print(f"  Mean total length: {np.mean(total_tl_lengths):.4f}")
            print(f"  Max total length: {np.max(total_tl_lengths):.4f}")
        
        # Correlation Analysis
        if all_curvatures and all_tl_steps:
            # Ensure same length for correlation
            min_len = min(len(all_curvatures), len(all_tl_steps))
            correlation = np.corrcoef(all_curvatures[:min_len], all_tl_steps[:min_len])[0, 1]
            print(f"\n🔗 Correlation (Curvature vs TL steps): {correlation:.4f}")

        # Save results
        output_data = {
            'metadata': {
                'model_name': model_name,
                'num_samples': len(results),
                'processing_date': str(np.datetime64('now'))
            },
            'summary_statistics': {
                'spectral_curvature': {
                    'mean': float(np.mean(all_curvatures)) if all_curvatures else 0,
                    'std': float(np.std(all_curvatures)) if all_curvatures else 0,
                    'min': float(np.min(all_curvatures)) if all_curvatures else 0,
                    'max': float(np.max(all_curvatures)) if all_curvatures else 0
                },
                'thermodynamic_length': {
                    'mean_step': float(np.mean(all_tl_steps)) if all_tl_steps else 0,
                    'std_step': float(np.std(all_tl_steps)) if all_tl_steps else 0,
                    'mean_total': float(np.mean(total_tl_lengths)) if total_tl_lengths else 0,
                    'max_total': float(np.max(total_tl_lengths)) if total_tl_lengths else 0
                },
                'correlation': float(correlation) if 'correlation' in locals() else 0
            },
            'detailed_results': results
        }
        
        with open('/content/enhanced_analysis_results.json', 'w') as f:
            json.dump(output_data, f, indent=2)

        print("\n✅ Results saved to /content/enhanced_analysis_results.json")

        # Create visualizations
        print("\n🎨 Creating visualizations...")
        
        # Create 3D visualizations and animated GIFs
        try:
            create_3d_visualizations(results, save_path='/content/')
            print("✅ 3D visualizations created")
        except Exception as e:
            print(f"❌ Error creating 3D visualizations: {e}")
        
        # Create advanced visualizations
        try:
            create_advanced_visualizations(results, save_path='/content/')
            print("✅ Advanced visualizations created")
        except Exception as e:
            print(f"❌ Error creating advanced visualizations: {e}")
        
        # Create simple comparison plots
        try:
            plt.figure(figsize=(15, 10))
            
            # 1. Layer-wise comparison
            plt.subplot(2, 3, 1)
            if results and results[0]['spectral_curvatures']:
                max_layers = max(len(r['spectral_curvatures']) for r in results)
                layer_curv_means = []
                layer_tl_means = []
                
                for layer_idx in range(max_layers):
                    layer_curvatures = [r['spectral_curvatures'][layer_idx] 
                                      for r in results 
                                      if len(r['spectral_curvatures']) > layer_idx]
                    layer_tl_steps = [r['tl_step_lengths'][layer_idx] 
                                    for r in results 
                                    if len(r['tl_step_lengths']) > layer_idx]
                    
                    layer_curv_means.append(np.mean(layer_curvatures) if layer_curvatures else 0)
                    layer_tl_means.append(np.mean(layer_tl_steps) if layer_tl_steps else 0)
                
                x = range(len(layer_curv_means))
                plt.plot(x, layer_curv_means, 'o-', label='Spectral Curvature', linewidth=2)
                plt.plot(x, layer_tl_means, 's-', label='TL Step Length', linewidth=2)
                plt.xlabel('Layer Index')
                plt.ylabel('Mean Value')
                plt.title('Layer-wise Comparison')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # 2. Distribution comparison
            plt.subplot(2, 3, 2)
            plt.hist(all_curvatures, bins=20, alpha=0.7, label='Spectral Curvature', density=True)
            plt.hist(all_tl_steps, bins=20, alpha=0.7, label='TL Steps', density=True)
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title('Distribution Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 3. Scatter plot
            plt.subplot(2, 3, 3)
            if len(all_curvatures) == len(all_tl_steps):
                plt.scatter(all_curvatures, all_tl_steps, alpha=0.6)
                plt.xlabel('Spectral Curvature')
                plt.ylabel('TL Step Length')
                plt.title('Curvature vs TL Steps')
            
            # 4. Sample-wise total TL
            plt.subplot(2, 3, 4)
            plt.plot(total_tl_lengths, 'o-', alpha=0.7)
            plt.xlabel('Sample Index')
            plt.ylabel('Total Thermodynamic Length')
            plt.title('Total TL per Sample')
            plt.grid(True, alpha=0.3)
            
            # 5. Box plots
            plt.subplot(2, 3, 5)
            data_to_plot = [all_curvatures, all_tl_steps]
            labels = ['Spectral\nCurvature', 'TL Steps']
            plt.boxplot(data_to_plot, labels=labels)
            plt.ylabel('Value')
            plt.title('Value Distributions')
            
            # 6. Cumulative plots
            plt.subplot(2, 3, 6)
            sorted_curvatures = np.sort(all_curvatures)
            sorted_tl_steps = np.sort(all_tl_steps)
            
            y_curv = np.arange(1, len(sorted_curvatures) + 1) / len(sorted_curvatures)
            y_tl = np.arange(1, len(sorted_tl_steps) + 1) / len(sorted_tl_steps)
            
            plt.plot(sorted_curvatures, y_curv, label='Spectral Curvature')
            plt.plot(sorted_tl_steps, y_tl, label='TL Steps')
            plt.xlabel('Value')
            plt.ylabel('Cumulative Probability')
            plt.title('Cumulative Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/content/comparison_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Comparison plots created")
            
        except Exception as e:
            print(f"❌ Error creating comparison plots: {e}")
        
        # Print file summary
        print(f"\n📁 Files created:")
        print(f"  • /content/enhanced_analysis_results.json - Detailed results")
        print(f"  • /content/3d_analysis_static.png - 3D static visualizations")
        print(f"  • /content/3d_analysis_animation.gif - Animated 3D surfaces")
        print(f"  • /content/advanced_analysis.png - Advanced analysis plots")
        print(f"  • /content/comparison_analysis.png - Comparison visualizations")

    else:
        print("❌ No samples processed successfully")
    
    print(f"\n🎉 Enhanced analysis complete!")
    print(f"This analysis computed both spectral curvature and thermodynamic length,")
    print(f"providing insights into the model's computational dynamics across layers.")

# Run the enhanced analysis
if __name__ == "__main__":
    main()