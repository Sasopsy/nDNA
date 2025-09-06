import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import numpy as np
from typing import List, Dict
import json
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def setup_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    return tokenizer, model

class ThermodynamicLengthCalculator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def get_lm_head(self):
        head_attrs = ['lm_head', 'head', 'classifier']
        
        for attr in head_attrs:
            if hasattr(self.model, attr):
                head = getattr(self.model, attr)
                if isinstance(head, nn.Linear):
                    return head
        
        vocab_size = len(self.tokenizer)
        hidden_size = self.model.config.hidden_size if hasattr(self.model, 'config') else 768
        return nn.Linear(hidden_size, vocab_size)
    
    def apply_logit_lens(self, hidden_states_sequence: List[torch.Tensor]) -> List[torch.Tensor]:
        lm_head = self.get_lm_head()
        lm_head = lm_head.to(next(self.model.parameters()).device)
        
        probability_distributions = []
        
        with torch.no_grad():
            for hidden_state in hidden_states_sequence:
                hidden_state = hidden_state.to(next(self.model.parameters()).device)
                logits = lm_head(hidden_state)
                probs = F.softmax(logits, dim=-1)
                probability_distributions.append(probs.cpu())
        
        return probability_distributions
    
    def kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        log_q = torch.log(q + 1e-8)
        kl_div = F.kl_div(log_q, p, reduction='none', log_target=False)
        return torch.sum(kl_div, dim=-1)
    
    def calculate_thermodynamic_length(self, hidden_states_sequence: List[torch.Tensor]) -> Dict:
        prob_distributions = self.apply_logit_lens(hidden_states_sequence)
        
        if len(prob_distributions) < 2:
            return {'total_thermodynamic_length': 0.0, 'step_lengths_array': []}
        
        step_lengths_array = []
        total_length = 0.0
        
        for layer_idx in range(len(prob_distributions) - 1):
            p_curr = prob_distributions[layer_idx]
            p_next = prob_distributions[layer_idx + 1]
            
            kl_divs = self.kl_divergence(p_curr, p_next)
            avg_kl = torch.mean(kl_divs).item()
            step_length = np.sqrt(2 * avg_kl) if avg_kl > 0 else 0.0
            
            step_lengths_array.append(step_length)
            total_length += step_length
        
        return {
            'total_thermodynamic_length': total_length,
            'step_lengths_array': step_lengths_array
        }

class AnalysisCalculator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.tl_calculator = ThermodynamicLengthCalculator(model, tokenizer)

    def get_activations(self, text: str, max_length: int = 256) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        activations = {}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                for i, hidden_state in enumerate(outputs.hidden_states):
                    activations[f'layer_{i}'] = hidden_state.cpu().detach()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return activations

    def compute_spectral_curvature(self, activations_sequence: List[torch.Tensor]) -> List[float]:
        curvatures = []
        n_layers = len(activations_sequence)

        for i in range(1, n_layers - 1):
            h_t_minus_1 = activations_sequence[i - 1]
            h_t = activations_sequence[i]
            h_t_plus_1 = activations_sequence[i + 1]

            delta_h_t = h_t - h_t_minus_1
            delta_h_t_plus_1 = h_t_plus_1 - h_t
            delta_2_h_t = delta_h_t_plus_1 - delta_h_t

            numerator = torch.norm(delta_2_h_t, dim=-1)
            denominator = torch.norm(delta_h_t_plus_1 + delta_h_t, dim=-1) + 1e-8

            curvature = numerator / denominator
            curvatures.append(curvature.mean().item())

        return curvatures

    def process_sample(self, sample: Dict, max_length: int = 256) -> Dict:
        context = sample['context'][:500]
        question = sample['question'][:100]
        text = f"Context: {context} Question: {question}"

        activations = self.get_activations(text, max_length)
        sorted_layers = sorted(activations.keys(), key=lambda x: int(x.split('_')[1]))
        activation_sequence = [activations[layer] for layer in sorted_layers]

        curvatures = self.compute_spectral_curvature(activation_sequence)
        tl_results = self.tl_calculator.calculate_thermodynamic_length(activation_sequence)

        return {
            'spectral_curvatures': curvatures,
            'thermodynamic_length': tl_results['total_thermodynamic_length'],
            'tl_step_lengths': tl_results['step_lengths_array']
        }

def create_visualization(results: List[Dict]):
    if not results:
        return
    
    max_layers = max(len(r['spectral_curvatures']) for r in results if r['spectral_curvatures'])
    curvature_matrix = np.zeros((len(results), max_layers))
    
    for i, result in enumerate(results):
        curvatures = result['spectral_curvatures']
        if curvatures:
            curvature_matrix[i, :len(curvatures)] = curvatures
    
    X, Y = np.meshgrid(range(max_layers), range(len(results)))
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, curvature_matrix, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Sample')
    ax.set_zlabel('Curvature')
    ax.set_title('Spectral Curvature')
    fig.colorbar(surf, shrink=0.5)
    
    plt.tight_layout()
    plt.savefig('analysis_result.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer, model = setup_model(model_name)
    calculator = AnalysisCalculator(tokenizer, model)
    
    dataset = load_dataset("squad", split="validation[:25]")
    results = []

    for i, sample in enumerate(tqdm(dataset, desc="Processing")):
        try:
            result = calculator.process_sample(sample, max_length=128)
            results.append(result)

            if i % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    if results:
        all_curvatures = []
        all_tl_steps = []
        total_tl_lengths = []
        
        for result in results:
            all_curvatures.extend(result['spectral_curvatures'])
            all_tl_steps.extend(result['tl_step_lengths'])
            total_tl_lengths.append(result['thermodynamic_length'])
        
        print(f"Processed {len(results)} samples")
        print(f"Mean curvature: {np.mean(all_curvatures):.4f}")
        print(f"Mean TL: {np.mean(total_tl_lengths):.4f}")
        
        create_visualization(results)
        
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()