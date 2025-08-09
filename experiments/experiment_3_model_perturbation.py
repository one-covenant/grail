#!/usr/bin/env python3
"""
Experiment 3: Model Perturbation Robustness Test

This experiment demonstrates that even tiny modifications to model weights
destroy the ability to produce correct tokens while maintaining sketch values.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import sys
from copy import deepcopy
sys.path.append('..')
from grail.grail import PRIME_Q, dot_mod_q, r_vec_from_randomness, CHALLENGE_K

class ModelPerturbationExperiment:
    def __init__(self, model_name="sshleifer/tiny-gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Force safetensors to avoid torch.load vulnerability
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True).to(self.device).eval()
        except:
            # Fallback if safetensors not available
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.results = {}
        
    def perturb_model_weights(self, model, layer_idx: int, perturbation_scale: float):
        """Apply controlled perturbation to specific layer weights"""
        perturbed_model = deepcopy(model)
        
        # Find the target layer
        layers = list(perturbed_model.transformer.h)
        if layer_idx < len(layers):
            layer = layers[layer_idx]
            
            # Perturb attention weights
            with torch.no_grad():
                for param_name, param in layer.named_parameters():
                    if 'weight' in param_name:
                        noise = torch.randn_like(param) * perturbation_scale
                        param.data += noise
                        
        return perturbed_model
    
    def compute_model_outputs(self, model, prompt: str, max_length: int = 32):
        """Generate tokens and hidden states from model"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_length,
                return_dict_in_generate=True,
                output_hidden_states=True,
                do_sample=False,  # Deterministic generation
                temperature=1.0
            )
        
        tokens = outputs.sequences[0].tolist()
        
        # Get hidden states
        with torch.no_grad():
            full_outputs = model(outputs.sequences, output_hidden_states=True)
            hidden_states = full_outputs.hidden_states[-1][0]
            
        return tokens, hidden_states
    
    def measure_perturbation_effects(self, perturbation_scales: list):
        """Measure how model perturbations affect tokens and sketches"""
        results = []
        prompt = "The scientific method involves"
        
        # Generate reference outputs
        ref_tokens, ref_hidden = self.compute_model_outputs(self.model, prompt)
        
        # Compute reference sketches
        randomness = "0x" + "c" * 64
        r_vec = r_vec_from_randomness(randomness, self.model.config.hidden_size)
        ref_sketches = [dot_mod_q(ref_hidden[i], r_vec) for i in range(len(ref_tokens))]
        
        for scale in perturbation_scales:
            print(f"\nTesting perturbation scale: {scale}")
            
            # Test perturbations at different layers
            layer_results = []
            
            for layer_idx in [0, len(self.model.transformer.h)//2, -1]:
                if layer_idx == -1:
                    layer_idx = len(self.model.transformer.h) - 1
                    
                # Perturb model
                perturbed_model = self.perturb_model_weights(self.model, layer_idx, scale)
                
                try:
                    # Generate with perturbed model
                    pert_tokens, pert_hidden = self.compute_model_outputs(perturbed_model, prompt)
                    
                    # Compute metrics
                    # Token accuracy
                    min_len = min(len(ref_tokens), len(pert_tokens))
                    token_matches = sum(1 for i in range(min_len) if ref_tokens[i] == pert_tokens[i])
                    token_accuracy = token_matches / len(ref_tokens)
                    
                    # Sketch preservation (for positions with matching tokens)
                    sketch_diffs = []
                    sketch_preserved_count = 0
                    
                    for i in range(min_len):
                        if i < len(pert_hidden):
                            pert_sketch = dot_mod_q(pert_hidden[i], r_vec)
                            diff = abs(pert_sketch - ref_sketches[i])
                            sketch_diffs.append(diff)
                            if diff <= 3:  # Within tolerance
                                sketch_preserved_count += 1
                    
                    # Hidden state divergence
                    hidden_divergences = []
                    for i in range(min(len(ref_hidden), len(pert_hidden))):
                        div = torch.norm(ref_hidden[i] - pert_hidden[i]).item()
                        hidden_divergences.append(div)
                    
                    result = {
                        'layer_idx': layer_idx,
                        'perturbation_scale': scale,
                        'token_accuracy': token_accuracy,
                        'sketch_preservation_rate': sketch_preserved_count / len(ref_sketches),
                        'mean_sketch_diff': np.mean(sketch_diffs) if sketch_diffs else float('inf'),
                        'mean_hidden_divergence': np.mean(hidden_divergences) if hidden_divergences else float('inf'),
                        'sequence_length_ratio': len(pert_tokens) / len(ref_tokens)
                    }
                    
                except Exception as e:
                    print(f"  Generation failed at layer {layer_idx}: {str(e)}")
                    result = {
                        'layer_idx': layer_idx,
                        'perturbation_scale': scale,
                        'token_accuracy': 0,
                        'sketch_preservation_rate': 0,
                        'mean_sketch_diff': float('inf'),
                        'mean_hidden_divergence': float('inf'),
                        'sequence_length_ratio': 0
                    }
                
                layer_results.append(result)
                print(f"  Layer {layer_idx}: token_acc={result['token_accuracy']:.2%}, "
                      f"sketch_preserve={result['sketch_preservation_rate']:.2%}")
            
            results.append({
                'perturbation_scale': scale,
                'layer_results': layer_results,
                'avg_token_accuracy': np.mean([r['token_accuracy'] for r in layer_results]),
                'avg_sketch_preservation': np.mean([r['sketch_preservation_rate'] for r in layer_results])
            })
        
        return results
    
    def test_adversarial_weight_search(self, num_iterations: int = 1000):
        """
        Attempt to find weight perturbations that maintain sketches while
        changing model behavior (simulated adversarial attack).
        """
        prompt = "Machine learning models can"
        target_layer = len(self.model.transformer.h) // 2
        
        # Generate reference
        ref_tokens, ref_hidden = self.compute_model_outputs(self.model, prompt)
        randomness = "0x" + "d" * 64
        r_vec = r_vec_from_randomness(randomness, self.model.config.hidden_size)
        
        # Select challenge positions
        challenge_positions = np.random.choice(len(ref_tokens), 
                                            min(CHALLENGE_K, len(ref_tokens)), 
                                            replace=False)
        # Convert numpy int64 to Python int
        challenge_positions = [int(pos) for pos in challenge_positions]
        ref_sketches = {pos: dot_mod_q(ref_hidden[pos], r_vec) 
                       for pos in challenge_positions}
        
        print(f"\nSearching for adversarial weights...")
        print(f"Challenge positions: {sorted(challenge_positions)}")
        
        # Initialize perturbation
        target_layer_obj = self.model.transformer.h[target_layer]
        weight_shapes = {}
        for name, param in target_layer_obj.named_parameters():
            if 'weight' in name:
                weight_shapes[name] = param.shape
        
        # Try different perturbation strategies
        strategies = ['random', 'gradient_based', 'structured']
        strategy_results = []
        
        for strategy in strategies:
            print(f"\nTrying {strategy} strategy...")
            
            best_result = {
                'strategy': strategy,
                'sketch_preservation': 0,
                'token_difference': 0,
                'found_valid_attack': False
            }
            
            for iteration in range(num_iterations // len(strategies)):
                # Generate perturbation based on strategy
                if strategy == 'random':
                    scale = 0.01 * (1 + iteration / 100)  # Gradually increase
                    perturbed_model = self.perturb_model_weights(self.model, target_layer, scale)
                    
                elif strategy == 'gradient_based':
                    # Simplified gradient-based search
                    perturbed_model = deepcopy(self.model)
                    layer = perturbed_model.transformer.h[target_layer]
                    
                    # Add small targeted perturbations
                    with torch.no_grad():
                        for name, param in layer.named_parameters():
                            if 'weight' in name and np.random.rand() > 0.5:
                                # Selective perturbation
                                mask = torch.rand_like(param) > 0.9
                                param.data[mask] *= (1 + np.random.randn() * 0.1)
                                
                elif strategy == 'structured':
                    # Structured perturbations (e.g., low-rank)
                    perturbed_model = deepcopy(self.model)
                    layer = perturbed_model.transformer.h[target_layer]
                    
                    with torch.no_grad():
                        for name, param in layer.named_parameters():
                            if 'weight' in name and len(param.shape) == 2:
                                # Low-rank perturbation
                                rank = min(param.shape) // 8
                                U = torch.randn(param.shape[0], rank, device=param.device) * 0.01
                                V = torch.randn(rank, param.shape[1], device=param.device) * 0.01
                                param.data += U @ V
                
                try:
                    # Test perturbed model
                    pert_tokens, pert_hidden = self.compute_model_outputs(perturbed_model, prompt)
                    
                    # Check sketch preservation at challenge positions
                    preserved_count = 0
                    for pos in challenge_positions:
                        if pos < len(pert_hidden):
                            pert_sketch = dot_mod_q(pert_hidden[pos], r_vec)
                            if abs(pert_sketch - ref_sketches[pos]) <= 3:
                                preserved_count += 1
                    
                    preservation_rate = preserved_count / len(challenge_positions)
                    
                    # Check token difference
                    token_diff_rate = 1 - sum(1 for i in range(min(len(ref_tokens), len(pert_tokens)))
                                             if ref_tokens[i] == pert_tokens[i]) / len(ref_tokens)
                    
                    # Update best result if this is a successful attack
                    if (preservation_rate > best_result['sketch_preservation'] and 
                        token_diff_rate > 0.1):  # At least 10% token difference
                        best_result['sketch_preservation'] = preservation_rate
                        best_result['token_difference'] = token_diff_rate
                        best_result['found_valid_attack'] = (preservation_rate > 0.8 and 
                                                            token_diff_rate > 0.2)
                        
                except:
                    pass  # Generation failed
                    
                if iteration % 100 == 0:
                    print(f"  Iteration {iteration}: best preservation={best_result['sketch_preservation']:.2%}")
            
            strategy_results.append(best_result)
        
        return strategy_results
    
    def run_all_experiments(self):
        """Run all model perturbation experiments"""
        print("Running Model Perturbation Experiments...")
        
        # Experiment 3a: Basic perturbation effects
        print("\n3a. Testing basic perturbation effects...")
        perturbation_scales = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
        perturbation_results = self.measure_perturbation_effects(perturbation_scales)
        
        # Experiment 3b: Adversarial weight search
        print("\n3b. Testing adversarial weight search...")
        adversarial_results = self.test_adversarial_weight_search(num_iterations=1000)
        
        # Save results
        self.results = {
            'perturbation_effects': perturbation_results,
            'adversarial_search': adversarial_results
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/experiment_3_model_perturbation.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate plots
        self._generate_plots()
        
    def _generate_plots(self):
        """Generate visualization plots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot 1: Perturbation effects
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        pert_data = self.results['perturbation_effects']
        scales = [r['perturbation_scale'] for r in pert_data]
        token_acc = [r['avg_token_accuracy'] for r in pert_data]
        sketch_preserve = [r['avg_sketch_preservation'] for r in pert_data]
        
        # Token accuracy
        ax1.semilogx(scales, token_acc, 'bo-', markersize=8, linewidth=2, label='Token Accuracy')
        ax1.semilogx(scales, sketch_preserve, 'ro-', markersize=8, linewidth=2, label='Sketch Preservation')
        ax1.set_xlabel('Perturbation Scale')
        ax1.set_ylabel('Rate')
        ax1.set_title('Model Perturbation Effects')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # Adversarial search results
        adv_data = self.results['adversarial_search']
        strategies = [r['strategy'] for r in adv_data]
        sketch_rates = [r['sketch_preservation'] for r in adv_data]
        token_diffs = [r['token_difference'] for r in adv_data]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        ax2.bar(x - width/2, sketch_rates, width, label='Sketch Preservation', alpha=0.7)
        ax2.bar(x + width/2, token_diffs, width, label='Token Difference', alpha=0.7)
        
        ax2.set_xlabel('Attack Strategy')
        ax2.set_ylabel('Rate')
        ax2.set_title('Adversarial Weight Search Results')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1.05)
        
        # Add text indicating if valid attacks were found
        for i, r in enumerate(adv_data):
            if r['found_valid_attack']:
                ax2.text(i, 0.95, '✓', ha='center', va='center', fontsize=20, color='green')
            else:
                ax2.text(i, 0.95, '✗', ha='center', va='center', fontsize=20, color='red')
        
        plt.tight_layout()
        plt.savefig('results/experiment_3_perturbation_analysis.png', dpi=150)
        plt.close()

if __name__ == "__main__":
    experiment = ModelPerturbationExperiment()
    experiment.run_all_experiments()
    print("\nExperiment 3 completed. Results saved to results/")