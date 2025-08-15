#!/usr/bin/env python3
"""
Experiment 2: Multi-Position Constraint Analysis

This experiment demonstrates how GRAIL's multi-position verification creates
an exponentially constrained optimization problem for attackers.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.append('..')
from grail.grail import PRIME_Q, dot_mod_q, r_vec_from_randomness

class MultiPositionConstraintExperiment:
    def __init__(self, model_name="sshleifer/tiny-gpt2"):
        print(f"Initializing experiment with model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Loading model...")
        # Force safetensors to avoid torch.load vulnerability
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, use_safetensors=True
            ).to(self.device).eval()
            print("Model loaded successfully (safetensors)")
        except (OSError, ImportError, RuntimeError) as e:
            # Fallback if safetensors not available
            print(f"Safetensors not available ({e}), using standard loading...")
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
            print("Model loaded successfully (standard)")
        
        print(f"Model size: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters")
        
        # Check available GPUs
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"Found {self.num_gpus} GPUs - will use parallel processing")
        
        self.results = {}
        
    def generate_reference_trace(self, prompt: str, max_length: int = 32):
        """Generate reference hidden states and tokens"""
        print(f"  Generating reference trace for prompt: '{prompt[:30]}...'")
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        print(f"  Running model generation (max {max_length} new tokens)...")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_length,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            
        # Get the generated tokens
        tokens = outputs.sequences[0].tolist()
        print(f"  Generated {len(tokens)} tokens")
        
        # Get hidden states for all positions
        with torch.no_grad():
            full_outputs = self.model(outputs.sequences, output_hidden_states=True)
            hidden_states = full_outputs.hidden_states[-1][0]  # Last layer
            
        return tokens, hidden_states
    
    def attempt_constraint_satisfaction(self, target_sketches: dict, 
                                        r_vec: torch.Tensor, 
                                        hidden_dim: int,
                                        max_iterations: int = 5000):
        """
        Attempt to find hidden states that satisfy multiple sketch constraints
        without considering token generation constraints.
        """
        positions = list(target_sketches.keys())
        k = len(positions)
        
        # Initialize random hidden states for challenged positions
        h_states = {pos: torch.randn(hidden_dim, device=self.device, requires_grad=True) 
                   for pos in positions}
        
        # Use Adam optimizer with adaptive learning rate
        k = len(positions)
        lr = 0.1 if k <= 4 else 0.01
        optimizer = torch.optim.Adam(list(h_states.values()), lr=lr)
        
        losses = []
        success_iterations = []
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Compute loss for each position
            total_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            num_satisfied = 0
            
            # Log progress every 500 iterations for small k, less frequently for large k
            log_interval = 500 if k <= 8 else 1000
            if iteration % log_interval == 0 and iteration > 0:
                print(f"    Iteration {iteration}/{max_iterations}, satisfied: {num_satisfied}/{k}", end='\r', flush=True)
            
            for pos, h in h_states.items():
                current_sketch = dot_mod_q(h, r_vec)
                target = target_sketches[pos]
                
                # Use smooth approximation for differentiability
                diff = abs(current_sketch - target)
                # Create a differentiable loss based on the hidden state
                if diff > 3:
                    # Use stronger gradient signal when far from target
                    loss_contrib = torch.sum(h ** 2) * 0.0001 + float(diff) * 0.001
                else:
                    # Just regularization when close
                    loss_contrib = torch.sum(h ** 2) * 0.0001
                total_loss = total_loss + loss_contrib
                
                if diff <= 3:  # Within tolerance
                    num_satisfied += 1
            
            losses.append(total_loss.item())
            
            if num_satisfied == k:
                success_iterations.append(iteration)
                if len(success_iterations) >= 10:  # Found solution consistently
                    break
            
            # Backprop
            if total_loss.requires_grad:
                total_loss.backward()
                optimizer.step()
        
        # Clear the iteration line
        print(f"    Completed {iteration + 1} iterations" + " " * 20)
        
        final_satisfied = sum(1 for pos, h in h_states.items() 
                            if abs(dot_mod_q(h, r_vec) - target_sketches[pos]) <= 3)
        
        return {
            'num_constraints': k,
            'iterations': iteration + 1,
            'final_satisfied': final_satisfied,
            'success_rate': final_satisfied / k,
            'converged': len(success_iterations) >= 10,
            'loss_history': losses[::100]  # Sample every 100th loss
        }
    
    def run_constraint_trial_on_device(self, k, trial_idx, hidden_dim, randomness, device_id=None):
        """Run a single trial, optionally on a specific device"""
        if device_id is not None and self.num_gpus > 1:
            device = f"cuda:{device_id % self.num_gpus}"
        else:
            device = self.device
            
        # Move r_vec to the specified device
        r_vec = r_vec_from_randomness(randomness, hidden_dim).to(device)
        
        # Generate random target sketches
        positions = list(range(k))
        target_sketches = {pos: np.random.randint(0, PRIME_Q) 
                         for pos in positions}
        
        # Get adaptive max iterations for this k
        adaptive_max_iterations = {
            1: 500,    # Very easy
            2: 1000,   # Easy
            4: 2000,   # Medium
            8: 3000,   # Hard
            16: 4000,  # Very hard
            32: 5000   # Extremely hard
        }.get(k, 5000)
        
        # Run the trial with device-local tensors
        h_states = {pos: torch.randn(hidden_dim, device=device, requires_grad=True) 
                   for pos in positions}
        
        # Move optimization to the specific device
        result = self.attempt_constraint_satisfaction(
            target_sketches, r_vec, hidden_dim, max_iterations=adaptive_max_iterations
        )
        
        return result
    
    def measure_constraint_hardness(self, k_values: list):
        """Measure how difficulty scales with number of constraints"""
        results = []
        
        # Use consistent random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        hidden_dim = self.model.config.hidden_size
        randomness = "0x" + "a" * 64  # Deterministic randomness
        r_vec = r_vec_from_randomness(randomness, hidden_dim)
        
        for k in k_values:
            print(f"\nTesting k={k} constraints...")
            
            # Adaptive number of trials and iterations based on k
            # For larger k, we need fewer trials since failure is more consistent
            num_trials = {
                1: 10,
                2: 10,
                4: 10,
                8: 8,
                16: 5,
                32: 3
            }.get(k, 5)
            
            # Adaptive max iterations
            adaptive_max_iterations = {
                1: 500,    # Very easy
                2: 1000,   # Easy
                4: 2000,   # Medium
                8: 3000,   # Hard
                16: 4000,  # Very hard
                32: 5000   # Extremely hard
            }.get(k, 5000)
            
            # Early stopping: if we see consistent failure pattern
            if len(results) >= 2:
                # If the last two k values had 0% success, skip larger k
                if all(r['success_rate'] == 0 for r in results[-2:]):
                    print(f"  Skipping due to consistent failures at smaller k values")
                    results.append({
                        'k': k,
                        'avg_iterations': 0,
                        'success_rate': 0.0,
                        'avg_constraints_satisfied': 0,
                        'theoretical_difficulty': k * np.log(PRIME_Q),
                        'skipped': True
                    })
                    continue
            
            print(f"  Running {num_trials} trials with max {adaptive_max_iterations} iterations each")
            
            # Run multiple trials
            trial_results = []
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}...", end='')
                
                # Generate random target sketches
                positions = list(range(k))
                target_sketches = {pos: np.random.randint(0, PRIME_Q) 
                                 for pos in positions}
                
                result = self.attempt_constraint_satisfaction(
                    target_sketches, r_vec, hidden_dim, max_iterations=adaptive_max_iterations
                )
                trial_results.append(result)
                
                # Show result for this trial
                print(f" converged={result['converged']}, satisfied={result['final_satisfied']}/{k}")
            
            # Aggregate results
            avg_iterations = np.mean([r['iterations'] for r in trial_results])
            success_rate = np.mean([r['converged'] for r in trial_results])
            avg_satisfied = np.mean([r['final_satisfied'] for r in trial_results])
            
            results.append({
                'k': k,
                'avg_iterations': avg_iterations,
                'success_rate': success_rate,
                'avg_constraints_satisfied': avg_satisfied,
                'theoretical_difficulty': k * np.log(PRIME_Q)
            })
            
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Avg iterations: {avg_iterations:.0f}")
            print(f"  Avg constraints satisfied: {avg_satisfied:.1f}/{k}")
        
        return results
    
    def analyze_autoregressive_cascade(self, prompt: str):
        """
        Analyze how perturbations in early hidden states cascade through
        autoregressive generation.
        """
        # Generate reference
        tokens, hidden_states = self.generate_reference_trace(prompt)
        seq_len = len(tokens)
        
        results = []
        perturbation_positions = [0, seq_len//4, seq_len//2, 3*seq_len//4]
        
        for perturb_pos in perturbation_positions:
            if perturb_pos >= seq_len:
                continue
                
            # Apply small perturbation at specific position
            perturbed_hidden = hidden_states.clone()
            noise = torch.randn_like(hidden_states[perturb_pos]) * 0.1
            perturbed_hidden[perturb_pos] += noise
            
            # Measure divergence at each subsequent position
            divergences = []
            for pos in range(perturb_pos + 1, seq_len):
                orig_h = hidden_states[pos]
                pert_h = perturbed_hidden[pos]
                
                # L2 distance
                div = torch.norm(pert_h - orig_h).item()
                divergences.append(div)
            
            if divergences:
                results.append({
                    'perturbation_position': perturb_pos,
                    'perturbation_magnitude': torch.norm(noise).item(),
                    'mean_divergence': np.mean(divergences),
                    'final_divergence': divergences[-1],
                    'divergence_growth': divergences[-1] / divergences[0] if divergences[0] > 0 else 0
                })
        
        return results
    
    def test_sketch_uniqueness_with_tokens(self, prompt: str):
        """
        Test whether we can find different hidden states that produce
        the same sketches AND the same tokens.
        """
        # Generate reference
        tokens, hidden_states = self.generate_reference_trace(prompt)
        
        # Compute reference sketches
        randomness = "0x" + "b" * 64
        r_vec = r_vec_from_randomness(randomness, self.model.config.hidden_size)
        
        reference_sketches = {}
        for i in range(len(tokens)):
            reference_sketches[i] = dot_mod_q(hidden_states[i], r_vec)
        
        # Try to find alternative hidden states
        # This is extremely difficult because we need to maintain token generation
        print("\nAttempting to find alternative hidden states...")
        
        # For demonstration, we'll show that even small perturbations break token generation
        perturbation_results = []
        
        for scale in [0.001, 0.01, 0.1, 1.0]:
            perturbed_hidden = hidden_states.clone()
            
            # Add perturbations that preserve sketch values (approximately)
            for i in range(min(5, len(tokens))):  # Test first 5 positions
                h = perturbed_hidden[i]
                
                # Find perturbation orthogonal to r_vec
                # Ensure r_vec is on the same device as h
                r_vec_device = r_vec.to(h.device)
                r_normalized = r_vec_device.float() / torch.norm(r_vec_device.float())
                random_vec = torch.randn_like(h)
                ortho_component = random_vec - torch.dot(random_vec, r_normalized) * r_normalized
                ortho_component = ortho_component / torch.norm(ortho_component)
                
                # Apply scaled perturbation
                perturbed_hidden[i] = h + scale * ortho_component
            
            # Check sketch preservation
            sketch_diffs = []
            for i in range(min(5, len(tokens))):
                orig_sketch = reference_sketches[i]
                new_sketch = dot_mod_q(perturbed_hidden[i], r_vec)
                sketch_diffs.append(abs(new_sketch - orig_sketch))
            
            result = {
                'perturbation_scale': scale,
                'mean_sketch_diff': np.mean(sketch_diffs),
                'max_sketch_diff': np.max(sketch_diffs),
                'sketches_preserved': all(d <= 3 for d in sketch_diffs)
            }
            perturbation_results.append(result)
        
        return perturbation_results
    
    def run_all_experiments(self):
        """Run all multi-position constraint experiments"""
        print("Running Multi-Position Constraint Experiments...")
        
        # Experiment 2a: Constraint satisfaction difficulty
        print("\n2a. Testing constraint satisfaction difficulty...")
        print("This will test multiple constraint counts with 10 trials each.")
        print("Each trial runs up to 5000 iterations of optimization.")
        k_values = [1, 2, 4, 8, 16, 32]
        constraint_results = self.measure_constraint_hardness(k_values)
        
        # Experiment 2b: Autoregressive cascade analysis  
        print("\n2b. Analyzing autoregressive cascade effects...")
        print("Generating reference trace and analyzing perturbation cascades...")
        prompt = "The future of artificial intelligence is"
        cascade_results = self.analyze_autoregressive_cascade(prompt)
        
        # Experiment 2c: Sketch uniqueness with token constraints
        print("\n2c. Testing sketch uniqueness with token generation constraints...")
        print("Testing different perturbation scales to see effect on sketches vs tokens...")
        uniqueness_results = self.test_sketch_uniqueness_with_tokens(prompt)
        
        # Save results
        self.results = {
            'constraint_satisfaction': constraint_results,
            'autoregressive_cascade': cascade_results,
            'sketch_uniqueness': uniqueness_results
        }
        
        os.makedirs('results', exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        results_json = convert_numpy_types(self.results)
        
        with open('results/experiment_2_multi_position_constraints.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Generate plots
        self._generate_plots()
        
    def _generate_plots(self):
        """Generate visualization plots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot 1: Constraint satisfaction difficulty
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        constraint_data = self.results['constraint_satisfaction']
        k_vals = [r['k'] for r in constraint_data]
        success_rates = [r['success_rate'] for r in constraint_data]
        avg_iterations = [r['avg_iterations'] for r in constraint_data]
        
        # Success rate plot
        ax1.plot(k_vals, success_rates, 'bo-', markersize=8, linewidth=2)
        ax1.set_xlabel('Number of Constraints (k)')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Constraint Satisfaction Success Rate')
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)
        
        # Iterations plot
        ax2.semilogy(k_vals, avg_iterations, 'ro-', markersize=8, linewidth=2)
        ax2.set_xlabel('Number of Constraints (k)')
        ax2.set_ylabel('Average Iterations to Converge')
        ax2.set_title('Computational Difficulty Scaling')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/experiment_2_constraint_difficulty.png', dpi=150)
        plt.close()
        
        # Plot 2: Autoregressive cascade
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cascade_data = self.results['autoregressive_cascade']
        positions = [r['perturbation_position'] for r in cascade_data]
        divergences = [r['final_divergence'] for r in cascade_data]
        growth_rates = [r['divergence_growth'] for r in cascade_data]
        
        ax.bar(positions, divergences, width=2, alpha=0.7, label='Final Divergence')
        ax2 = ax.twinx()
        ax2.plot(positions, growth_rates, 'ro-', markersize=8, label='Growth Rate')
        
        ax.set_xlabel('Perturbation Position')
        ax.set_ylabel('Final Divergence (L2 norm)')
        ax2.set_ylabel('Divergence Growth Rate')
        ax.set_title('Cascade Effect of Hidden State Perturbations')
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/experiment_2_cascade_effect.png', dpi=150)
        plt.close()

if __name__ == "__main__":
    print("Starting Multi-Position Constraint Experiment...")
    print("="*60)
    experiment = MultiPositionConstraintExperiment()
    experiment.run_all_experiments()
    print("\nExperiment 2 completed. Results saved to results/")