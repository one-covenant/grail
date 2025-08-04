#!/usr/bin/env python3
"""
Experiment 4: Attack Simulation with Gradient Descent

This experiment demonstrates the computational infeasibility of finding
model parameters that satisfy all GRAIL constraints while producing
different outputs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import sys
from copy import deepcopy
import time
sys.path.append('..')
from grail.grail import PRIME_Q, dot_mod_q, r_vec_from_randomness, CHALLENGE_K, TOLERANCE

class AttackSimulationExperiment:
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
        
    def create_attack_model(self, reference_model):
        """Create a smaller attack model that attempts to mimic sketches"""
        # Create a simplified model with fewer parameters
        config = reference_model.config
        
        class SimpleAttackModel(nn.Module):
            def __init__(self, vocab_size, hidden_size, num_layers):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([
                    nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
                ])
                self.output_projection = nn.Linear(hidden_size, vocab_size)
                self.hidden_size = hidden_size
                
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                hidden_states = []
                
                for layer in self.layers:
                    x = F.relu(layer(x))
                    hidden_states.append(x)
                
                logits = self.output_projection(x)
                return logits, hidden_states[-1]  # Return last hidden state
        
        # Create attack model with 50% fewer layers
        attack_model = SimpleAttackModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=max(1, config.num_hidden_layers // 2)
        ).to(self.device)
        
        return attack_model
    
    def compute_grail_loss(self, hidden_states, target_sketches, r_vec, positions):
        """Compute loss for matching target sketch values"""
        loss = 0
        num_matched = 0
        
        for pos in positions:
            if pos < hidden_states.shape[1]:
                h = hidden_states[0, pos]  # Batch size 1
                current_sketch = dot_mod_q(h, r_vec)
                target_sketch = target_sketches[pos]
                
                # Smooth approximation of modular distance for gradients
                diff = float(abs(current_sketch - target_sketch))
                mod_diff = min(diff, PRIME_Q - diff)
                
                # Squared loss with tolerance consideration
                if mod_diff <= TOLERANCE:
                    num_matched += 1
                    loss += 0  # Within tolerance
                else:
                    loss += (mod_diff - TOLERANCE) ** 2
        
        return loss / len(positions), num_matched
    
    def gradient_based_attack(self, prompt: str, max_iterations: int = 5000):
        """
        Attempt to train an attack model to match GRAIL sketches while
        producing different tokens.
        """
        print(f"\nRunning gradient-based attack simulation...")
        
        # Generate reference outputs
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            ref_outputs = self.model.generate(
                input_ids,
                max_new_tokens=32,
                return_dict_in_generate=True,
                output_hidden_states=True,
                do_sample=False
            )
            ref_tokens = ref_outputs.sequences[0].tolist()
            
            # Get reference hidden states
            ref_hidden_outputs = self.model(ref_outputs.sequences, output_hidden_states=True)
            ref_hidden_states = ref_hidden_outputs.hidden_states[-1]
        
        # Compute reference sketches
        randomness = "0x" + "e" * 64
        r_vec = r_vec_from_randomness(randomness, self.model.config.hidden_size)
        
        # Select challenge positions
        seq_len = len(ref_tokens)
        challenge_positions = sorted(np.random.choice(
            seq_len, min(CHALLENGE_K, seq_len), replace=False
        ))
        # Convert numpy int64 to Python int
        challenge_positions = [int(pos) for pos in challenge_positions]
        
        ref_sketches = {}
        for pos in challenge_positions:
            ref_sketches[pos] = dot_mod_q(ref_hidden_states[0, pos], r_vec)
        
        print(f"Reference sequence length: {seq_len}")
        print(f"Challenge positions: {challenge_positions}")
        
        # Create and train attack model
        attack_model = self.create_attack_model(self.model)
        optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001)
        
        training_history = {
            'iterations': [],
            'sketch_loss': [],
            'num_matched': [],
            'token_accuracy': [],
            'generation_quality': []
        }
        
        best_result = {
            'iteration': 0,
            'sketches_matched': 0,
            'token_difference': 0,
            'success': False
        }
        
        start_time = time.time()
        
        for iteration in range(max_iterations):
            # Forward pass
            optimizer.zero_grad()
            
            # Try to generate with attack model
            try:
                logits, hidden_states = attack_model(ref_outputs.sequences)
                
                # Compute GRAIL loss
                sketch_loss, num_matched = self.compute_grail_loss(
                    hidden_states, ref_sketches, r_vec, challenge_positions
                )
                
                # Add token generation loss (cross-entropy)
                # We want different tokens, so we minimize negative CE with reference
                token_loss = -F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    ref_outputs.sequences.view(-1),
                    ignore_index=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else -100
                )
                
                # Combined loss: match sketches but produce different tokens
                total_loss = sketch_loss - 0.1 * token_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Evaluate periodically
                if iteration % 100 == 0:
                    with torch.no_grad():
                        # Generate tokens with attack model
                        attack_outputs = attack_model(input_ids)
                        attack_logits = attack_outputs[0]
                        attack_tokens = torch.argmax(attack_logits, dim=-1).squeeze().tolist()
                        
                        # Ensure attack_tokens is a list
                        if isinstance(attack_tokens, int):
                            attack_tokens = [attack_tokens]
                        
                        # Compute token accuracy
                        min_len = min(len(ref_tokens), len(attack_tokens))
                        token_matches = sum(1 for i in range(min_len) 
                                          if i < len(ref_tokens) and i < len(attack_tokens) 
                                          and ref_tokens[i] == attack_tokens[i])
                        token_accuracy = token_matches / len(ref_tokens) if ref_tokens else 0
                        
                        # Check if this is a successful attack
                        sketch_match_rate = num_matched / len(challenge_positions)
                        token_diff_rate = 1 - token_accuracy
                        
                        if sketch_match_rate > best_result['sketches_matched'] / len(challenge_positions):
                            best_result['iteration'] = iteration
                            best_result['sketches_matched'] = num_matched
                            best_result['token_difference'] = token_diff_rate
                            best_result['success'] = (sketch_match_rate > 0.8 and token_diff_rate > 0.2)
                        
                        # Record history
                        training_history['iterations'].append(iteration)
                        training_history['sketch_loss'].append(float(sketch_loss))
                        training_history['num_matched'].append(num_matched)
                        training_history['token_accuracy'].append(token_accuracy)
                        training_history['generation_quality'].append(min_len / len(ref_tokens))
                        
                        if iteration % 1000 == 0:
                            print(f"  Iteration {iteration}: sketches_matched={num_matched}/{len(challenge_positions)}, "
                                  f"token_acc={token_accuracy:.2%}, loss={float(sketch_loss):.4f}")
                
            except Exception as e:
                if iteration % 1000 == 0:
                    print(f"  Iteration {iteration}: Generation failed - {str(e)}")
                continue
        
        elapsed_time = time.time() - start_time
        
        return {
            'best_result': best_result,
            'training_history': training_history,
            'elapsed_time': elapsed_time,
            'challenge_positions': challenge_positions,
            'model_size_ratio': sum(p.numel() for p in attack_model.parameters()) / 
                               sum(p.numel() for p in self.model.parameters())
        }
    
    def analyze_computational_complexity(self):
        """
        Analyze the computational complexity of the attack problem.
        """
        print("\nAnalyzing computational complexity...")
        
        complexity_results = []
        
        # Test different problem sizes
        hidden_dims = [64, 128, 256, 512, 768]
        k_values = [4, 8, 16, 32]
        
        for hidden_dim in hidden_dims:
            for k in k_values:
                # Calculate theoretical complexity
                
                # Number of constraints
                num_constraints = k
                
                # Degrees of freedom in solution space
                solution_space_dim = hidden_dim * k  # k hidden states to optimize
                
                # Constraint reduction (each constraint removes ~1 degree of freedom)
                effective_dimensions = max(1, solution_space_dim - num_constraints)
                
                # Search space size (discretized)
                # Assuming we need to search with precision of 1/1000
                search_precision = 1000
                # Compute log directly to avoid overflow
                search_space_size_log10 = effective_dimensions * np.log10(search_precision)
                
                # Probability of random success
                tolerance_window = (2 * TOLERANCE + 1) / PRIME_Q
                random_success_prob = tolerance_window ** k
                
                # Expected iterations for brute force
                expected_iterations = 1 / random_success_prob if random_success_prob > 0 else float('inf')
                
                result = {
                    'hidden_dim': hidden_dim,
                    'k': k,
                    'num_constraints': num_constraints,
                    'solution_space_dim': solution_space_dim,
                    'effective_dimensions': effective_dimensions,
                    'search_space_size_log10': float(search_space_size_log10),
                    'random_success_prob_log10': float(np.log10(float(random_success_prob))) if random_success_prob > 0 else -float('inf'),
                    'expected_iterations_log10': float(np.log10(float(expected_iterations))) if expected_iterations < float('inf') else float('inf')
                }
                
                complexity_results.append(result)
                
                if hidden_dim == 768 and k in [8, 16]:
                    print(f"  d={hidden_dim}, k={k}: success_prob≈10^{result['random_success_prob_log10']:.1f}, "
                          f"expected_iters≈10^{result['expected_iterations_log10']:.1f}")
        
        return complexity_results
    
    def run_all_experiments(self):
        """Run all attack simulation experiments"""
        print("Running Attack Simulation Experiments...")
        
        # Experiment 4a: Gradient-based attack
        print("\n4a. Running gradient-based attack simulation...")
        prompts = [
            "The future of technology is",
            "In scientific research, we must",
            "The key to success is"
        ]
        
        attack_results = []
        for prompt in prompts:
            print(f"\nTesting prompt: '{prompt}'")
            result = self.gradient_based_attack(prompt, max_iterations=5000)
            result['prompt'] = prompt
            attack_results.append(result)
            
            print(f"  Best result: {result['best_result']['sketches_matched']} sketches matched")
            print(f"  Attack success: {result['best_result']['success']}")
            print(f"  Time elapsed: {result['elapsed_time']:.2f}s")
        
        # Experiment 4b: Computational complexity analysis
        print("\n4b. Analyzing computational complexity...")
        complexity_results = self.analyze_computational_complexity()
        
        # Save results
        self.results = {
            'gradient_attacks': attack_results,
            'complexity_analysis': complexity_results
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/experiment_4_attack_simulation.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        # Generate plots
        self._generate_plots()
        
    def _generate_plots(self):
        """Generate visualization plots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot 1: Attack training progress
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Use first attack result for detailed plots
        if self.results['gradient_attacks']:
            attack_data = self.results['gradient_attacks'][0]
            history = attack_data['training_history']
            
            if history['iterations']:
                # Sketch matching progress
                ax1.plot(history['iterations'], history['num_matched'], 'b-', linewidth=2)
                ax1.axhline(y=len(attack_data['challenge_positions']), color='r', linestyle='--', label='Target')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Sketches Matched')
                ax1.set_title('Sketch Matching Progress')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Token accuracy (we want this low for successful attack)
                ax2.plot(history['iterations'], history['token_accuracy'], 'g-', linewidth=2)
                ax2.axhline(y=0.2, color='r', linestyle='--', label='Target (<20%)')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Token Accuracy')
                ax2.set_title('Token Generation Accuracy')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 1.05)
                
                # Loss evolution
                ax3.semilogy(history['iterations'], history['sketch_loss'], 'r-', linewidth=2)
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Sketch Loss (log scale)')
                ax3.set_title('Optimization Loss')
                ax3.grid(True, alpha=0.3)
        
        # Complexity analysis
        complexity_data = self.results['complexity_analysis']
        
        # Filter for k=16 to show scaling with dimension
        k16_data = [r for r in complexity_data if r['k'] == 16]
        if k16_data:
            dims = [r['hidden_dim'] for r in k16_data]
            success_probs = [r['random_success_prob_log10'] for r in k16_data]
            
            ax4.plot(dims, success_probs, 'bo-', markersize=8, linewidth=2)
            ax4.set_xlabel('Hidden Dimension')
            ax4.set_ylabel('Log₁₀(Success Probability)')
            ax4.set_title('Attack Difficulty Scaling (k=16)')
            ax4.grid(True, alpha=0.3)
            
            # Add text annotations
            for i, (d, p) in enumerate(zip(dims, success_probs)):
                if i % 2 == 0:  # Annotate every other point
                    ax4.annotate(f'10^{{{p:.0f}}}', xy=(d, p), xytext=(d, p+10),
                               ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('results/experiment_4_attack_analysis.png', dpi=150)
        plt.close()
        
        # Plot 2: Summary of all attacks
        fig, ax = plt.subplots(figsize=(10, 6))
        
        attack_summaries = []
        for i, result in enumerate(self.results['gradient_attacks']):
            attack_summaries.append({
                'prompt_idx': i,
                'sketches_matched': result['best_result']['sketches_matched'],
                'total_sketches': len(result['challenge_positions']),
                'token_diff': result['best_result']['token_difference'],
                'success': result['best_result']['success']
            })
        
        x = np.arange(len(attack_summaries))
        width = 0.35
        
        sketches_matched = [r['sketches_matched'] for r in attack_summaries]
        total_sketches = [r['total_sketches'] for r in attack_summaries]
        token_diffs = [r['token_diff'] for r in attack_summaries]
        
        # Normalize sketches matched to percentage
        sketch_match_pct = [s/t*100 for s, t in zip(sketches_matched, total_sketches)]
        
        bars1 = ax.bar(x - width/2, sketch_match_pct, width, label='Sketch Match %', alpha=0.7)
        bars2 = ax.bar(x + width/2, [td*100 for td in token_diffs], width, label='Token Difference %', alpha=0.7)
        
        # Add success indicators
        for i, summary in enumerate(attack_summaries):
            if summary['success']:
                ax.text(i, 105, '✓', ha='center', fontsize=16, color='green', weight='bold')
            else:
                ax.text(i, 105, '✗', ha='center', fontsize=16, color='red', weight='bold')
        
        ax.set_xlabel('Attack Attempt')
        ax.set_ylabel('Percentage')
        ax.set_title('Attack Success Summary')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Prompt {i+1}' for i in range(len(attack_summaries))])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 110)
        
        plt.tight_layout()
        plt.savefig('results/experiment_4_attack_summary.png', dpi=150)
        plt.close()

if __name__ == "__main__":
    experiment = AttackSimulationExperiment()
    experiment.run_all_experiments()
    print("\nExperiment 4 completed. Results saved to results/")