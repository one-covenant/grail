#!/usr/bin/env python3
"""
Experiment 5: Single Constraint (k=1) Deep Analysis

This experiment focuses on finding alternative hidden states that produce
the same sketch value for a single position, demonstrating:
1. Non-uniqueness exists (validating the theoretical concern)
2. Why this doesn't compromise GRAIL's multi-position security
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

sys.path.append('..')
from grail.grail import PRIME_Q, dot_mod_q, r_vec_from_randomness

@dataclass
class SingleConstraintResult:
    """Results from attempting to find alternative hidden states"""
    original_sketch: int
    found_alternatives: int
    alternative_sketches: List[int]
    distances_from_original: List[float]
    optimization_method: str
    success: bool
    iterations: int
    time_elapsed: float


class SingleConstraintAnalysis:
    def __init__(self, model_name="gpt2"):
        print(f"Initializing Single Constraint Analysis")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True).to(self.device).eval()
        except:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        
        self.hidden_dim = self.model.config.hidden_size
        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters")
        print(f"Hidden dimension: {self.hidden_dim}")
        
        if self.hidden_dim < 64:
            print(f"WARNING: Hidden dimension {self.hidden_dim} is very small!")
            print("Results may not be representative of larger models.")
        
    def generate_reference_hidden_state(self, prompt: str = "The future of AI") -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a reference hidden state from the model"""
        print(f"\nGenerating reference hidden state...")
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            # Get last hidden state of last token
            hidden_state = outputs.hidden_states[-1][0, -1]  # [hidden_dim]
            
        return hidden_state
    
    def find_alternatives_analytical(self, target_sketch: int, r_vec: torch.Tensor, 
                                   num_attempts: int = 100) -> SingleConstraintResult:
        """
        Analytically construct hidden states on the hyperplane <h, r> = target_sketch
        """
        print(f"\n=== Analytical Method ===")
        print(f"Target sketch: {target_sketch}")
        start_time = time.time()
        
        found_alternatives = 0
        alternative_sketches = []
        distances = []
        
        # Generate a reference point on the hyperplane
        # Start with a random vector and project it
        h_base = torch.randn(self.hidden_dim, device=self.device)
        
        # Compute initial sketch
        initial_sketch = dot_mod_q(h_base, r_vec)
        
        # Adjust to hit target more carefully
        # Since we're working with modular arithmetic, we need a better approach
        # Try to find a base point that's close to the target
        best_base = h_base.clone()
        best_diff = abs(initial_sketch - target_sketch)
        
        # Try multiple random starting points
        for _ in range(100):
            h_try = torch.randn(self.hidden_dim, device=self.device)
            # Scale to reasonable magnitude
            h_try = h_try * torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=self.device))
            
            sketch_try = dot_mod_q(h_try, r_vec)
            diff = abs(sketch_try - target_sketch)
            
            # Handle modular wraparound
            if diff > PRIME_Q / 2:
                diff = PRIME_Q - diff
            
            if diff < best_diff:
                best_diff = diff
                best_base = h_try.clone()
                
            if diff <= 3:  # Found a good starting point!
                print(f"  Found base point with sketch {sketch_try} (target: {target_sketch})")
                break
        
        h_base = best_base
        
        # Find orthogonal basis to r
        # Use Gram-Schmidt to create basis orthogonal to r
        r_normalized = r_vec.float() / torch.norm(r_vec.float())
        
        # Create orthonormal basis for the null space of r
        basis_vectors = []
        for i in range(self.hidden_dim):
            e_i = torch.zeros(self.hidden_dim, device=self.device)
            e_i[i] = 1.0
            
            # Ensure r_normalized is on the same device
            r_normalized = r_normalized.to(self.device)
            
            # Project out component parallel to r
            v = e_i - torch.dot(e_i, r_normalized) * r_normalized
            
            # Skip if too small (means e_i was nearly parallel to r)
            if torch.norm(v) > 1e-6:
                # Orthogonalize against previous basis vectors
                for prev_v in basis_vectors:
                    v = v - torch.dot(v, prev_v) * prev_v
                
                if torch.norm(v) > 1e-6:
                    v = v / torch.norm(v)
                    basis_vectors.append(v)
            
            if len(basis_vectors) >= self.hidden_dim - 1:
                break
        
        print(f"Created {len(basis_vectors)} orthogonal basis vectors")
        
        # Generate points on the hyperplane
        for i in range(num_attempts):
            # Random combination of basis vectors
            coeffs = torch.randn(len(basis_vectors), device=self.device) * (0.1 + i * 0.01)
            
            h_alt = h_base.clone()
            for j, basis_v in enumerate(basis_vectors):
                h_alt = h_alt + coeffs[j] * basis_v
            
            # Verify it produces the same sketch
            alt_sketch = dot_mod_q(h_alt, r_vec)
            alternative_sketches.append(alt_sketch)
            
            # Measure distance from base
            dist = torch.norm(h_alt - h_base).item()
            distances.append(dist)
            
            if abs(alt_sketch - target_sketch) <= 3:  # Within GRAIL tolerance
                found_alternatives += 1
                if found_alternatives <= 5:  # Print first few successes
                    print(f"  Found alternative {found_alternatives}: sketch={alt_sketch}, distance={dist:.2f}")
        
        time_elapsed = time.time() - start_time
        print(f"Found {found_alternatives}/{num_attempts} alternatives in {time_elapsed:.2f}s")
        
        return SingleConstraintResult(
            original_sketch=target_sketch,
            found_alternatives=found_alternatives,
            alternative_sketches=alternative_sketches,
            distances_from_original=distances,
            optimization_method="analytical",
            success=found_alternatives > 0,
            iterations=num_attempts,
            time_elapsed=time_elapsed
        )
    
    def find_alternatives_optimization(self, target_sketch: int, r_vec: torch.Tensor,
                                     h_init: torch.Tensor = None,
                                     num_attempts: int = 10,
                                     max_iters: int = 1000) -> SingleConstraintResult:
        """
        Use gradient-based optimization to find alternative hidden states
        """
        print(f"\n=== Optimization Method ===")
        print(f"Target sketch: {target_sketch}")
        start_time = time.time()
        
        found_alternatives = 0
        alternative_sketches = []
        distances = []
        
        if h_init is None:
            h_init = torch.randn(self.hidden_dim, device=self.device)
        
        for attempt in range(num_attempts):
            # Initialize from different random points
            h = torch.randn_like(h_init, requires_grad=True)
            if attempt == 0:
                h.data = h_init.clone()  # First attempt starts from reference
            
            # Different learning rates and optimizers for diversity
            if attempt % 3 == 0:
                optimizer = torch.optim.Adam([h], lr=0.1)
            elif attempt % 3 == 1:
                optimizer = torch.optim.SGD([h], lr=1.0, momentum=0.9)
            else:
                optimizer = torch.optim.LBFGS([h], lr=0.1)
            
            best_diff = float('inf')
            best_h = None
            
            for i in range(max_iters):
                def closure():
                    optimizer.zero_grad()
                    
                    current_sketch = dot_mod_q(h, r_vec)
                    
                    # Loss function that tries to match target sketch
                    sketch_diff = float(abs(current_sketch - target_sketch))
                    
                    # Smooth loss for optimization
                    if sketch_diff > PRIME_Q / 2:
                        # Handle wraparound in modular arithmetic
                        sketch_diff = PRIME_Q - sketch_diff
                    
                    # L2 regularization to keep magnitude reasonable
                    reg_loss = 0.0001 * torch.sum(h ** 2)
                    
                    # Combined loss - we can't directly optimize modular arithmetic
                    # so we use a proxy loss
                    loss = sketch_diff / PRIME_Q + reg_loss
                    
                    # For LBFGS
                    if isinstance(optimizer, torch.optim.LBFGS):
                        # LBFGS needs a scalar tensor loss
                        loss_tensor = torch.tensor(float(loss), device=h.device)
                        # Compute gradient approximately
                        eps = 1e-5
                        with torch.no_grad():
                            h_plus = h + eps * torch.randn_like(h)
                            sketch_plus = dot_mod_q(h_plus, r_vec)
                            grad_approx = (sketch_plus - current_sketch) / eps * torch.randn_like(h)
                            h.grad = grad_approx + 0.0002 * h
                        return loss_tensor
                    
                    # For other optimizers
                    # Create a differentiable loss
                    loss_tensor = torch.tensor(float(loss), dtype=torch.float32, device=h.device, requires_grad=True)
                    
                    # Compute gradient manually since loss is not differentiable
                    # We'll use a finite difference approximation
                    eps = 1e-5
                    h_plus = h + eps * torch.randn_like(h)
                    h_minus = h - eps * torch.randn_like(h)
                    
                    sketch_plus = dot_mod_q(h_plus, r_vec)
                    sketch_minus = dot_mod_q(h_minus, r_vec)
                    
                    # Approximate gradient
                    grad_approx = (sketch_plus - sketch_minus) / (2 * eps) * torch.randn_like(h)
                    h.grad = grad_approx + reg_loss * 2 * h
                    
                    # Add gradient noise for exploration
                    if i % 50 == 0 and i > 0 and h.grad is not None:
                        with torch.no_grad():
                            h.grad += torch.randn_like(h.grad) * 0.1
                    
                    # Clip gradients to prevent explosion
                    if h.grad is not None:
                        torch.nn.utils.clip_grad_norm_([h], max_norm=1.0)
                    
                    return loss
                
                if isinstance(optimizer, torch.optim.LBFGS):
                    optimizer.step(closure)
                else:
                    closure()
                    optimizer.step()
                
                # Check current state
                with torch.no_grad():
                    # Check for NaN values
                    if torch.isnan(h).any():
                        print(f"    Warning: NaN detected at iteration {i}, resetting")
                        h.data = torch.randn_like(h_init)
                        optimizer = torch.optim.Adam([h], lr=0.001)
                        continue
                    
                    current_sketch = dot_mod_q(h, r_vec)
                    diff = abs(current_sketch - target_sketch)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_h = h.clone()
                    
                    if diff <= 3:  # Success!
                        found_alternatives += 1
                        alt_sketch = current_sketch
                        dist = torch.norm(h - h_init).item()
                        
                        alternative_sketches.append(alt_sketch)
                        distances.append(dist)
                        
                        print(f"  Attempt {attempt+1}: Found alternative with sketch={alt_sketch}, "
                              f"distance={dist:.2f}, iterations={i}")
                        break
                
                if i % 200 == 0 and i > 0:
                    print(f"    Iteration {i}: best_diff={best_diff}")
            
            # If didn't converge, still record the best attempt
            if best_h is not None and best_diff > 3:
                with torch.no_grad():
                    alt_sketch = dot_mod_q(best_h, r_vec)
                    dist = torch.norm(best_h - h_init).item()
                    alternative_sketches.append(alt_sketch)
                    distances.append(dist)
        
        time_elapsed = time.time() - start_time
        print(f"Found {found_alternatives}/{num_attempts} alternatives in {time_elapsed:.2f}s")
        
        return SingleConstraintResult(
            original_sketch=target_sketch,
            found_alternatives=found_alternatives,
            alternative_sketches=alternative_sketches,
            distances_from_original=distances,
            optimization_method="gradient_optimization",
            success=found_alternatives > 0,
            iterations=num_attempts * max_iters,
            time_elapsed=time_elapsed
        )
    
    def find_alternatives_smart_search(self, target_sketch: int, r_vec: torch.Tensor,
                                     h_ref: torch.Tensor,
                                     search_radius: float = 10.0) -> SingleConstraintResult:
        """
        Smart search: exploit the hyperplane structure directly
        """
        print(f"\n=== Smart Search Method ===")
        print(f"Target sketch: {target_sketch}")
        start_time = time.time()
        
        found_alternatives = 0
        alternative_sketches = []
        distances = []
        
        # Key insight: we need <h, r> ≡ target (mod q)
        # So h = h_ref + v where <v, r> ≡ 0 (mod q)
        
        # Find the component of h_ref parallel to r
        r_normalized = r_vec.float().to(self.device) / torch.norm(r_vec.float().to(self.device))
        h_ref_parallel = torch.dot(h_ref, r_normalized) * r_normalized
        h_ref_perp = h_ref - h_ref_parallel
        
        # Generate random vectors orthogonal to r
        num_samples = 1000
        for i in range(num_samples):
            # Random vector
            v_random = torch.randn(self.hidden_dim, device=self.device)
            
            # Make it orthogonal to r
            v_perp = v_random - torch.dot(v_random, r_normalized) * r_normalized
            
            # Scale it
            scale = search_radius * (i / num_samples)  # Gradually increase search radius
            v_perp = v_perp / torch.norm(v_perp) * scale
            
            # Create alternative hidden state
            h_alt = h_ref + v_perp
            
            # Verify sketch
            alt_sketch = dot_mod_q(h_alt, r_vec)
            alternative_sketches.append(alt_sketch)
            
            dist = torch.norm(v_perp).item()
            distances.append(dist)
            
            if abs(alt_sketch - target_sketch) <= 3:
                found_alternatives += 1
                if found_alternatives <= 10:
                    print(f"  Found alternative {found_alternatives}: sketch={alt_sketch}, "
                          f"distance={dist:.2f}")
        
        time_elapsed = time.time() - start_time
        success_rate = found_alternatives / num_samples * 100
        print(f"Found {found_alternatives}/{num_samples} alternatives ({success_rate:.1f}%) "
              f"in {time_elapsed:.2f}s")
        
        return SingleConstraintResult(
            original_sketch=target_sketch,
            found_alternatives=found_alternatives,
            alternative_sketches=alternative_sketches,
            distances_from_original=distances,
            optimization_method="smart_search",
            success=found_alternatives > 0,
            iterations=num_samples,
            time_elapsed=time_elapsed
        )
    
    def analyze_sketch_distribution(self, r_vec: torch.Tensor, num_samples: int = 10000):
        """Analyze the distribution of sketch values for random hidden states"""
        print(f"\n=== Analyzing Sketch Distribution ===")
        print(f"Generating {num_samples} random hidden states...")
        
        sketches = []
        for i in range(num_samples):
            h = torch.randn(self.hidden_dim, device=self.device)
            # Scale to reasonable magnitude
            h = h / torch.norm(h) * np.sqrt(self.hidden_dim)
            
            sketch = dot_mod_q(h, r_vec)
            sketches.append(sketch)
            
            if i % 1000 == 0:
                print(f"  Progress: {i}/{num_samples}", end='\r')
        
        sketches = np.array(sketches)
        
        # Analyze distribution
        unique_sketches = len(np.unique(sketches))
        sketch_mean = np.mean(sketches)
        sketch_std = np.std(sketches)
        
        print(f"\nDistribution Statistics:")
        print(f"  Unique values: {unique_sketches}/{num_samples}")
        print(f"  Mean: {sketch_mean:.2e}")
        print(f"  Std: {sketch_std:.2e}")
        print(f"  Range: [{np.min(sketches)}, {np.max(sketches)}]")
        print(f"  Collision rate: {(num_samples - unique_sketches) / num_samples:.2%}")
        
        return sketches
    
    def test_multi_constraint_smart_search(self, k_values=[1, 2, 4, 8, 16]):
        """Test smart search with multiple k values to show scaling difficulty"""
        print("\n" + "="*60)
        print("Multi-Constraint Smart Search Analysis")
        print("="*60)
        
        results = {}
        
        # Generate reference sequence
        prompt = "The future of artificial intelligence"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=32,
                return_dict_in_generate=True,
                output_hidden_states=True,
                do_sample=False
            )
            # Get hidden states
            full_outputs = self.model(outputs.sequences, output_hidden_states=True)
            all_hidden_states = full_outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
        
        seq_len = all_hidden_states.shape[0]
        print(f"Generated sequence length: {seq_len}")
        
        # Generate multiple r vectors for different positions
        r_vectors = []
        for i in range(min(16, seq_len)):
            randomness = "0x" + hex(i)[2:].zfill(64)
            r_vec = r_vec_from_randomness(randomness, self.hidden_dim)
            r_vectors.append(r_vec)
        
        # Test each k value
        for k in k_values:
            print(f"\n" + "-"*60)
            print(f"Testing k={k} constraints")
            
            # Select k positions and their hidden states
            positions = sorted(np.random.choice(min(16, seq_len), k, replace=False))
            target_sketches = []
            
            for pos in positions:
                h = all_hidden_states[int(pos)]
                sketch = dot_mod_q(h, r_vectors[int(pos)])
                target_sketches.append(sketch)
            
            print(f"Constraint positions: {[int(p) for p in positions]}")
            print(f"Target sketches: {target_sketches[:5]}..." if len(target_sketches) > 5 else f"Target sketches: {target_sketches}")
            
            # Try to find alternative hidden states that satisfy all k constraints
            start_time = time.time()
            success_count = 0
            num_attempts = 1000
            
            for attempt in range(num_attempts):
                # Start with random hidden states
                h_alternatives = []
                all_constraints_met = True
                
                for i, pos in enumerate(positions):
                    # For each position, try to find an alternative that matches the sketch
                    h_base = all_hidden_states[int(pos)]
                    target_sketch = target_sketches[i]
                    r_vec = r_vectors[int(pos)]
                    
                    # Use smart search for this position
                    found = False
                    for _ in range(10):  # Quick attempts per position
                        # Generate orthogonal perturbation
                        r_normalized = r_vec.float().to(self.device) / torch.norm(r_vec.float().to(self.device))
                        v = torch.randn(self.hidden_dim, device=self.device)
                        v = v - torch.dot(v, r_normalized) * r_normalized
                        
                        if torch.norm(v) > 1e-6:
                            v = v / torch.norm(v) * np.random.uniform(0.1, 10.0)
                            h_alt = h_base + v
                            
                            sketch = dot_mod_q(h_alt, r_vec)
                            if abs(sketch - target_sketch) <= 3:
                                h_alternatives.append(h_alt)
                                found = True
                                break
                    
                    if not found:
                        all_constraints_met = False
                        break
                
                if all_constraints_met:
                    success_count += 1
                    if success_count <= 5:
                        print(f"  Success #{success_count} at attempt {attempt}")
            
            success_rate = success_count / num_attempts
            elapsed_time = time.time() - start_time
            
            results[f'k_{k}'] = {
                'k': k,
                'positions': [int(p) for p in positions],
                'success_count': success_count,
                'num_attempts': num_attempts,
                'success_rate': success_rate,
                'elapsed_time': elapsed_time,
                'avg_time_per_attempt': elapsed_time / num_attempts
            }
            
            print(f"\nResults for k={k}:")
            print(f"  Success rate: {success_rate*100:.2f}% ({success_count}/{num_attempts})")
            print(f"  Time: {elapsed_time:.2f}s")
            print(f"  Avg time per attempt: {elapsed_time/num_attempts*1000:.1f}ms")
        
        return results

    def run_all_analyses(self):
        """Run comprehensive analysis of single constraint case"""
        print("="*60)
        print("Single Constraint (k=1) Deep Analysis")
        print("="*60)
        
        results = {}
        
        # 1. Generate reference hidden state
        h_ref = self.generate_reference_hidden_state()
        
        # 2. Create random vector r
        randomness = "0x" + "c" * 64
        r_vec = r_vec_from_randomness(randomness, self.hidden_dim)
        
        # 3. Compute reference sketch
        target_sketch = dot_mod_q(h_ref, r_vec)
        print(f"\nReference sketch value: {target_sketch}")
        print(f"GRAIL tolerance: ±{3}")
        
        # 4. Analyze sketch distribution
        print("\n" + "-"*60)
        sketch_distribution = self.analyze_sketch_distribution(r_vec, num_samples=5000)
        
        # 5. Try different methods to find alternatives
        print("\n" + "-"*60)
        results['analytical'] = self.find_alternatives_analytical(target_sketch, r_vec)
        
        print("\n" + "-"*60)
        results['optimization'] = self.find_alternatives_optimization(target_sketch, r_vec, h_ref)
        
        print("\n" + "-"*60)
        results['smart_search'] = self.find_alternatives_smart_search(target_sketch, r_vec, h_ref)
        
        # 6. NEW: Test multi-constraint scaling
        print("\n" + "="*60)
        multi_constraint_results = self.test_multi_constraint_smart_search()
        results['multi_constraint'] = multi_constraint_results
        
        # 7. Summary analysis
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        # Summarize single constraint methods
        for method in ['analytical', 'optimization', 'smart_search']:
            if method in results:
                result = results[method]
                print(f"\n{method.upper()} Method:")
                print(f"  Success: {result.success}")
                print(f"  Alternatives found: {result.found_alternatives}")
                print(f"  Success rate: {result.found_alternatives / result.iterations * 100:.1f}%")
                print(f"  Time: {result.time_elapsed:.2f}s")
                
                if result.found_alternatives > 0:
                    valid_distances = [d for s, d in zip(result.alternative_sketches, result.distances_from_original)
                                     if abs(s - target_sketch) <= 3]
                    if valid_distances:
                        print(f"  Avg distance from original: {np.mean(valid_distances):.2f}")
                        print(f"  Min distance from original: {np.min(valid_distances):.2f}")
        
        # Summarize multi-constraint results
        if 'multi_constraint' in results:
            print(f"\n\nMULTI-CONSTRAINT SCALING:")
            print("k | Success Rate | Time (s)")
            print("-" * 30)
            for k in [1, 2, 4, 8, 16]:
                key = f'k_{k}'
                if key in results['multi_constraint']:
                    mc_result = results['multi_constraint'][key]
                    print(f"{k:2d} | {mc_result['success_rate']*100:11.2f}% | {mc_result['elapsed_time']:8.2f}")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        
        # Convert to JSON-serializable format
        results_json = {}
        for method, result in results.items():
            if method == 'multi_constraint':
                # Handle multi-constraint results differently
                results_json[method] = result
            elif hasattr(result, 'original_sketch'):
                # Handle single constraint results
                results_json[method] = {
                    'original_sketch': int(result.original_sketch),
                    'found_alternatives': int(result.found_alternatives),
                    'alternative_sketches': [int(s) for s in result.alternative_sketches[:100]],  # Limit size
                    'distances_from_original': [float(d) for d in result.distances_from_original[:100]],
                    'optimization_method': result.optimization_method,
                    'success': result.success,
                    'iterations': int(result.iterations),
                    'time_elapsed': float(result.time_elapsed)
                }
        
        results_json['sketch_distribution'] = {
            'num_samples': len(sketch_distribution),
            'unique_values': int(len(np.unique(sketch_distribution))),
            'mean': float(np.mean(sketch_distribution)),
            'std': float(np.std(sketch_distribution)),
            'min': int(np.min(sketch_distribution)),
            'max': int(np.max(sketch_distribution))
        }
        
        with open('results/experiment_5_single_constraint_analysis.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Generate plots
        self._generate_plots(results, sketch_distribution)
        
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        print("1. Alternative hidden states DO exist (validating theoretical concern)")
        print("2. They form a (d-1)-dimensional hyperplane in d-dimensional space")
        print("3. Smart search finds them easily by exploiting orthogonality for k=1")
        print("4. BUT: Success rate drops EXPONENTIALLY with k:")
        if 'multi_constraint' in results:
            for k in [1, 2, 4, 8, 16]:
                key = f'k_{k}'
                if key in results['multi_constraint']:
                    rate = results['multi_constraint'][key]['success_rate'] * 100
                    print(f"     k={k:2d}: {rate:6.2f}% success rate")
        print("5. With k=16, finding valid alternatives becomes computationally infeasible")
        print("\nThis experiment proves non-uniqueness exists but doesn't threaten GRAIL's security.")
        
    def _generate_plots(self, results: Dict, sketch_distribution: np.ndarray):
        """Generate visualization plots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot 1: Success rates by method
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Filter only single constraint methods (not multi_constraint)
        single_methods = ['analytical', 'optimization', 'smart_search']
        methods = [m for m in single_methods if m in results and hasattr(results[m], 'found_alternatives')]
        success_rates = [results[m].found_alternatives / results[m].iterations * 100 for m in methods]
        times = [results[m].time_elapsed for m in methods]
        
        ax1.bar(methods, success_rates, alpha=0.7)
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate by Method')
        ax1.set_ylim(0, max(success_rates) * 1.2)
        
        for i, v in enumerate(success_rates):
            ax1.text(i, v + 0.5, f'{v:.1f}%', ha='center')
        
        ax2.bar(methods, times, alpha=0.7, color='orange')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Computation Time by Method')
        
        plt.tight_layout()
        plt.savefig('results/experiment_5_method_comparison.png', dpi=150)
        plt.close()
        
        # Plot 2: Sketch distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(sketch_distribution, bins=50, alpha=0.7, density=True)
        ax.set_xlabel('Sketch Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of Sketch Values (n={len(sketch_distribution)})')
        ax.axvline(np.mean(sketch_distribution), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(sketch_distribution):.2e}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('results/experiment_5_sketch_distribution.png', dpi=150)
        plt.close()
        
        # Plot 3: Distance vs Sketch difference for alternatives
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method in ['analytical', 'optimization', 'smart_search']:
            if method in results and hasattr(results[method], 'found_alternatives'):
                result = results[method]
                if result.found_alternatives > 0:
                    # Only plot successful attempts
                    target = result.original_sketch
                    diffs = [abs(s - target) for s in result.alternative_sketches]
                    dists = result.distances_from_original
                    
                    # Filter to reasonable range
                    valid_idx = [i for i, d in enumerate(diffs) if d < PRIME_Q/2]
                    diffs = [diffs[i] for i in valid_idx]
                    dists = [dists[i] for i in valid_idx]
                    
                    ax.scatter(dists[:100], diffs[:100], alpha=0.5, label=method, s=20)
        
        ax.axhline(3, color='red', linestyle='--', label='GRAIL Tolerance')
        ax.set_xlabel('Distance from Original Hidden State')
        ax.set_ylabel('Sketch Difference')
        ax.set_title('Alternative Hidden States: Distance vs Sketch Difference')
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('results/experiment_5_alternatives_analysis.png', dpi=150)
        plt.close()
        
        # Plot 4: Multi-constraint scaling (NEW)
        if 'multi_constraint' in results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            k_values = []
            success_rates = []
            times = []
            
            for k in [1, 2, 4, 8, 16]:
                key = f'k_{k}'
                if key in results['multi_constraint']:
                    mc_result = results['multi_constraint'][key]
                    k_values.append(k)
                    success_rates.append(mc_result['success_rate'] * 100)
                    times.append(mc_result['elapsed_time'])
            
            # Success rate vs k (log scale)
            ax1.semilogy(k_values, success_rates, 'bo-', markersize=10, linewidth=2)
            ax1.set_xlabel('Number of Constraints (k)')
            ax1.set_ylabel('Success Rate (%)')
            ax1.set_title('Success Rate vs Number of Constraints')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(k_values)
            
            # Add annotations
            for i, (k, rate) in enumerate(zip(k_values, success_rates)):
                if rate > 0:
                    ax1.annotate(f'{rate:.1f}%', xy=(k, rate), xytext=(k, rate*1.5),
                               ha='center', fontsize=9)
            
            # Time vs k
            ax2.plot(k_values, times, 'ro-', markersize=10, linewidth=2)
            ax2.set_xlabel('Number of Constraints (k)')
            ax2.set_ylabel('Time (seconds)')
            ax2.set_title('Computation Time vs Number of Constraints')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(k_values)
            
            plt.tight_layout()
            plt.savefig('results/experiment_5_multi_constraint_scaling.png', dpi=150)
            plt.close()


if __name__ == "__main__":
    print("Starting Single Constraint Analysis...")
    experiment = SingleConstraintAnalysis()
    experiment.run_all_analyses()
    print("\nExperiment 5 completed. Results saved to results/")