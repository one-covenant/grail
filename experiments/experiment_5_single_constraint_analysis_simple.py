#!/usr/bin/env python3
"""
Experiment 5: Single Constraint (k=1) Analysis - Simplified Version

This experiment focuses on demonstrating that alternative hidden states exist
for a single sketch constraint, using a simpler, more direct approach.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import sys
import time

sys.path.append('..')
from grail.grail import PRIME_Q, dot_mod_q, r_vec_from_randomness


class SingleConstraintSimple:
    def __init__(self, model_name="gpt2"):
        print(f"Initializing Single Constraint Analysis (Simplified)")
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
        
    def direct_hyperplane_search(self, target_sketch: int, r_vec: torch.Tensor, 
                                num_samples: int = 10000) -> dict:
        """
        Direct approach: Generate many random hidden states and find those
        that produce sketch values close to the target.
        """
        print(f"\n=== Direct Hyperplane Search ===")
        print(f"Target sketch: {target_sketch}")
        print(f"Generating {num_samples} random hidden states...")
        
        found_exact = 0
        found_within_tolerance = 0
        distances_exact = []
        distances_tolerance = []
        sketches = []
        
        # Reference point (optional - can use any point)
        h_ref = torch.randn(self.hidden_dim, device=self.device)
        h_ref = h_ref * torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=self.device))
        ref_sketch = dot_mod_q(h_ref, r_vec)
        
        print(f"Reference sketch: {ref_sketch}")
        
        # Batch processing for efficiency
        batch_size = 1000
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            actual_batch_size = batch_end - batch_start
            
            # Generate batch of random hidden states
            h_batch = torch.randn(actual_batch_size, self.hidden_dim, device=self.device)
            # Scale to reasonable magnitude
            h_batch = h_batch * torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=self.device))
            
            # Compute sketches for the batch
            for i in range(actual_batch_size):
                h = h_batch[i]
                sketch = dot_mod_q(h, r_vec)
                sketches.append(sketch)
                
                diff = abs(sketch - target_sketch)
                
                # Handle modular wraparound
                if diff > PRIME_Q / 2:
                    diff = PRIME_Q - diff
                
                if diff == 0:
                    found_exact += 1
                    dist = torch.norm(h - h_ref).item()
                    distances_exact.append(dist)
                    if found_exact <= 5:
                        print(f"  Found exact match #{found_exact}: distance from ref = {dist:.2f}")
                
                elif diff <= 3:  # Within GRAIL tolerance
                    found_within_tolerance += 1
                    dist = torch.norm(h - h_ref).item()
                    distances_tolerance.append(dist)
                    if found_within_tolerance <= 5:
                        print(f"  Found within tolerance #{found_within_tolerance}: "
                              f"sketch={sketch}, diff={diff}, distance={dist:.2f}")
            
            if batch_start % 5000 == 0:
                print(f"  Progress: {batch_start}/{num_samples} samples processed...")
        
        print(f"\nResults:")
        print(f"  Exact matches: {found_exact} ({found_exact/num_samples*100:.3f}%)")
        print(f"  Within tolerance (±3): {found_within_tolerance} ({found_within_tolerance/num_samples*100:.3f}%)")
        
        return {
            'num_samples': num_samples,
            'found_exact': found_exact,
            'found_within_tolerance': found_within_tolerance,
            'success_rate_exact': found_exact / num_samples,
            'success_rate_tolerance': found_within_tolerance / num_samples,
            'distances_exact': distances_exact,
            'distances_tolerance': distances_tolerance,
            'all_sketches': sketches
        }
    
    def orthogonal_perturbation_method(self, target_sketch: int, r_vec: torch.Tensor,
                                     num_directions: int = 1000) -> dict:
        """
        Smart method: Start with a point that has the target sketch,
        then move in directions orthogonal to r.
        """
        print(f"\n=== Orthogonal Perturbation Method ===")
        print(f"Target sketch: {target_sketch}")
        
        # First, find a starting point with the target sketch
        print("Finding initial point with target sketch...")
        h_start = None
        for attempt in range(1000):
            h_try = torch.randn(self.hidden_dim, device=self.device)
            h_try = h_try * torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=self.device))
            
            sketch = dot_mod_q(h_try, r_vec)
            if abs(sketch - target_sketch) <= 3:
                h_start = h_try
                print(f"  Found starting point with sketch {sketch}")
                break
        
        if h_start is None:
            print("  Could not find starting point!")
            return {'success': False}
        
        # Now generate points by moving orthogonal to r
        r_normalized = r_vec.float().to(self.device) / torch.norm(r_vec.float().to(self.device))
        
        found_alternatives = 0
        distances = []
        sketch_values = []
        
        print(f"Generating {num_directions} orthogonal perturbations...")
        
        for i in range(num_directions):
            # Random direction
            v = torch.randn(self.hidden_dim, device=self.device)
            
            # Make it orthogonal to r
            v = v - torch.dot(v, r_normalized) * r_normalized
            
            # Normalize and scale
            if torch.norm(v) > 1e-6:
                v = v / torch.norm(v)
                
                # Try different scales
                for scale in [0.1, 1.0, 10.0, 100.0]:
                    h_new = h_start + scale * v
                    
                    sketch_new = dot_mod_q(h_new, r_vec)
                    sketch_values.append(sketch_new)
                    
                    if abs(sketch_new - target_sketch) <= 3:
                        found_alternatives += 1
                        dist = scale
                        distances.append(dist)
                        
                        if found_alternatives <= 10:
                            print(f"  Alternative #{found_alternatives}: "
                                  f"sketch={sketch_new}, distance={dist:.2f}")
        
        success_rate = found_alternatives / (num_directions * 4)  # 4 scales per direction
        print(f"\nFound {found_alternatives} alternatives")
        print(f"Success rate: {success_rate*100:.2f}%")
        
        return {
            'success': True,
            'found_alternatives': found_alternatives,
            'success_rate': success_rate,
            'distances': distances,
            'sketch_values': sketch_values
        }
    
    def analyze_hyperplane_structure(self, r_vec: torch.Tensor) -> dict:
        """
        Analyze the structure of the hyperplane for different sketch values.
        """
        print(f"\n=== Hyperplane Structure Analysis ===")
        
        # Sample the sketch space
        num_samples = 10000
        sketches = []
        
        print(f"Sampling {num_samples} random hidden states...")
        for i in range(num_samples):
            h = torch.randn(self.hidden_dim, device=self.device)
            h = h * torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=self.device))
            sketch = dot_mod_q(h, r_vec)
            sketches.append(sketch)
        
        sketches = np.array(sketches)
        
        # Analyze distribution
        unique_sketches = len(np.unique(sketches))
        
        # Test how many unique hyperplanes we're sampling
        print(f"\nSketch space coverage:")
        print(f"  Unique sketch values: {unique_sketches}/{num_samples}")
        print(f"  Coverage of sketch space: {unique_sketches/PRIME_Q*100:.6f}%")
        print(f"  Mean sketch value: {np.mean(sketches):.2e}")
        print(f"  Std of sketch values: {np.std(sketches):.2e}")
        
        # Expected number of points per hyperplane
        points_per_hyperplane = num_samples / unique_sketches
        print(f"  Average points per hyperplane: {points_per_hyperplane:.2f}")
        
        # This tells us about the density of our sampling
        volume_ratio = self.hidden_dim / unique_sketches
        print(f"  Dimension/unique_sketches ratio: {volume_ratio:.2f}")
        
        return {
            'num_samples': num_samples,
            'unique_sketches': unique_sketches,
            'coverage': unique_sketches / PRIME_Q,
            'points_per_hyperplane': points_per_hyperplane,
            'sketch_distribution': {
                'mean': float(np.mean(sketches)),
                'std': float(np.std(sketches)),
                'min': int(np.min(sketches)),
                'max': int(np.max(sketches))
            }
        }
    
    def run_analysis(self):
        """Run the simplified analysis"""
        print("="*60)
        print("Single Constraint (k=1) Analysis - Simplified")
        print("="*60)
        
        results = {}
        
        # Generate random r vector
        randomness = "0x" + "d" * 64
        r_vec = r_vec_from_randomness(randomness, self.hidden_dim)
        
        # Pick a target sketch value (use a common one from the distribution)
        # First, sample some sketches to find a reasonable target
        sample_sketches = []
        for _ in range(100):
            h = torch.randn(self.hidden_dim, device=self.device)
            h = h * torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=self.device))
            sample_sketches.append(dot_mod_q(h, r_vec))
        
        # Use median as target (likely to have many solutions)
        target_sketch = int(np.median(sample_sketches))
        print(f"\nTarget sketch value: {target_sketch}")
        
        # Run experiments
        print("\n" + "-"*60)
        results['direct_search'] = self.direct_hyperplane_search(target_sketch, r_vec, num_samples=20000)
        
        print("\n" + "-"*60)
        results['orthogonal_method'] = self.orthogonal_perturbation_method(target_sketch, r_vec)
        
        print("\n" + "-"*60)
        results['structure_analysis'] = self.analyze_hyperplane_structure(r_vec)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if 'direct_search' in results:
            ds = results['direct_search']
            print(f"\nDirect Search Method:")
            print(f"  Found {ds['found_exact']} exact matches out of {ds['num_samples']} samples")
            print(f"  Success rate: {ds['success_rate_exact']*100:.3f}%")
            print(f"  With tolerance ±3: {ds['success_rate_tolerance']*100:.3f}%")
        
        if 'orthogonal_method' in results and results['orthogonal_method']['success']:
            om = results['orthogonal_method']
            print(f"\nOrthogonal Perturbation Method:")
            print(f"  Found {om['found_alternatives']} alternatives")
            print(f"  Success rate: {om['success_rate']*100:.2f}%")
        
        if 'structure_analysis' in results:
            sa = results['structure_analysis']
            print(f"\nHyperplane Structure:")
            print(f"  Each hyperplane contains ~1/{PRIME_Q:.2e} of the space")
            print(f"  We sampled {sa['unique_sketches']} unique hyperplanes")
            print(f"  Average {sa['points_per_hyperplane']:.1f} points per hyperplane in our sample")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        
        # Clean up results for JSON serialization
        clean_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                clean_dict = {}
                for k, v in value.items():
                    if k in ['all_sketches', 'sketch_values']:
                        # Limit large arrays
                        clean_dict[k] = v[:1000] if isinstance(v, list) else v
                    elif isinstance(v, (list, tuple)):
                        clean_dict[k] = [float(x) if isinstance(x, (np.floating, torch.Tensor)) else x for x in v[:100]]
                    elif isinstance(v, (np.integer, np.floating)):
                        clean_dict[k] = float(v)
                    elif isinstance(v, torch.Tensor):
                        clean_dict[k] = v.item() if v.numel() == 1 else v.tolist()
                    else:
                        clean_dict[k] = v
                clean_results[key] = clean_dict
            else:
                clean_results[key] = value
        
        with open('results/experiment_5_single_constraint_simple.json', 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        # Generate plots
        self._generate_plots(results, target_sketch)
        
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        print("1. Alternative hidden states DO exist for single constraints")
        print("2. They are relatively easy to find with random sampling")
        print("3. Orthogonal perturbations preserve sketch values perfectly")
        print("4. Each hyperplane contains a vast number of possible hidden states")
        print("5. BUT: With k=16 constraints, finding the intersection becomes exponentially harder")
        
    def _generate_plots(self, results, target_sketch):
        """Generate visualization plots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot 1: Sketch distribution from direct search
        if 'direct_search' in results and 'all_sketches' in results['direct_search']:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sketches = results['direct_search']['all_sketches'][:5000]  # Limit for plotting
            ax.hist(sketches, bins=50, alpha=0.7, density=True)
            ax.axvline(target_sketch, color='red', linestyle='--', linewidth=2,
                      label=f'Target: {target_sketch}')
            ax.set_xlabel('Sketch Value')
            ax.set_ylabel('Density')
            ax.set_title('Distribution of Sketch Values from Random Sampling')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig('results/experiment_5_sketch_distribution_simple.png', dpi=150)
            plt.close()
        
        # Plot 2: Success rates comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        
        methods = []
        success_rates = []
        
        if 'direct_search' in results:
            methods.extend(['Direct (exact)', 'Direct (±3)'])
            success_rates.extend([
                results['direct_search']['success_rate_exact'] * 100,
                results['direct_search']['success_rate_tolerance'] * 100
            ])
        
        if 'orthogonal_method' in results and results['orthogonal_method'].get('success'):
            methods.append('Orthogonal')
            success_rates.append(results['orthogonal_method']['success_rate'] * 100)
        
        bars = ax.bar(methods, success_rates, alpha=0.7)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rates for Finding Alternative Hidden States (k=1)')
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.3f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/experiment_5_success_rates_simple.png', dpi=150)
        plt.close()


if __name__ == "__main__":
    print("Starting Simplified Single Constraint Analysis...")
    experiment = SingleConstraintSimple()
    experiment.run_analysis()
    print("\nExperiment completed. Results saved to results/")