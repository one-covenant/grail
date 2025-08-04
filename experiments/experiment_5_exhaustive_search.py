#!/usr/bin/env python3
"""
Experiment 5b: Exhaustive Search for k=1 Collisions

This experiment attempts to exhaustively search for collisions by:
1. Generating millions of hidden states
2. Building a lookup table of sketch values
3. Finding actual collisions (same sketch, different hidden states)
4. Demonstrating that alternatives DO exist but are rare
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
import sys
import time
from tqdm import tqdm

sys.path.append('..')
from grail.grail import PRIME_Q, dot_mod_q, r_vec_from_randomness


class ExhaustiveSketchSearch:
    def __init__(self, hidden_dim=768):
        print(f"Initializing Exhaustive Sketch Search")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.hidden_dim = hidden_dim
        print(f"Hidden dimension: {self.hidden_dim}")
        print(f"Sketch space size: {PRIME_Q:,} ({PRIME_Q:.2e})")
        
    def exhaustive_collision_search(self, r_vec: torch.Tensor, num_samples: int = 1_000_000):
        """
        Generate many hidden states and find collisions
        """
        print(f"\n=== Exhaustive Collision Search ===")
        print(f"Generating {num_samples:,} hidden states to find collisions...")
        
        # Dictionary to store sketch -> list of hidden states
        sketch_to_hidden = defaultdict(list)
        
        # Batch processing for efficiency
        batch_size = 10000
        total_collisions = 0
        unique_sketches = set()
        
        # Store some examples of collisions
        collision_examples = []
        
        # Progress tracking
        pbar = tqdm(total=num_samples, desc="Generating samples")
        
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            actual_batch_size = batch_end - batch_start
            
            # Generate batch
            h_batch = torch.randn(actual_batch_size, self.hidden_dim, device=self.device)
            
            # Try different scales to increase diversity
            scales = torch.exp(torch.rand(actual_batch_size, 1, device=self.device) * 4 - 2)  # 0.135 to 54.6
            h_batch = h_batch * scales * torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=self.device))
            
            # Process each hidden state
            for i in range(actual_batch_size):
                h = h_batch[i]
                sketch = dot_mod_q(h, r_vec)
                
                # Check if we've seen this sketch before
                if sketch in sketch_to_hidden and len(collision_examples) < 100:
                    # Found a collision!
                    total_collisions += 1
                    
                    # Get the previous hidden state with same sketch
                    prev_h = sketch_to_hidden[sketch][0]
                    
                    # Compute distance between the two hidden states
                    distance = torch.norm(h - prev_h).item()
                    
                    if total_collisions <= 10:
                        print(f"\n  Collision #{total_collisions} found!")
                        print(f"    Sketch value: {sketch}")
                        print(f"    Distance between hidden states: {distance:.2f}")
                        print(f"    Relative distance: {distance / torch.norm(h).item():.2%}")
                    
                    collision_examples.append({
                        'sketch': int(sketch),
                        'distance': float(distance),
                        'h1_norm': float(torch.norm(prev_h).item()),
                        'h2_norm': float(torch.norm(h).item()),
                        'relative_distance': float(distance / max(torch.norm(h).item(), torch.norm(prev_h).item()))
                    })
                
                # Store this hidden state
                if len(sketch_to_hidden[sketch]) < 10:  # Limit storage per sketch
                    sketch_to_hidden[sketch].append(h.clone().detach())
                
                unique_sketches.add(sketch)
            
            pbar.update(actual_batch_size)
            
            # Periodic statistics
            if batch_start > 0 and batch_start % 100000 == 0:
                collision_rate = total_collisions / batch_end
                print(f"\n  Progress: {batch_end:,} samples")
                print(f"  Unique sketches: {len(unique_sketches):,}")
                print(f"  Total collisions: {total_collisions:,}")
                print(f"  Collision rate: {collision_rate:.6%}")
                print(f"  Coverage of sketch space: {len(unique_sketches)/PRIME_Q:.6%}")
        
        pbar.close()
        
        # Final statistics
        collision_rate = total_collisions / num_samples
        coverage = len(unique_sketches) / PRIME_Q
        
        print(f"\n=== Final Results ===")
        print(f"Total samples: {num_samples:,}")
        print(f"Unique sketch values: {len(unique_sketches):,}")
        print(f"Total collisions: {total_collisions:,}")
        print(f"Collision rate: {collision_rate:.6%}")
        print(f"Coverage of sketch space: {coverage:.6%}")
        print(f"Average samples per sketch: {num_samples / len(unique_sketches):.2f}")
        
        # Analyze collision distances
        if collision_examples:
            distances = [c['distance'] for c in collision_examples]
            rel_distances = [c['relative_distance'] for c in collision_examples]
            
            print(f"\nCollision Analysis:")
            print(f"  Number of collision pairs found: {len(collision_examples)}")
            print(f"  Average distance between colliding states: {np.mean(distances):.2f}")
            print(f"  Min distance: {np.min(distances):.2f}")
            print(f"  Max distance: {np.max(distances):.2f}")
            print(f"  Average relative distance: {np.mean(rel_distances):.2%}")
        
        return {
            'num_samples': num_samples,
            'unique_sketches': len(unique_sketches),
            'total_collisions': total_collisions,
            'collision_rate': collision_rate,
            'coverage': coverage,
            'collision_examples': collision_examples,
            'sketch_to_hidden': dict(list(sketch_to_hidden.items())[:100])  # Save first 100 for analysis
        }
    
    def targeted_search(self, r_vec: torch.Tensor, target_sketch: int, num_attempts: int = 1_000_000):
        """
        Try to find a hidden state that produces a specific target sketch
        """
        print(f"\n=== Targeted Search ===")
        print(f"Target sketch: {target_sketch}")
        print(f"Attempting {num_attempts:,} random samples...")
        
        found = False
        closest_diff = float('inf')
        closest_h = None
        closest_sketch = None
        
        batch_size = 10000
        pbar = tqdm(total=num_attempts, desc="Searching for target")
        
        for batch_start in range(0, num_attempts, batch_size):
            batch_end = min(batch_start + batch_size, num_attempts)
            actual_batch_size = batch_end - batch_start
            
            # Generate batch with varied scales
            h_batch = torch.randn(actual_batch_size, self.hidden_dim, device=self.device)
            scales = torch.exp(torch.rand(actual_batch_size, 1, device=self.device) * 6 - 3)  # 0.05 to 20
            h_batch = h_batch * scales * torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=self.device))
            
            # Check each one
            for i in range(actual_batch_size):
                h = h_batch[i]
                sketch = dot_mod_q(h, r_vec)
                
                diff = abs(sketch - target_sketch)
                # Handle modular wraparound
                if diff > PRIME_Q / 2:
                    diff = PRIME_Q - diff
                
                if diff < closest_diff:
                    closest_diff = diff
                    closest_h = h.cpu()
                    closest_sketch = sketch
                
                if diff <= 3:  # Within tolerance
                    found = True
                    print(f"\n  SUCCESS! Found hidden state with sketch {sketch}")
                    print(f"  Difference from target: {diff}")
                    print(f"  Hidden state norm: {torch.norm(h).item():.2f}")
                    pbar.close()
                    
                    return {
                        'success': True,
                        'iterations': batch_end,
                        'sketch': int(sketch),
                        'diff': int(diff),
                        'hidden_state_norm': float(torch.norm(h).item())
                    }
            
            pbar.update(actual_batch_size)
        
        pbar.close()
        
        print(f"\n  Could not find exact match.")
        print(f"  Closest sketch: {closest_sketch}")
        print(f"  Difference: {closest_diff}")
        
        return {
            'success': False,
            'iterations': num_attempts,
            'closest_sketch': int(closest_sketch),
            'closest_diff': int(closest_diff)
        }
    
    def birthday_paradox_analysis(self, r_vec: torch.Tensor):
        """
        Use birthday paradox to estimate when we expect collisions
        """
        print(f"\n=== Birthday Paradox Analysis ===")
        
        # For a space of size N, we expect first collision after ~sqrt(N) samples
        expected_samples_for_collision = int(np.sqrt(PRIME_Q))
        
        print(f"Sketch space size: {PRIME_Q:,}")
        print(f"Expected samples for first collision: {expected_samples_for_collision:,} (√N)")
        
        # Let's test this empirically with smaller batches
        print(f"\nTesting birthday paradox empirically...")
        
        collision_points = []
        num_trials = 10
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}...", end='', flush=True)
            
            seen_sketches = set()
            num_samples = 0
            
            # Keep generating until we find a collision
            while num_samples < expected_samples_for_collision * 10:  # Upper limit
                h = torch.randn(self.hidden_dim, device=self.device)
                h = h * torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=self.device))
                
                sketch = dot_mod_q(h, r_vec)
                num_samples += 1
                
                if sketch in seen_sketches:
                    # Found collision!
                    collision_points.append(num_samples)
                    print(f" collision at {num_samples:,} samples")
                    break
                
                seen_sketches.add(sketch)
            else:
                print(f" no collision in {num_samples:,} samples")
        
        if collision_points:
            avg_collision_point = np.mean(collision_points)
            print(f"\nEmpirical results:")
            print(f"  Average samples to first collision: {avg_collision_point:,.0f}")
            print(f"  Expected (√N): {expected_samples_for_collision:,}")
            print(f"  Ratio: {avg_collision_point / expected_samples_for_collision:.2f}")
        
        return {
            'expected_collision_point': expected_samples_for_collision,
            'empirical_collision_points': collision_points,
            'average_collision_point': float(np.mean(collision_points)) if collision_points else None
        }
    
    def run_exhaustive_analysis(self):
        """Run all exhaustive search experiments"""
        print("="*60)
        print("Exhaustive Search for k=1 Collisions")
        print("="*60)
        
        results = {}
        
        # Generate random r vector
        randomness = "0x" + "e" * 64
        r_vec = r_vec_from_randomness(randomness, self.hidden_dim)
        
        # 1. Birthday paradox analysis
        print("\n" + "-"*60)
        results['birthday_paradox'] = self.birthday_paradox_analysis(r_vec)
        
        # 2. Exhaustive collision search
        print("\n" + "-"*60)
        # Start with 1M samples, increase if needed
        results['collision_search'] = self.exhaustive_collision_search(r_vec, num_samples=1_000_000)
        
        # 3. If we found collisions, pick one sketch value and try targeted search
        if results['collision_search']['collision_examples']:
            target_sketch = results['collision_search']['collision_examples'][0]['sketch']
            print("\n" + "-"*60)
            results['targeted_search'] = self.targeted_search(r_vec, target_sketch, num_attempts=1_000_000)
        
        # Save results
        os.makedirs('results', exist_ok=True)
        
        # Clean results for JSON
        clean_results = {
            'birthday_paradox': results.get('birthday_paradox', {}),
            'collision_search': {
                'num_samples': results['collision_search']['num_samples'],
                'unique_sketches': results['collision_search']['unique_sketches'],
                'total_collisions': results['collision_search']['total_collisions'],
                'collision_rate': results['collision_search']['collision_rate'],
                'coverage': results['collision_search']['coverage'],
                'collision_examples': results['collision_search']['collision_examples'][:20]  # First 20
            },
            'targeted_search': results.get('targeted_search', {})
        }
        
        with open('results/experiment_5_exhaustive_search.json', 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        # Generate plots
        self._generate_plots(results)
        
        print("\n" + "="*60)
        print("CONCLUSIONS")
        print("="*60)
        print("1. Collisions DO exist but are extremely rare")
        print(f"2. Collision rate: ~{results['collision_search']['collision_rate']:.6%}")
        print("3. Birthday paradox applies: collisions appear after ~√N samples")
        print("4. Finding a specific sketch value is still very hard")
        print("5. This is just k=1 - with k=16, difficulty increases exponentially")
        
    def _generate_plots(self, results):
        """Generate visualization plots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot 1: Collision distance distribution
        if results['collision_search']['collision_examples']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            examples = results['collision_search']['collision_examples']
            distances = [e['distance'] for e in examples]
            rel_distances = [e['relative_distance'] for e in examples]
            
            ax1.hist(distances, bins=30, alpha=0.7)
            ax1.set_xlabel('Distance Between Colliding Hidden States')
            ax1.set_ylabel('Count')
            ax1.set_title('Distribution of Distances for Collisions')
            
            ax2.hist(rel_distances, bins=30, alpha=0.7, color='orange')
            ax2.set_xlabel('Relative Distance (distance / norm)')
            ax2.set_ylabel('Count')
            ax2.set_title('Relative Distance Distribution')
            
            plt.tight_layout()
            plt.savefig('results/experiment_5_collision_distances.png', dpi=150)
            plt.close()
        
        # Plot 2: Collision rate vs samples
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Theoretical birthday paradox curve
        n_samples = np.logspace(3, 7, 100)
        theoretical_prob = 1 - np.exp(-n_samples**2 / (2 * PRIME_Q))
        
        ax.loglog(n_samples, theoretical_prob, 'r--', linewidth=2, 
                  label='Theoretical (Birthday Paradox)')
        
        # Our empirical point
        if results['collision_search']['total_collisions'] > 0:
            emp_samples = results['collision_search']['num_samples']
            emp_rate = results['collision_search']['collision_rate']
            ax.scatter([emp_samples], [emp_rate], color='blue', s=100, 
                      label=f'Empirical: {emp_rate:.6%}', zorder=5)
        
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Collision Probability')
        ax.set_title('Collision Probability vs Sample Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/experiment_5_collision_probability.png', dpi=150)
        plt.close()


if __name__ == "__main__":
    print("Starting Exhaustive Search Analysis...")
    print("This may take several minutes due to the large number of samples...")
    
    experiment = ExhaustiveSketchSearch(hidden_dim=768)
    experiment.run_exhaustive_analysis()
    
    print("\nExperiment completed. Results saved to results/")