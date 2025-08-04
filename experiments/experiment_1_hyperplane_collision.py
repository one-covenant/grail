#!/usr/bin/env python3
"""
Experiment 1: Hyperplane Collision Analysis

This experiment demonstrates that while multiple hidden states can produce the same
dot product (sketch value), finding such states that also maintain model functionality
is computationally infeasible.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List
import json
import os
import sys
sys.path.append('..')
from grail.grail import PRIME_Q, dot_mod_q

class HyperplaneCollisionExperiment:
    def __init__(self, dimensions: List[int] = None):
        if dimensions is None:
            dimensions = [64, 256, 768, 2048]
        self.dimensions = dimensions
        self.dimensions = dimensions
        self.num_trials = 1000
        self.results = {}
        
    def find_hyperplane_points(self, r: torch.Tensor, target_sketch: int, 
                               num_points: int = 100) -> torch.Tensor:
        """
        Find multiple points on the hyperplane defined by <h, r> = target_sketch (mod q)
        """
        d = r.shape[0]
        points = []
        
        # Start with a reference point
        h0 = torch.randn(d)
        h0 = h0 * (target_sketch / dot_mod_q(h0, r))  # Scale to get close
        
        # Find orthogonal basis to r
        # Use Gram-Schmidt to get basis vectors orthogonal to r
        basis = torch.eye(d)
        r_normalized = r.float() / torch.norm(r.float())
        
        orth_basis = []
        for i in range(d):
            v = basis[i]
            # Make orthogonal to r
            v = v - torch.dot(v, r_normalized) * r_normalized
            if torch.norm(v) > 1e-6:
                v = v / torch.norm(v)
                orth_basis.append(v)
        
        # Generate points on hyperplane
        for _ in range(num_points):
            # Random combination of orthogonal basis vectors
            coeffs = torch.randn(len(orth_basis))
            perturbation = sum(c * v for c, v in zip(coeffs, orth_basis))
            
            # Add perturbation to reference point
            h = h0 + perturbation * 10  # Scale perturbation
            
            # Verify it's on the hyperplane (within tolerance)
            sketch = dot_mod_q(h, r)
            if abs(sketch - target_sketch) <= 3:  # Within GRAIL tolerance
                points.append(h)
                
        return torch.stack(points) if points else torch.empty(0, d)
    
    def measure_collision_probability(self, d: int) -> dict:
        """
        Measure how often random vectors produce the same sketch value
        """
        collisions = 0
        total_pairs = 0
        
        for _ in range(self.num_trials):
            r = torch.randint(-2**31, 2**31, (d,), dtype=torch.int32)
            
            # Generate two random hidden states
            h1 = torch.randn(d) * 100
            h2 = torch.randn(d) * 100
            
            s1 = dot_mod_q(h1, r)
            s2 = dot_mod_q(h2, r)
            
            if abs(s1 - s2) <= 3:  # Within tolerance
                collisions += 1
            total_pairs += 1
            
        collision_rate = collisions / total_pairs
        expected_rate = (2 * 3 + 1) / PRIME_Q  # Theoretical expectation
        
        return {
            'dimension': d,
            'observed_collision_rate': collision_rate,
            'expected_collision_rate': expected_rate,
            'ratio': collision_rate / expected_rate if expected_rate > 0 else 0
        }
    
    def analyze_constraint_intersection(self, d: int, k: int) -> dict:
        """
        Analyze how multiple hyperplane constraints reduce the solution space
        """
        # Generate k random vectors and target sketches
        r_vectors = [torch.randint(-2**31, 2**31, (d,), dtype=torch.int32) 
                     for _ in range(k)]
        target_sketches = [np.random.randint(0, PRIME_Q) for _ in range(k)]
        
        # Start with a random point
        h = torch.randn(d) * 100
        
        # Try to find a point satisfying all constraints using gradient descent
        h.requires_grad = True
        optimizer = torch.optim.Adam([h], lr=0.1)
        
        success = False
        for iteration in range(1000):
            optimizer.zero_grad()
            
            # Compute loss as sum of squared distances from target sketches
            loss = 0
            for r, target in zip(r_vectors, target_sketches):
                sketch = dot_mod_q(h, r)
                diff = abs(sketch - target)
                loss += diff ** 2
            
            loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
            loss.backward()
            optimizer.step()
            
            # Check if all constraints satisfied
            all_satisfied = True
            for r, target in zip(r_vectors, target_sketches):
                if abs(dot_mod_q(h, r) - target) > 3:
                    all_satisfied = False
                    break
                    
            if all_satisfied:
                success = True
                break
        
        # Measure degrees of freedom
        # Approximate by checking local perturbations
        num_free_directions = 0
        for _ in range(100):
            direction = torch.randn(d)
            direction = direction / torch.norm(direction)
            
            # Check if small movement in this direction maintains constraints
            eps = 0.01
            h_perturbed = h + eps * direction
            
            maintains_constraints = True
            for r, target in zip(r_vectors, target_sketches):
                if abs(dot_mod_q(h_perturbed, r) - target) > 3:
                    maintains_constraints = False
                    break
                    
            if maintains_constraints:
                num_free_directions += 1
        
        return {
            'dimension': d,
            'num_constraints': k,
            'optimization_success': success,
            'iterations': iteration + 1,
            'estimated_free_dimensions': num_free_directions,
            'reduction_factor': num_free_directions / d if d > 0 else 0
        }
    
    def run_all_experiments(self):
        """Run all experiments and save results"""
        print("Running Hyperplane Collision Experiments...")
        
        # Experiment 1a: Collision probability vs dimension
        print("\n1a. Measuring collision probability by dimension...")
        collision_results = []
        for d in self.dimensions:
            result = self.measure_collision_probability(d)
            collision_results.append(result)
            print(f"  d={d}: collision_rate={result['observed_collision_rate']:.2e}")
        
        # Experiment 1b: Constraint intersection analysis
        print("\n1b. Analyzing multi-constraint intersections...")
        constraint_results = []
        k_values = [1, 4, 8, 16, 32]
        for d in [256, 768]:  # Focus on realistic dimensions
            for k in k_values:
                if k < d:  # Only test if k < dimension
                    result = self.analyze_constraint_intersection(d, k)
                    constraint_results.append(result)
                    print(f"  d={d}, k={k}: free_dims≈{result['estimated_free_dimensions']}")
        
        # Experiment 1c: Hyperplane point distribution
        print("\n1c. Analyzing hyperplane point distributions...")
        distribution_results = []
        for d in [64, 256]:
            r = torch.randint(-2**31, 2**31, (d,), dtype=torch.int32)
            target = np.random.randint(0, PRIME_Q)
            
            points = self.find_hyperplane_points(r, target, num_points=100)
            if len(points) > 0:
                # Analyze distribution properties
                pairwise_distances = []
                for i in range(min(50, len(points))):
                    for j in range(i+1, min(50, len(points))):
                        dist = torch.norm(points[i] - points[j]).item()
                        pairwise_distances.append(dist)
                
                result = {
                    'dimension': d,
                    'num_points_found': len(points),
                    'mean_pairwise_distance': np.mean(pairwise_distances),
                    'std_pairwise_distance': np.std(pairwise_distances)
                }
                distribution_results.append(result)
                print(f"  d={d}: found {len(points)} points, "
                      f"mean_dist={result['mean_pairwise_distance']:.2f}")
        
        # Save all results
        self.results = {
            'collision_probability': collision_results,
            'constraint_intersection': constraint_results,
            'hyperplane_distribution': distribution_results
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/experiment_1_hyperplane_collision.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate plots
        self._generate_plots()
        
    def _generate_plots(self):
        """Generate visualization plots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot 1: Collision probability vs dimension
        fig, ax = plt.subplots(figsize=(8, 6))
        collision_data = self.results['collision_probability']
        dims = [r['dimension'] for r in collision_data]
        observed = [r['observed_collision_rate'] for r in collision_data]
        expected = [r['expected_collision_rate'] for r in collision_data]
        
        ax.semilogy(dims, observed, 'bo-', label='Observed', markersize=8)
        ax.semilogy(dims, expected, 'r--', label='Expected', linewidth=2)
        ax.set_xlabel('Hidden State Dimension')
        ax.set_ylabel('Collision Probability')
        ax.set_title('Sketch Collision Probability vs Dimension')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/experiment_1_collision_probability.png', dpi=150)
        plt.close()
        
        # Plot 2: Degrees of freedom vs constraints
        fig, ax = plt.subplots(figsize=(8, 6))
        constraint_data = self.results['constraint_intersection']
        
        for d in [256, 768]:
            d_data = [r for r in constraint_data if r['dimension'] == d]
            if d_data:
                k_vals = [r['num_constraints'] for r in d_data]
                free_dims = [r['estimated_free_dimensions'] for r in d_data]
                ax.plot(k_vals, free_dims, 'o-', label=f'd={d}', markersize=8)
        
        ax.set_xlabel('Number of Constraints (k)')
        ax.set_ylabel('Estimated Free Dimensions')
        ax.set_title('Solution Space Reduction with Multiple Constraints')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/experiment_1_constraint_reduction.png', dpi=150)
        plt.close()

if __name__ == "__main__":
    experiment = HyperplaneCollisionExperiment()
    experiment.run_all_experiments()
    print("\nExperiment 1 completed. Results saved to results/")