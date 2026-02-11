"""
Pulse Propagation Module with Custom CUDA/MPS Kernels

This module provides optimized pulse propagation through the lattice using
custom CUDA kernels (NVIDIA GPUs) or MPS operations (Apple Silicon).

Usage:
    from pulse_propagation import AcceleratedDifferentiableLattice
    
    # Replace DifferentiableLattice with accelerated version
    diff_lattice = AcceleratedDifferentiableLattice(lattice, max_steps=15)
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import numpy as np

# Import custom kernels
from kernels import create_kernels, PulsePropagationKernels


class AcceleratedDifferentiableLattice(nn.Module):
    """
    Differentiable lattice implementation with custom CUDA/MPS kernels.
    
    Drop-in replacement for DifferentiableLattice with optimized propagation.
    """
    
    def __init__(
        self, 
        lattice,
        max_steps: int = 10,
        use_custom_kernels: bool = True
    ):
        """
        Initialize accelerated differentiable lattice.
        
        Args:
            lattice: LatticeVisualizer instance containing cells
            max_steps: Maximum number of propagation steps
            use_custom_kernels: Whether to use custom CUDA/MPS kernels
        """
        super().__init__()
        
        self.lattice = lattice
        self.n_cells = len(lattice.cells)
        self.max_steps = max_steps
        self.device = lattice.device
        self.use_custom_kernels = use_custom_kernels
        
        # Initialize custom kernels
        if use_custom_kernels:
            self.kernels = create_kernels(self.device)
            print(f"Using {self.kernels.backend} backend for propagation")
        else:
            self.kernels = None
        
        # Extract cell parameters as learnable tensors
        means = torch.stack([cell.mean for cell in lattice.cells]).squeeze()
        std_devs = torch.stack([cell.std_dev for cell in lattice.cells]).squeeze()
        bounce_angles = torch.stack([cell.bounce_angles for cell in lattice.cells])
        
        self.means = nn.Parameter(means)
        self.std_devs = nn.Parameter(std_devs)
        self.bounce_angles = nn.Parameter(bounce_angles)
        
        # Learnable propagation step parameters
        self._init_step_parameters()
        
        # Pre-compute and register adjacency matrix
        self.register_buffer('adjacency', self._build_adjacency())
        self.register_buffer('positions', lattice.positions.clone())
    
    def _init_step_parameters(self):
        """Initialize learnable parameters for adaptive step computation."""
        # Step weights - how much each step contributes to final output
        self.step_weights = nn.Parameter(torch.ones(self.max_steps + 1, device=self.device))
        
        # Effective number of steps (learnable, allows fractional steps)
        self.effective_steps = nn.Parameter(
            torch.tensor(float(self.max_steps) / 2, device=self.device)
        )
        
        # Decay rate per step
        self.decay_rate = nn.Parameter(
            torch.tensor(0.85, device=self.device)
        )
    
    def _build_adjacency(self) -> torch.Tensor:
        """Build adjacency matrix based on cell positions."""
        positions = self.lattice.positions
        n = len(positions)
        
        # Compute pairwise distances
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        distances = torch.norm(diff, dim=2)
        
        # Adjacent cells are within 1.5 * hex_spacing
        threshold = 1.5 * self.lattice.hex_spacing
        adjacency = (distances < threshold).float()
        
        # Remove self-connections
        adjacency = adjacency - torch.eye(n, device=self.device)
        
        # Normalize rows
        row_sums = adjacency.sum(dim=1, keepdim=True).clamp(min=1)
        adjacency = adjacency / row_sums
        
        return adjacency
    
    def compute_propagation_weights(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute propagation weights using custom kernels if available.
        
        Args:
            activations: Current cell activations (batch_size, n_cells)
            
        Returns:
            Propagation weights (n_cells, n_cells)
        """
        if self.use_custom_kernels and self.kernels is not None:
            return self.kernels.compute_propagation_weights(
                self.std_devs, self.adjacency, threshold=0.5
            )
        else:
            # Fallback to standard implementation
            threshold = 0.3
            std_devs_safe = torch.clamp(torch.abs(self.std_devs), min=0.3)
            straight_prob = torch.erf(threshold / (std_devs_safe * np.sqrt(2)))
            
            identity = torch.diag(straight_prob * 0.3)
            spread_factor = (1 - straight_prob).unsqueeze(1)
            neighbor_spread = self.adjacency * spread_factor
            
            propagation = identity + neighbor_spread * 0.7
            propagation = propagation / propagation.sum(dim=1, keepdim=True).clamp(min=1e-6)
            
            return propagation
    
    def apply_bounce_transform(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Apply bounce transformation using custom kernels if available.
        
        Args:
            activations: Cell activations (batch_size, n_cells)
            
        Returns:
            Transformed activations (batch_size, n_cells)
        """
        if self.use_custom_kernels and self.kernels is not None:
            return self.kernels.apply_bounce_transform(activations, self.bounce_angles)
        else:
            # Angles are in radians (max 2 rad each)
            angle_magnitude = torch.abs(self.bounce_angles).mean(dim=1)
            angle_factor = torch.cos(angle_magnitude)  # Already in radians
            transformed = activations * (0.5 + 0.5 * angle_factor.unsqueeze(0))
            return transformed
    
    def forward(
        self, 
        initial_activations: torch.Tensor, 
        n_steps: int = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Differentiable forward pass through the lattice with custom kernels.
        
        Args:
            initial_activations: Initial cell activations (batch_size, n_cells)
            n_steps: Number of propagation steps (None = use learnable effective_steps)
            
        Returns:
            final_state: Final cell states (weighted by learnable step_weights)
            history: List of intermediate states
        """
        state = initial_activations.to(self.device)
        history = [state]
        
        # Determine actual number of steps
        if n_steps is None:
            effective = torch.clamp(self.effective_steps, min=1.0, max=float(self.max_steps))
            actual_steps = int(torch.ceil(effective).item())
        else:
            actual_steps = min(n_steps, self.max_steps)
        
        # Get learnable decay rate
        decay = torch.clamp(self.decay_rate, min=0.001, max=1.0)
        decay_val = decay.item()
        
        # Compute propagation weights once (they don't change during propagation)
        prop_weights = self.compute_propagation_weights(state)
        
        # Propagate for actual_steps using fused kernel if available
        for step in range(actual_steps):
            state_max = state.max().item()
            
            if self.use_custom_kernels and self.kernels is not None:
                # Use fused kernel for maximum efficiency
                state = self.kernels.fused_propagation_step(
                    state, prop_weights, self.bounce_angles, decay_val, state_max
                )
            else:
                # Fallback implementation
                state = torch.matmul(state, prop_weights.T)
                state = self.apply_bounce_transform(state)
                state = state * decay
                state = torch.LeakyReLU(state * 2 - 1) * max(state_max, 0.1)
            
            history.append(state)
        
        # Compute weighted output using learnable step_weights
        if len(history) > 1:
            weights = torch.softmax(self.step_weights[:len(history)], dim=0)
            
            # Soft step masking based on effective_steps
            if n_steps is None:
                effective = torch.clamp(self.effective_steps, min=1.0, max=float(self.max_steps))
                step_mask = torch.zeros(len(history), device=self.device)
                for i in range(len(history)):
                    step_mask[i] = torch.LeakyReLU(5.0 * (effective - i))
                weights = weights * step_mask
                weights = weights / (weights.sum() + 1e-8)
            
            # Use custom kernel for weighted sum if available
            if self.use_custom_kernels and self.kernels is not None:
                history_tensor = torch.stack(history, dim=0)
                final_state = self.kernels.weighted_history_sum(history_tensor, weights)
            else:
                final_state = sum(w * h for w, h in zip(weights, history))
        else:
            final_state = state
        
        return final_state, history
    
    def get_effective_steps(self) -> float:
        """Get current effective number of steps."""
        return torch.clamp(self.effective_steps, min=1.0, max=float(self.max_steps)).item()
    
    def get_step_weights(self) -> torch.Tensor:
        """Get normalized step weights."""
        return torch.softmax(self.step_weights, dim=0)
    
    def sync_to_lattice(self):
        """Sync learned parameters back to the original lattice cells."""
        with torch.no_grad():
            for i, cell in enumerate(self.lattice.cells):
                cell.mean = self.means[i:i+1].clone()
                cell.std_dev = torch.clamp(torch.abs(self.std_devs[i:i+1].clone()), min=0.3)
                cell.bounce_angles = self.bounce_angles[i].clone()
    
    def get_learned_parameters_summary(self) -> dict:
        """Get a summary of all learned parameters including step parameters."""
        with torch.no_grad():
            return {
                'effective_steps': self.get_effective_steps(),
                'decay_rate': torch.clamp(self.decay_rate, min=0.5, max=0.99).item(),
                'step_weights': self.get_step_weights().cpu().numpy(),
                'mean_avg': self.means.mean().item(),
                'mean_std': self.means.std().item(),
                'std_dev_avg': self.std_devs.mean().item(),
                'std_dev_std': self.std_devs.std().item(),
                'bounce_angles_avg': self.bounce_angles.abs().mean().item(),
                'bounce_angles_std': self.bounce_angles.std().item(),
                'backend': self.kernels.backend if self.kernels else 'pytorch'
            }


def benchmark_kernels(
    n_cells: int = 1000,
    batch_size: int = 64,
    n_steps: int = 10,
    n_iterations: int = 100,
    device: Optional[torch.device] = None
) -> dict:
    """
    Benchmark custom kernels vs PyTorch implementations.
    
    Args:
        n_cells: Number of cells in lattice
        batch_size: Batch size for testing
        n_steps: Number of propagation steps
        n_iterations: Number of benchmark iterations
        device: Device to test on
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print(f"\n{'='*60}")
    print(f"Benchmarking on {device}")
    print(f"n_cells={n_cells}, batch_size={batch_size}, n_steps={n_steps}")
    print(f"{'='*60}\n")
    
    # Create test data
    state = torch.randn(batch_size, n_cells, device=device)
    std_devs = torch.rand(n_cells, device=device) * 0.5 + 0.5
    bounce_angles = torch.rand(n_cells, 3, device=device) * 2.0  # Max 2 radians
    adjacency = torch.rand(n_cells, n_cells, device=device)
    adjacency = (adjacency > 0.9).float()  # Sparse adjacency
    
    results = {}
    
    # Benchmark custom kernels
    kernels = create_kernels(device)
    
    # Warmup
    for _ in range(10):
        _ = kernels.compute_propagation_weights(std_devs, adjacency)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
    
    # Benchmark propagation weights
    start = time.perf_counter()
    for _ in range(n_iterations):
        prop = kernels.compute_propagation_weights(std_devs, adjacency)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    results['propagation_weights_ms'] = (elapsed / n_iterations) * 1000
    print(f"Propagation weights: {results['propagation_weights_ms']:.3f} ms")
    
    # Benchmark fused propagation step
    prop = kernels.compute_propagation_weights(std_devs, adjacency)
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = kernels.fused_propagation_step(state, prop, bounce_angles, 0.85, 1.0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    results['fused_step_ms'] = (elapsed / n_iterations) * 1000
    print(f"Fused propagation step: {results['fused_step_ms']:.3f} ms")
    
    # Benchmark full propagation
    start = time.perf_counter()
    for _ in range(n_iterations // 10):  # Fewer iterations for full propagation
        final, history = kernels.full_propagation(
            state, std_devs, adjacency, bounce_angles, 0.85, n_steps
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    results['full_propagation_ms'] = (elapsed / (n_iterations // 10)) * 1000
    print(f"Full propagation ({n_steps} steps): {results['full_propagation_ms']:.3f} ms")
    
    # Throughput
    samples_per_second = batch_size / (results['full_propagation_ms'] / 1000)
    results['throughput_samples_per_sec'] = samples_per_second
    print(f"\nThroughput: {samples_per_second:.0f} samples/sec")
    
    results['device'] = str(device)
    results['backend'] = kernels.backend
    
    return results


if __name__ == '__main__':
    # Run benchmark when executed directly
    benchmark_kernels()
