"""
Kernel module for 3D Cell Pulse Propagation

Provides optimized CUDA and MPS (Metal) kernels for lattice propagation.
Automatically selects the appropriate backend based on available hardware.
"""

import torch
import os
import warnings
from typing import Optional, Tuple, List

# Try to load custom CUDA kernels
_cuda_available = False
_cuda_kernels = None

try:
    if torch.cuda.is_available():
        from torch.utils.cpp_extension import load
        
        cuda_src = os.path.join(os.path.dirname(__file__), 'cuda_kernels.cu')
        if os.path.exists(cuda_src):
            try:
                _cuda_kernels = load(
                    name='pulse_propagation_cuda',
                    sources=[cuda_src],
                    extra_cuda_cflags=['-O3', '--use_fast_math'],
                    verbose=False
                )
                _cuda_available = True
                print("Custom CUDA kernels loaded successfully")
            except Exception as e:
                warnings.warn(f"Failed to compile CUDA kernels: {e}")
except ImportError:
    pass


# MPS backend using PyTorch's MPS support
_mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False


def get_backend() -> str:
    """Get the currently available backend."""
    if _cuda_available:
        return 'cuda'
    elif _mps_available:
        return 'mps'
    else:
        return 'cpu'


class PulsePropagationKernels:
    """
    Unified interface for pulse propagation kernels.
    
    Automatically uses the best available backend (CUDA > MPS > CPU).
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize kernels for the specified device.
        
        Args:
            device: Target device. If None, auto-detects best available.
        """
        if device is None:
            if _cuda_available:
                device = torch.device('cuda')
            elif _mps_available:
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        
        self.device = device
        self.backend = self._determine_backend()
        print(f"PulsePropagationKernels initialized with backend: {self.backend}")
    
    def _determine_backend(self) -> str:
        """Determine which backend to use based on device."""
        if self.device.type == 'cuda' and _cuda_available:
            return 'cuda'
        elif self.device.type == 'mps':
            return 'mps'
        else:
            return 'cpu'
    
    def compute_propagation_weights(
        self,
        std_devs: torch.Tensor,
        adjacency: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Compute propagation weights based on cell parameters.
        
        Args:
            std_devs: Standard deviations for each cell (n_cells,)
            adjacency: Adjacency matrix (n_cells, n_cells)
            threshold: Threshold for straight probability computation
            
        Returns:
            Propagation matrix (n_cells, n_cells)
        """
        if self.backend == 'cuda' and _cuda_kernels is not None:
            return _cuda_kernels.compute_propagation_weights(
                std_devs.contiguous(),
                adjacency.contiguous(),
                threshold
            )
        else:
            # Fallback to optimized PyTorch implementation
            return self._compute_propagation_weights_torch(std_devs, adjacency, threshold)
    
    def _compute_propagation_weights_torch(
        self,
        std_devs: torch.Tensor,
        adjacency: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """PyTorch fallback for propagation weight computation."""
        import math
        
        n_cells = std_devs.size(0)
        std_devs_safe = torch.clamp(torch.abs(std_devs), min=0.5)
        straight_prob = torch.erf(threshold / (std_devs_safe * math.sqrt(2)))
        
        identity = torch.diag(straight_prob * 0.3)
        spread_factor = (1 - straight_prob).unsqueeze(1)
        neighbor_spread = adjacency * spread_factor
        
        propagation = identity + neighbor_spread * 0.7
        propagation = propagation / propagation.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        return propagation
    
    def apply_bounce_transform(
        self,
        activations: torch.Tensor,
        bounce_angles: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply bounce transformation to activations.
        
        Args:
            activations: Cell activations (batch_size, n_cells)
            bounce_angles: Bounce angles per cell (n_cells, 3)
            
        Returns:
            Transformed activations (batch_size, n_cells)
        """
        if self.backend == 'cuda' and _cuda_kernels is not None:
            return _cuda_kernels.apply_bounce_transform(
                activations.contiguous(),
                bounce_angles.contiguous()
            )
        else:
            return self._apply_bounce_transform_torch(activations, bounce_angles)
    
    def _apply_bounce_transform_torch(
        self,
        activations: torch.Tensor,
        bounce_angles: torch.Tensor
    ) -> torch.Tensor:
        """PyTorch fallback for bounce transformation."""
        # Angles are in radians (max 2 rad each)
        angle_magnitude = torch.abs(bounce_angles).mean(dim=1)
        angle_factor = torch.cos(angle_magnitude)  # Already in radians
        transformed = activations * (0.5 + 0.5 * angle_factor.unsqueeze(0))
        return transformed
    
    def apply_decay_sigmoid(
        self,
        state: torch.Tensor,
        decay: float,
        state_max: float
    ) -> torch.Tensor:
        """
        Apply decay and sigmoid nonlinearity.
        
        Args:
            state: Current state (batch_size, n_cells)
            decay: Decay factor (0.5-0.99)
            state_max: Maximum state value for normalization
            
        Returns:
            Transformed state (batch_size, n_cells)
        """
        if self.backend == 'cuda' and _cuda_kernels is not None:
            return _cuda_kernels.apply_decay_sigmoid(
                state.contiguous(),
                decay,
                state_max
            )
        else:
            return self._apply_decay_sigmoid_torch(state, decay, state_max)
    
    def _apply_decay_sigmoid_torch(
        self,
        state: torch.Tensor,
        decay: float,
        state_max: float
    ) -> torch.Tensor:
        """PyTorch fallback for decay and sigmoid."""
        state = state * decay
        state = torch.sigmoid(state * 2 - 1) * max(state_max, 0.1)
        return state
    
    def propagate_state(
        self,
        state: torch.Tensor,
        propagation: torch.Tensor
    ) -> torch.Tensor:
        """
        Propagate state through lattice.
        
        Computes: output = state @ propagation^T
        
        Args:
            state: Current state (batch_size, n_cells)
            propagation: Propagation matrix (n_cells, n_cells)
            
        Returns:
            Propagated state (batch_size, n_cells)
        """
        if self.backend == 'cuda' and _cuda_kernels is not None:
            return _cuda_kernels.propagate_state(
                state.contiguous(),
                propagation.contiguous()
            )
        else:
            return torch.matmul(state, propagation.T)
    
    def fused_propagation_step(
        self,
        state_in: torch.Tensor,
        propagation: torch.Tensor,
        bounce_angles: torch.Tensor,
        decay: float,
        state_max: float
    ) -> torch.Tensor:
        """
        Fused propagation step combining all operations.
        
        More efficient than calling individual kernels.
        
        Args:
            state_in: Input state (batch_size, n_cells)
            propagation: Propagation matrix (n_cells, n_cells)
            bounce_angles: Bounce angles (n_cells, 3)
            decay: Decay factor
            state_max: Maximum state value
            
        Returns:
            Output state (batch_size, n_cells)
        """
        if self.backend == 'cuda' and _cuda_kernels is not None:
            return _cuda_kernels.fused_propagation_step(
                state_in.contiguous(),
                propagation.contiguous(),
                bounce_angles.contiguous(),
                decay,
                state_max
            )
        else:
            # Fallback: call individual operations
            state = self.propagate_state(state_in, propagation)
            state = self.apply_bounce_transform(state, bounce_angles)
            state = self.apply_decay_sigmoid(state, decay, state_max)
            return state
    
    def weighted_history_sum(
        self,
        history: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted sum of history states.
        
        Args:
            history: History tensor (n_steps, batch_size, n_cells)
            weights: Normalized weights (n_steps,)
            
        Returns:
            Weighted sum (batch_size, n_cells)
        """
        if self.backend == 'cuda' and _cuda_kernels is not None:
            return _cuda_kernels.weighted_history_sum(
                history.contiguous(),
                weights.contiguous()
            )
        else:
            # Efficient einsum operation
            return torch.einsum('sbc,s->bc', history, weights)
    
    def full_propagation(
        self,
        initial_state: torch.Tensor,
        std_devs: torch.Tensor,
        adjacency: torch.Tensor,
        bounce_angles: torch.Tensor,
        decay: float,
        n_steps: int,
        step_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Full propagation through the lattice.
        
        Args:
            initial_state: Initial activations (batch_size, n_cells)
            std_devs: Cell standard deviations (n_cells,)
            adjacency: Adjacency matrix (n_cells, n_cells)
            bounce_angles: Bounce angles (n_cells, 3)
            decay: Decay factor
            n_steps: Number of propagation steps
            step_weights: Optional weights for combining steps
            
        Returns:
            final_state: Final state (optionally weighted)
            history: List of intermediate states
        """
        # Compute propagation weights once
        propagation = self.compute_propagation_weights(std_devs, adjacency)
        
        state = initial_state.to(self.device)
        history = [state]
        
        for step in range(n_steps):
            state_max = state.max().item()
            state = self.fused_propagation_step(
                state, propagation, bounce_angles, decay, state_max
            )
            history.append(state)
        
        # Compute weighted output if weights provided
        if step_weights is not None and len(history) > 1:
            weights = torch.softmax(step_weights[:len(history)], dim=0)
            history_tensor = torch.stack(history, dim=0)
            final_state = self.weighted_history_sum(history_tensor, weights)
        else:
            final_state = state
        
        return final_state, history


# MPS-specific implementation using PyTorch MPS operations
class MPSPulsePropagationKernels(PulsePropagationKernels):
    """
    MPS-optimized implementation using Metal Performance Shaders.
    
    Uses PyTorch's MPS backend with optimized memory patterns for M1/M2/M3.
    """
    
    def __init__(self):
        super().__init__(device=torch.device('mps'))
        self._compile_metal_shaders()
    
    def _compile_metal_shaders(self):
        """
        Compile Metal shaders for MPS.
        
        Note: PyTorch's MPS backend handles most operations efficiently.
        For maximum performance, we use contiguous memory layouts and
        avoid unnecessary copies.
        """
        self._metal_compiled = False
        
        # Check if we can use Metal directly via PyObjC (advanced)
        try:
            import Metal
            import MetalPerformanceShaders as MPS
            self._metal_compiled = True
            print("Metal framework available for advanced optimizations")
        except ImportError:
            # Fall back to PyTorch MPS operations
            pass
    
    def propagate_state(
        self,
        state: torch.Tensor,
        propagation: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimized matrix multiplication for MPS.
        
        Uses contiguous memory and avoids transpose where possible.
        """
        # Ensure contiguous memory for MPS
        state = state.contiguous()
        propagation = propagation.contiguous()
        
        # MPS-optimized matmul (PyTorch handles this efficiently)
        return torch.matmul(state, propagation.T)
    
    def apply_bounce_transform(
        self,
        activations: torch.Tensor,
        bounce_angles: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimized bounce transformation for MPS.
        
        Fuses operations to minimize memory bandwidth.
        """
        activations = activations.contiguous()
        bounce_angles = bounce_angles.contiguous()
        
        # Compute angle factor with fused operations
        angle_magnitude = torch.abs(bounce_angles).sum(dim=1)
        # Use torch.cos which is MPS-accelerated
        angle_factor = torch.cos(angle_magnitude * (3.14159265 / 180.0 / 3.0))
        
        # Fused multiply-add
        return activations * (0.5 + 0.5 * angle_factor.unsqueeze(0))
    
    def fused_propagation_step(
        self,
        state_in: torch.Tensor,
        propagation: torch.Tensor,
        bounce_angles: torch.Tensor,
        decay: float,
        state_max: float
    ) -> torch.Tensor:
        """
        Fused propagation step optimized for MPS memory architecture.
        
        Minimizes memory allocations and uses in-place operations where safe.
        """
        # Ensure all tensors are contiguous and on MPS
        state_in = state_in.contiguous()
        propagation = propagation.contiguous()
        bounce_angles = bounce_angles.contiguous()
        
        # Propagate
        state = torch.matmul(state_in, propagation.T)
        
        # Fused bounce transform
        angle_magnitude = torch.abs(bounce_angles).sum(dim=1)
        angle_factor = torch.cos(angle_magnitude * (3.14159265 / 180.0 / 3.0))
        state = state * (0.5 + 0.5 * angle_factor.unsqueeze(0))
        
        # Fused decay and sigmoid
        state = state * decay
        state = torch.sigmoid(state * 2 - 1) * max(state_max, 0.1)
        
        return state


def create_kernels(device: Optional[torch.device] = None) -> PulsePropagationKernels:
    """
    Factory function to create appropriate kernel implementation.
    
    Args:
        device: Target device. Auto-detects if None.
        
    Returns:
        Appropriate PulsePropagationKernels implementation.
    """
    if device is None:
        if _cuda_available:
            device = torch.device('cuda')
        elif _mps_available:
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    if device.type == 'mps':
        return MPSPulsePropagationKernels()
    else:
        return PulsePropagationKernels(device)


# Convenience exports
__all__ = [
    'PulsePropagationKernels',
    'MPSPulsePropagationKernels', 
    'create_kernels',
    'get_backend'
]
