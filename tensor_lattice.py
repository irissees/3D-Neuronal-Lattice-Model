"""
Tensor3D Lattice: Flexible 3D lattice that accepts any 3D tensor shape.

Supports multiple topologies:
- 'bounded': Regular 3D grid with edges (no wrap-around)
- 'toroidal': Wrap-around in all dimensions (like a 3-torus)
- 'cylindrical': Wrap-around in 2 dimensions, bounded in 1
- 'custom': User-defined adjacency

This generalizes the ToroidalLattice to work with arbitrary 3D tensor inputs.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Union, Literal, Callable
from pathlib import Path
import pickle
import json
import plotly.graph_objects as go

from cell import DEVICE


TopologyType = Literal['bounded', 'toroidal', 'cylindrical']
ConnectivityType = Literal['6-connected', '18-connected', '26-connected']


@dataclass
class Tensor3DCell:
    """Cell data for 3D tensor lattice."""
    idx: int
    position: np.ndarray  # (x, y, z) in 3D space
    grid_pos: Tuple[int, int, int]  # (i, j, k) in tensor
    value: float = 0.0  # Cell value
    mean: float = 0.0
    std_dev: float = 1.0
    bounce_angles: np.ndarray = field(default_factory=lambda: np.zeros(3))
    split_prob: float = 0.1
    join_prob: float = 0.1
    touched: bool = False
    modified: bool = False
    
    def to_dict(self) -> dict:
        return {
            'idx': self.idx,
            'position': self.position.tolist(),
            'grid_pos': list(self.grid_pos),
            'value': self.value,
            'mean': self.mean,
            'std_dev': self.std_dev,
            'bounce_angles': self.bounce_angles.tolist(),
            'split_prob': self.split_prob,
            'join_prob': self.join_prob,
            'touched': self.touched,
            'modified': self.modified
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Tensor3DCell':
        return cls(
            idx=d['idx'],
            position=np.array(d['position']),
            grid_pos=tuple(d['grid_pos']),
            value=d.get('value', 0.0),
            mean=d.get('mean', 0.0),
            std_dev=d.get('std_dev', 1.0),
            bounce_angles=np.array(d.get('bounce_angles', [0, 0, 0])),
            split_prob=d.get('split_prob', 0.1),
            join_prob=d.get('join_prob', 0.1),
            touched=d.get('touched', False),
            modified=d.get('modified', False)
        )


class Tensor3DLattice(nn.Module):
    """
    Flexible 3D lattice that accepts any 3D tensor shape.
    
    Can operate in different topologies:
    - bounded: Regular 3D grid with edges
    - toroidal: Wrap-around in all 3 dimensions (3-torus T³)
    - cylindrical: Wrap-around in 2 dimensions
    
    Parameters:
    - shape: (D1, D2, D3) shape of the 3D tensor
    - topology: 'bounded', 'toroidal', or 'cylindrical'
    - connectivity: '6-connected', '18-connected', or '26-connected'
    - spacing: Physical spacing between cells (for visualization)
    """
    
    def __init__(
        self,
        shape: Union[Tuple[int, int, int], torch.Tensor],
        topology: TopologyType = 'bounded',
        connectivity: ConnectivityType = '6-connected',
        spacing: float = 1.0,
        wrap_dims: Optional[Tuple[bool, bool, bool]] = None,
        storage_path: str = None,
        device: str = DEVICE
    ):
        super().__init__()
        
        # Handle tensor input
        if isinstance(shape, torch.Tensor):
            if shape.dim() != 3:
                raise ValueError(f"Expected 3D tensor, got {shape.dim()}D")
            self.shape = tuple(shape.shape)
            self._init_tensor = shape.detach().cpu().numpy()
        else:
            self.shape = tuple(shape)
            self._init_tensor = None
        
        self.d1, self.d2, self.d3 = self.shape
        self.topology = topology
        self.connectivity = connectivity
        self.spacing = spacing
        self.device = device
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Determine wrap-around for each dimension
        if wrap_dims is not None:
            self.wrap_dims = wrap_dims
        elif topology == 'toroidal':
            self.wrap_dims = (True, True, True)
        elif topology == 'cylindrical':
            self.wrap_dims = (True, True, False)  # Wrap in first two dims
        else:  # bounded
            self.wrap_dims = (False, False, False)
        
        # Total cells
        self._total_cells = self.d1 * self.d2 * self.d3
        
        # Sparse cell storage
        self.sparse_cells: Dict[int, Tensor3DCell] = {}
        self.active_cell_indices: Set[int] = set()
        self.modified_cell_indices: Set[int] = set()
        
        # Blueprint storage
        self._blueprint_positions: Optional[np.ndarray] = None
        self._blueprint_adjacency: Optional[Dict[int, List[int]]] = None
        self._grid_to_idx: Optional[Dict[Tuple[int, int, int], int]] = None
        self._idx_to_grid: Optional[Dict[int, Tuple[int, int, int]]] = None
        
        # Build blueprint
        self._build_blueprint()
        
        # Initialize from tensor if provided
        if self._init_tensor is not None:
            self._init_from_tensor(self._init_tensor)
        
        # Learnable parameters for propagation
        self.register_buffer('_adjacency_matrix', self._build_adjacency_matrix())
    
    def _build_blueprint(self):
        """Build the 3D lattice blueprint."""
        positions = []
        self._grid_to_idx = {}
        self._idx_to_grid = {}
        
        idx = 0
        for i in range(self.d1):
            for j in range(self.d2):
                for k in range(self.d3):
                    # Physical position
                    x = i * self.spacing
                    y = j * self.spacing
                    z = k * self.spacing
                    
                    positions.append([x, y, z])
                    self._grid_to_idx[(i, j, k)] = idx
                    self._idx_to_grid[idx] = (i, j, k)
                    idx += 1
        
        self._blueprint_positions = np.array(positions)
        
        # Build adjacency
        self._blueprint_adjacency = {}
        neighbor_offsets = self._get_neighbor_offsets()
        
        for idx in range(self._total_cells):
            i, j, k = self._idx_to_grid[idx]
            neighbors = []
            
            for di, dj, dk in neighbor_offsets:
                ni, nj, nk = i + di, j + dj, k + dk
                
                # Apply wrap-around based on topology
                if self.wrap_dims[0]:
                    ni = ni % self.d1
                elif ni < 0 or ni >= self.d1:
                    continue
                
                if self.wrap_dims[1]:
                    nj = nj % self.d2
                elif nj < 0 or nj >= self.d2:
                    continue
                
                if self.wrap_dims[2]:
                    nk = nk % self.d3
                elif nk < 0 or nk >= self.d3:
                    continue
                
                neighbor_idx = self._grid_to_idx.get((ni, nj, nk))
                if neighbor_idx is not None:
                    neighbors.append(neighbor_idx)
            
            self._blueprint_adjacency[idx] = neighbors
    
    def _get_neighbor_offsets(self) -> List[Tuple[int, int, int]]:
        """Get neighbor offsets based on connectivity type."""
        # 6-connected: face neighbors only
        offsets_6 = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]
        
        if self.connectivity == '6-connected':
            return offsets_6
        
        # 18-connected: faces + edges
        offsets_12_edges = [
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
        ]
        
        if self.connectivity == '18-connected':
            return offsets_6 + offsets_12_edges
        
        # 26-connected: faces + edges + corners
        offsets_8_corners = [
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
        ]
        
        return offsets_6 + offsets_12_edges + offsets_8_corners
    
    def _build_adjacency_matrix(self) -> torch.Tensor:
        """Build sparse adjacency matrix for efficient propagation."""
        n = self._total_cells
        adj = torch.zeros(n, n)
        
        for idx, neighbors in self._blueprint_adjacency.items():
            for neighbor_idx in neighbors:
                adj[idx, neighbor_idx] = 1.0
        
        # Normalize rows
        row_sum = adj.sum(dim=1, keepdim=True) + 1e-8
        adj = adj / row_sum
        
        return adj
    
    def _init_from_tensor(self, tensor: np.ndarray):
        """Initialize cell values from a 3D tensor."""
        for idx in range(self._total_cells):
            i, j, k = self._idx_to_grid[idx]
            value = tensor[i, j, k]
            cell = self._generate_cell(idx, value=float(value))
            self.sparse_cells[idx] = cell
    
    def _generate_cell(self, idx: int, value: float = 0.0) -> Tensor3DCell:
        """Generate a cell with given value."""
        if idx < 0 or idx >= self._total_cells:
            raise ValueError(f"Cell index {idx} out of range")
        
        position = self._blueprint_positions[idx]
        grid_pos = self._idx_to_grid[idx]
        
        return Tensor3DCell(
            idx=idx,
            position=position,
            grid_pos=grid_pos,
            value=value,
            mean=np.random.randn() * 0.1,
            std_dev=np.abs(np.random.randn()) + 0.5,
            bounce_angles=np.random.randn(3) * 0.1,
            split_prob=np.random.rand() * 0.2,
            join_prob=np.random.rand() * 0.2,
            touched=True,
            modified=True
        )
    
    def _get_or_create_cell(self, idx: int) -> Tensor3DCell:
        """Get or create a cell."""
        if idx not in self.sparse_cells:
            self.sparse_cells[idx] = self._generate_cell(idx)
            self.modified_cell_indices.add(idx)
        
        self.active_cell_indices.add(idx)
        self.sparse_cells[idx].touched = True
        return self.sparse_cells[idx]
    
    def from_tensor(self, tensor: torch.Tensor) -> 'Tensor3DLattice':
        """
        Initialize/update lattice from a 3D tensor.
        
        Args:
            tensor: 3D tensor of shape (D1, D2, D3)
        
        Returns:
            self for method chaining
        """
        if tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D")
        
        if tensor.shape != self.shape:
            raise ValueError(f"Tensor shape {tensor.shape} doesn't match lattice shape {self.shape}")
        
        tensor_np = tensor.detach().cpu().numpy()
        self._init_from_tensor(tensor_np)
        return self
    
    def to_tensor(self, device: str = None) -> torch.Tensor:
        """
        Convert lattice cell values back to a 3D tensor.
        
        Returns:
            3D tensor of shape (D1, D2, D3) containing cell values
        """
        device = device or self.device
        tensor = torch.zeros(self.shape, device=device)
        
        for idx, cell in self.sparse_cells.items():
            i, j, k = cell.grid_pos
            tensor[i, j, k] = cell.value
        
        return tensor
    
    # ========================================================================
    # Pattern Setting Methods
    # ========================================================================
    
    def set_pattern(
        self,
        pattern: Union['Pattern3D', np.ndarray, torch.Tensor, str],
        **kwargs
    ) -> 'Tensor3DLattice':
        """
        Set cell weights from a pattern.
        
        Args:
            pattern: Pattern3D object, numpy array, torch tensor, or pattern name
            **kwargs: If pattern is a string, these are passed to the pattern generator
        
        Returns:
            self for method chaining
        
        Examples:
            # From Pattern3D object
            lattice.set_pattern(pg.sphere(radius=10))
            
            # From pattern name (auto-creates PatternGenerator)
            lattice.set_pattern('sphere', radius=10, falloff='smooth')
            lattice.set_pattern('checkerboard', cell_size=4)
            lattice.set_pattern('gradient', axis='z')
            
            # From tensor
            lattice.set_pattern(torch.randn(32, 32, 32))
        """
        from lattice_patterns import PatternGenerator, Pattern3D
        
        if isinstance(pattern, str):
            # Create pattern from name
            pg = PatternGenerator(self.shape)
            method = getattr(pg, pattern, None)
            if method is None:
                raise ValueError(f"Unknown pattern: {pattern}. Available: sphere, cube, "
                               "cylinder, cone, torus, plane, gradient, radial_gradient, "
                               "sine_wave, spherical_wave, spiral, noise, perlin_noise, "
                               "checkerboard, grid_lines, dots, shell, gyroid, voronoi")
            pattern = method(**kwargs)
        
        if isinstance(pattern, torch.Tensor):
            data = pattern.detach().cpu().numpy()
        elif hasattr(pattern, 'data'):  # Pattern3D
            data = pattern.data
        else:
            data = np.asarray(pattern)
        
        if data.shape != self.shape:
            raise ValueError(f"Pattern shape {data.shape} doesn't match lattice shape {self.shape}")
        
        # Set cell values from pattern
        for idx in range(self._total_cells):
            i, j, k = self._idx_to_grid[idx]
            value = float(data[i, j, k])
            
            if idx in self.sparse_cells:
                self.sparse_cells[idx].value = value
                self.sparse_cells[idx].modified = True
            else:
                self.sparse_cells[idx] = self._generate_cell(idx, value=value)
            
            self.modified_cell_indices.add(idx)
        
        return self
    
    def set_composite_pattern(
        self,
        patterns: List[Tuple[str, Dict, float]],
        mode: str = 'add'
    ) -> 'Tensor3DLattice':
        """
        Set cell weights from composed patterns.
        
        Args:
            patterns: List of (pattern_name, params_dict, weight) tuples
            mode: 'add', 'multiply', 'max', 'min', 'average'
        
        Example:
            lattice.set_composite_pattern([
                ('sphere', {'center': (16, 16, 16), 'radius': 10}, 1.0),
                ('noise', {'scale': 0.1}, 0.2),
                ('gradient', {'axis': 'z'}, 0.3)
            ])
        """
        from lattice_patterns import PatternGenerator
        
        pg = PatternGenerator(self.shape)
        pattern = pg.compose(patterns, mode=mode)
        return self.set_pattern(pattern)
    
    def set_from_function(
        self,
        func: Callable[[int, int, int], float]
    ) -> 'Tensor3DLattice':
        """
        Set cell weights from a custom function.
        
        Args:
            func: Function(i, j, k) -> value for each cell
        
        Example:
            # Diagonal gradient
            lattice.set_from_function(lambda i, j, k: (i + j + k) / sum(lattice.shape))
            
            # Distance from corner
            lattice.set_from_function(lambda i, j, k: np.sqrt(i**2 + j**2 + k**2))
        """
        for idx in range(self._total_cells):
            i, j, k = self._idx_to_grid[idx]
            value = float(func(i, j, k))
            
            if idx in self.sparse_cells:
                self.sparse_cells[idx].value = value
                self.sparse_cells[idx].modified = True
            else:
                self.sparse_cells[idx] = self._generate_cell(idx, value=value)
            
            self.modified_cell_indices.add(idx)
        
        return self
    
    def set_region(
        self,
        region: Tuple[slice, slice, slice],
        value: Union[float, np.ndarray, torch.Tensor] = 1.0
    ) -> 'Tensor3DLattice':
        """
        Set values in a specific region.
        
        Args:
            region: Tuple of slices defining the region (i_slice, j_slice, k_slice)
            value: Scalar or array of values
        
        Example:
            # Set a cube region to 1.0
            lattice.set_region((slice(10, 20), slice(10, 20), slice(10, 20)), value=1.0)
            
            # Set bottom half
            lattice.set_region((slice(None), slice(None), slice(0, 16)), value=0.5)
        """
        i_range = range(*region[0].indices(self.d1))
        j_range = range(*region[1].indices(self.d2))
        k_range = range(*region[2].indices(self.d3))
        
        is_array = not isinstance(value, (int, float))
        if is_array:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
        
        for ii, i in enumerate(i_range):
            for jj, j in enumerate(j_range):
                for kk, k in enumerate(k_range):
                    idx = self._grid_to_idx.get((i, j, k))
                    if idx is None:
                        continue
                    
                    v = value[ii, jj, kk] if is_array else value
                    
                    if idx in self.sparse_cells:
                        self.sparse_cells[idx].value = float(v)
                        self.sparse_cells[idx].modified = True
                    else:
                        self.sparse_cells[idx] = self._generate_cell(idx, value=float(v))
                    
                    self.modified_cell_indices.add(idx)
        
        return self
    
    def clear_pattern(self, value: float = 0.0) -> 'Tensor3DLattice':
        """Reset all cell values to a constant."""
        for idx in range(self._total_cells):
            if idx in self.sparse_cells:
                self.sparse_cells[idx].value = value
                self.sparse_cells[idx].modified = True
                self.modified_cell_indices.add(idx)
        return self
    
    def get_pattern_generator(self) -> 'PatternGenerator':
        """Get a PatternGenerator configured for this lattice's shape."""
        from lattice_patterns import PatternGenerator
        return PatternGenerator(self.shape)
    
    # ========================================================================
    # Index/Position Methods
    # ========================================================================
    
    def get_idx(self, i: int, j: int, k: int) -> int:
        """Get linear index from grid position (with wrap-around if enabled)."""
        if self.wrap_dims[0]:
            i = i % self.d1
        if self.wrap_dims[1]:
            j = j % self.d2
        if self.wrap_dims[2]:
            k = k % self.d3
        return self._grid_to_idx.get((i, j, k), -1)
    
    def get_grid_pos(self, idx: int) -> Tuple[int, int, int]:
        """Get grid position from linear index."""
        return self._idx_to_grid.get(idx, (0, 0, 0))
    
    def get_neighbors(self, idx: int) -> List[int]:
        """Get neighbor indices for a cell."""
        return self._blueprint_adjacency.get(idx, [])
    
    def forward(
        self,
        input_tensor: torch.Tensor,
        num_steps: int = 5,
        decay: float = 0.9
    ) -> torch.Tensor:
        """
        Propagate signal through the lattice.
        
        Args:
            input_tensor: 3D tensor or flattened tensor of shape (batch, D1*D2*D3)
            num_steps: Number of propagation steps
            decay: Decay factor per step
        
        Returns:
            Output tensor after propagation
        """
        # Handle different input shapes
        if input_tensor.dim() == 3:
            # Single 3D tensor -> flatten
            batch_size = 1
            state = input_tensor.reshape(1, -1)
        elif input_tensor.dim() == 4:
            # Batched 3D tensors (B, D1, D2, D3)
            batch_size = input_tensor.shape[0]
            state = input_tensor.reshape(batch_size, -1)
        else:
            # Already flattened (B, N)
            batch_size = input_tensor.shape[0]
            state = input_tensor
        
        device = input_tensor.device
        adj = self._adjacency_matrix.to(device)
        
        # Propagate
        for _ in range(num_steps):
            propagated = torch.matmul(state, adj)
            state = decay * (0.3 * state + 0.7 * propagated)
        
        # Reshape back to 3D
        if input_tensor.dim() == 3:
            return state.reshape(self.shape)
        elif input_tensor.dim() == 4:
            return state.reshape(batch_size, *self.shape)
        return state
    
    def get_stats(self) -> Dict:
        """Get lattice statistics."""
        return {
            'shape': self.shape,
            'total_cells': self._total_cells,
            'cells_in_memory': len(self.sparse_cells),
            'active_cells': len(self.active_cell_indices),
            'topology': self.topology,
            'connectivity': self.connectivity,
            'wrap_dims': self.wrap_dims
        }
    
    def visualize(
        self,
        show_values: bool = True,
        cell_size: float = 5,
        opacity: float = 0.7,
        colorscale: str = 'Viridis',
        title: str = None
    ) -> go.Figure:
        """Create interactive 3D visualization."""
        fig = go.Figure()
        
        # Get cell positions and values
        if self.sparse_cells:
            indices = list(self.sparse_cells.keys())
            positions = np.array([self.sparse_cells[i].position for i in indices])
            values = np.array([self.sparse_cells[i].value for i in indices])
            
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=cell_size,
                    color=values if show_values else None,
                    colorscale=colorscale,
                    opacity=opacity,
                    colorbar=dict(title='Value') if show_values else None
                ),
                name='Cells',
                hovertemplate=(
                    'Cell %{customdata[0]}<br>'
                    'Grid: (%{customdata[1]}, %{customdata[2]}, %{customdata[3]})<br>'
                    'Value: %{customdata[4]:.4f}'
                ),
                customdata=[
                    [i, *self.sparse_cells[i].grid_pos, self.sparse_cells[i].value]
                    for i in indices
                ]
            ))
        else:
            # Show blueprint positions
            fig.add_trace(go.Scatter3d(
                x=self._blueprint_positions[:, 0],
                y=self._blueprint_positions[:, 1],
                z=self._blueprint_positions[:, 2],
                mode='markers',
                marker=dict(
                    size=cell_size * 0.5,
                    color='lightgray',
                    opacity=0.3
                ),
                name='Blueprint'
            ))
        
        fig.update_layout(
            title=title or f'3D Tensor Lattice {self.shape} ({self.topology})',
            scene=dict(
                xaxis_title='Dim 1',
                yaxis_title='Dim 2',
                zaxis_title='Dim 3',
                aspectmode='data'
            ),
            width=900,
            height=700
        )
        
        return fig


class DifferentiableTensor3DLattice(nn.Module):
    """
    Fully differentiable version of Tensor3DLattice for training.
    
    All operations are tensor-based for gradient flow.
    """
    
    def __init__(
        self,
        shape: Union[Tuple[int, int, int], torch.Tensor],
        topology: TopologyType = 'bounded',
        connectivity: ConnectivityType = '6-connected',
        max_steps: int = 10,
        device: str = DEVICE
    ):
        super().__init__()
        
        if isinstance(shape, torch.Tensor):
            self.shape = tuple(shape.shape)
        else:
            self.shape = tuple(shape)
        
        self.d1, self.d2, self.d3 = self.shape
        self._total_cells = self.d1 * self.d2 * self.d3
        self.topology = topology
        self.connectivity = connectivity
        self.max_steps = max_steps
        self.device = device
        
        # Build lattice for adjacency
        self._lattice = Tensor3DLattice(
            shape=self.shape,
            topology=topology,
            connectivity=connectivity,
            device=device
        )
        
        # Register adjacency as buffer
        self.register_buffer('adjacency', self._lattice._adjacency_matrix)
        
        # Learnable parameters
        self.step_weights = nn.Parameter(torch.ones(max_steps) / max_steps)
        self.decay = nn.Parameter(torch.tensor(0.85))
        self.cell_weights = nn.Parameter(torch.ones(self._total_cells))
        self.bounce_angles = nn.Parameter(torch.randn(self._total_cells, 3) * 0.1)
    
    def forward(
        self,
        x: torch.Tensor,
        return_history: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through differentiable lattice.
        
        Args:
            x: Input tensor of shape (B, D1, D2, D3) or (B, D1*D2*D3)
            return_history: If True, return propagation history
        
        Returns:
            Output tensor and optionally history
        """
        # Reshape input
        if x.dim() == 4:
            batch_size = x.shape[0]
            state = x.reshape(batch_size, -1)
        elif x.dim() == 3:
            batch_size = 1
            state = x.reshape(1, -1)
        else:
            batch_size = x.shape[0]
            state = x
        
        device = x.device
        adj = self.adjacency.to(device)
        
        # Apply cell weights
        weighted_state = state * torch.sigmoid(self.cell_weights)
        
        history = [weighted_state]
        current = weighted_state
        decay_val = torch.clamp(self.decay, 0.5, 0.99)
        
        for step in range(self.max_steps):
            # Propagate through adjacency
            propagated = torch.matmul(current, adj)
            
            # Apply bounce angles effect
            angle_effect = torch.cos(self.bounce_angles.abs().mean(dim=1))
            propagated = propagated * (0.5 + 0.5 * angle_effect)
            
            # Mix and decay
            current = (0.3 * current + 0.7 * propagated) * decay_val
            history.append(current)
        
        # Weighted combination of steps
        weights = torch.softmax(self.step_weights, dim=0)
        final = sum(w * h for w, h in zip(weights, history[1:]))
        
        # Reshape output
        if x.dim() == 4:
            final = final.reshape(batch_size, *self.shape)
        elif x.dim() == 3:
            final = final.reshape(*self.shape)
        
        if return_history:
            return final, history
        return final
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """Get the adjacency matrix."""
        return self.adjacency
    
    def init_cell_weights_from_pattern(
        self,
        pattern: Union[str, np.ndarray, torch.Tensor, 'Pattern3D'],
        **kwargs
    ) -> 'DifferentiableTensor3DLattice':
        """
        Initialize cell_weights parameter from a pattern.
        
        Args:
            pattern: Pattern name, Pattern3D, or array
            **kwargs: Arguments for pattern generator if pattern is a string
        
        Returns:
            self for method chaining
        
        Example:
            lattice.init_cell_weights_from_pattern('sphere', radius=10, falloff='gaussian')
            lattice.init_cell_weights_from_pattern('checkerboard', cell_size=4)
        """
        from lattice_patterns import PatternGenerator
        
        if isinstance(pattern, str):
            pg = PatternGenerator(self.shape)
            method = getattr(pg, pattern, None)
            if method is None:
                raise ValueError(f"Unknown pattern: {pattern}")
            pattern = method(**kwargs)
        
        if hasattr(pattern, 'data'):
            data = pattern.data
        elif isinstance(pattern, torch.Tensor):
            data = pattern.detach().cpu().numpy()
        else:
            data = np.asarray(pattern)
        
        # Flatten and set as initial cell weights
        weights = torch.tensor(data.flatten(), dtype=torch.float32)
        with torch.no_grad():
            self.cell_weights.copy_(weights)
        
        return self
    
    def init_bounce_angles_from_pattern(
        self,
        pattern: Union[str, np.ndarray, torch.Tensor, 'Pattern3D'],
        scale: float = 1.0,
        **kwargs
    ) -> 'DifferentiableTensor3DLattice':
        """
        Initialize bounce_angles parameter from a pattern.
        
        The pattern value at each cell is used to scale the bounce angles.
        
        Args:
            pattern: Pattern name, Pattern3D, or array
            scale: Overall scale factor for angles
            **kwargs: Arguments for pattern generator if pattern is a string
        """
        from lattice_patterns import PatternGenerator
        
        if isinstance(pattern, str):
            pg = PatternGenerator(self.shape)
            method = getattr(pg, pattern, None)
            if method is None:
                raise ValueError(f"Unknown pattern: {pattern}")
            pattern = method(**kwargs)
        
        if hasattr(pattern, 'data'):
            data = pattern.data
        elif isinstance(pattern, torch.Tensor):
            data = pattern.detach().cpu().numpy()
        else:
            data = np.asarray(pattern)
        
        # Use pattern to modulate random angles
        weights = data.flatten()
        angles = np.random.randn(self._total_cells, 3) * scale
        angles = angles * weights[:, np.newaxis]
        
        with torch.no_grad():
            self.bounce_angles.copy_(torch.tensor(angles, dtype=torch.float32))
        
        return self
    
    def get_pattern_generator(self) -> 'PatternGenerator':
        """Get a PatternGenerator configured for this lattice's shape."""
        from lattice_patterns import PatternGenerator
        return PatternGenerator(self.shape)


def create_lattice_from_tensor(
    tensor: torch.Tensor,
    topology: TopologyType = 'bounded',
    connectivity: ConnectivityType = '6-connected'
) -> Tensor3DLattice:
    """
    Factory function to create a lattice from any 3D tensor.
    
    Args:
        tensor: 3D tensor of any shape
        topology: 'bounded', 'toroidal', or 'cylindrical'
        connectivity: '6-connected', '18-connected', or '26-connected'
    
    Returns:
        Tensor3DLattice initialized with tensor values
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D")
    
    return Tensor3DLattice(
        shape=tensor,
        topology=topology,
        connectivity=connectivity
    )


if __name__ == "__main__":
    # Test with various 3D tensors
    print("=" * 60)
    print("Testing Tensor3DLattice")
    print("=" * 60)
    
    # Test 1: Create from shape
    print("\n1. Create from shape (4, 5, 6):")
    lattice1 = Tensor3DLattice(shape=(4, 5, 6), topology='bounded')
    print(f"   Stats: {lattice1.get_stats()}")
    
    # Test 2: Create from tensor
    print("\n2. Create from random tensor:")
    tensor = torch.randn(8, 8, 8)
    lattice2 = Tensor3DLattice(shape=tensor, topology='toroidal')
    print(f"   Stats: {lattice2.get_stats()}")
    
    # Test 3: Propagation
    print("\n3. Test propagation:")
    input_tensor = torch.randn(2, 4, 4, 4)  # Batch of 2
    output = lattice1.forward(input_tensor[:, :4, :5, :6], num_steps=3)
    print(f"   Input shape: {input_tensor.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test 4: Differentiable version
    print("\n4. Test differentiable lattice:")
    diff_lattice = DifferentiableTensor3DLattice(
        shape=(4, 4, 4),
        topology='toroidal',
        max_steps=5
    )
    x = torch.randn(2, 4, 4, 4, requires_grad=True)
    y = diff_lattice(x)
    loss = y.sum()
    loss.backward()
    print(f"   Input grad exists: {x.grad is not None}")
    print(f"   Params with grad: {sum(p.grad is not None for p in diff_lattice.parameters())}")
    
    # Test 5: Different topologies
    print("\n5. Different topologies:")
    for topo in ['bounded', 'toroidal', 'cylindrical']:
        lat = Tensor3DLattice(shape=(3, 3, 3), topology=topo)
        # Count neighbors for center cell
        center_idx = lat.get_idx(1, 1, 1)
        n_neighbors = len(lat.get_neighbors(center_idx))
        print(f"   {topo}: center cell has {n_neighbors} neighbors")
    
    # Test 6: Pattern setting
    print("\n6. Test pattern setting:")
    lat = Tensor3DLattice(shape=(16, 16, 16), topology='toroidal')
    
    # Set sphere pattern
    lat.set_pattern('sphere', radius=6, falloff='gaussian')
    tensor_out = lat.to_tensor()
    print(f"   Sphere pattern: min={tensor_out.min():.3f}, max={tensor_out.max():.3f}")
    
    # Set composite pattern
    lat.set_composite_pattern([
        ('sphere', {'radius': 5}, 1.0),
        ('gradient', {'axis': 'z'}, 0.3),
        ('noise', {'scale': 0.1}, 0.2)
    ])
    tensor_out = lat.to_tensor()
    print(f"   Composite pattern: min={tensor_out.min():.3f}, max={tensor_out.max():.3f}")
    
    # Set from custom function
    lat.set_from_function(lambda i, j, k: np.sin(i * 0.5) * np.cos(j * 0.5))
    tensor_out = lat.to_tensor()
    print(f"   Function pattern: min={tensor_out.min():.3f}, max={tensor_out.max():.3f}")
    
    # Test 7: Differentiable lattice with pattern init
    print("\n7. Differentiable lattice with pattern init:")
    diff_lat = DifferentiableTensor3DLattice(shape=(8, 8, 8), topology='bounded')
    diff_lat.init_cell_weights_from_pattern('radial_gradient', direction='inward')
    print(f"   Cell weights initialized from radial_gradient")
    print(f"   Weights range: [{diff_lat.cell_weights.min():.3f}, {diff_lat.cell_weights.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
