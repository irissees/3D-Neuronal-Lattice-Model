"""
Sparse Lattice implementation with on-demand cell generation and persistent storage.

Features:
- Cells are generated lazily during first propagation
- Only touched cells participate in kernel computations
- Full lattice state persisted to disk
- Changed cells tracked and saved incrementally
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict, Union, Set
from dataclasses import dataclass, field
import os
import json
import pickle
from pathlib import Path

from cell import Cell, DEVICE
from lattice_visualizer import LatticeVisualizer, PulseRecord, OriginPoint


@dataclass
class SparseCell:
    """Lightweight cell data for sparse storage."""
    idx: int
    position: np.ndarray
    mean: float
    std_dev: float
    bounce_angles: np.ndarray  # 3 values
    split_prob: float
    join_prob: float
    alternate_mode: str
    touched: bool = False
    modified: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'idx': self.idx,
            'position': self.position.tolist(),
            'mean': self.mean,
            'std_dev': self.std_dev,
            'bounce_angles': self.bounce_angles.tolist(),
            'split_prob': self.split_prob,
            'join_prob': self.join_prob,
            'alternate_mode': self.alternate_mode
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SparseCell':
        """Create from dictionary."""
        return cls(
            idx=data['idx'],
            position=np.array(data['position']),
            mean=data['mean'],
            std_dev=data['std_dev'],
            bounce_angles=np.array(data['bounce_angles']),
            split_prob=data['split_prob'],
            join_prob=data['join_prob'],
            alternate_mode=data['alternate_mode']
        )


class SparseLatticeVisualizer(LatticeVisualizer):
    """
    Sparse lattice that generates cells on-demand during propagation.
    
    Only cells touched by propagation are loaded into GPU memory and
    participate in computations. Full lattice blueprint is stored on disk.
    
    Architecture:
    1. Blueprint: Full lattice structure stored on disk (positions, adjacency)
    2. Active Set: Only cells touched during current propagation
    3. Modified Set: Cells whose parameters changed (for incremental save)
    """
    
    def __init__(
        self,
        layers: int = 3,
        hex_radius: int = 3,
        layer_spacing: float = 1.0,
        hex_spacing: float = 0.9,
        device: Union[str, torch.device] = None,
        storage_path: str = "./sparse_lattice_storage",
        lazy_init: bool = True
    ):
        """
        Initialize sparse lattice.
        
        Args:
            layers: Number of vertical layers
            hex_radius: Radius of hexagonal grid per layer
            layer_spacing: Vertical distance between layers
            hex_spacing: Horizontal distance between cells
            device: Torch device
            storage_path: Directory for persistent storage
            lazy_init: If True, don't generate cells until first propagation
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.lazy_init = lazy_init
        self._blueprint_built = False
        
        # Track active and modified cells
        self.active_cell_indices: Set[int] = set()
        self.modified_cell_indices: Set[int] = set()
        
        # Sparse cell storage (only loaded cells)
        self.sparse_cells: Dict[int, SparseCell] = {}
        
        # Blueprint data (loaded from disk or computed)
        self._blueprint_positions: Optional[np.ndarray] = None
        self._blueprint_adjacency: Optional[Dict[int, List[int]]] = None
        self._total_cells: int = 0
        
        # Initialize parent (but skip full cell generation if lazy)
        if lazy_init:
            # Store config without building
            self.layers = layers
            self.hex_radius = hex_radius
            self.layer_spacing = layer_spacing
            self.hex_spacing = hex_spacing
            
            if device is None:
                self.device = DEVICE
            elif isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device
            
            print(f"SparseLatticeVisualizer initialized (lazy mode)")
            print(f"  Storage: {self.storage_path}")
            
            # Initialize empty structures
            self.cells = []
            self.positions = None
            self.positions_np = None
            self.adjacency = {}
            self.pulse_history = []
            self.current_time = 0
            self.active_cells = set()
            self.origin_points = []
            
            # Try to load blueprint from storage
            self._load_or_build_blueprint()
        else:
            # Full initialization
            super().__init__(layers, hex_radius, layer_spacing, hex_spacing, device)
            self._blueprint_built = True
            self._save_blueprint()
    
    def _get_blueprint_path(self) -> Path:
        """Get path to blueprint file."""
        return self.storage_path / f"blueprint_L{self.layers}_R{self.hex_radius}.json"
    
    def _get_cells_path(self) -> Path:
        """Get path to cells storage file."""
        return self.storage_path / f"cells_L{self.layers}_R{self.hex_radius}.pkl"
    
    def _load_or_build_blueprint(self):
        """Load blueprint from disk or build if not exists."""
        blueprint_path = self._get_blueprint_path()
        
        if blueprint_path.exists():
            print(f"  Loading blueprint from {blueprint_path}")
            with open(blueprint_path, 'r') as f:
                data = json.load(f)
            
            self._blueprint_positions = np.array(data['positions'])
            self._blueprint_adjacency = {int(k): v for k, v in data['adjacency'].items()}
            self._total_cells = data['total_cells']
            self._blueprint_built = True
            
            # Set up minimal structures for compatibility
            self.positions_np = self._blueprint_positions
            self.positions = torch.tensor(self._blueprint_positions, device=self.device, dtype=torch.float32)
            self.adjacency = self._blueprint_adjacency
            
            print(f"  Blueprint loaded: {self._total_cells} cell positions")
        else:
            print(f"  Building new blueprint...")
            self._build_blueprint()
            self._save_blueprint()
    
    def _build_blueprint(self):
        """Build the lattice blueprint (positions and adjacency only)."""
        positions = []
        
        for layer in range(self.layers):
            for q in range(-self.hex_radius, self.hex_radius + 1):
                r1 = max(-self.hex_radius, -q - self.hex_radius)
                r2 = min(self.hex_radius, -q + self.hex_radius)
                for r in range(r1, r2 + 1):
                    x = self.hex_spacing * (3/2 * q)
                    y = self.hex_spacing * (np.sqrt(3) * (r + q/2))
                    z = layer * self.layer_spacing
                    positions.append([x, y, z])
        
        self._blueprint_positions = np.array(positions, dtype=np.float32)
        self._total_cells = len(positions)
        
        # Build adjacency
        self._blueprint_adjacency = {}
        hex_adjacent_dist = self.hex_spacing * np.sqrt(3) * 1.1
        
        for i in range(self._total_cells):
            neighbors = []
            for j in range(self._total_cells):
                if i == j:
                    continue
                dist = np.linalg.norm(self._blueprint_positions[i] - self._blueprint_positions[j])
                # Horizontal neighbors
                if dist < hex_adjacent_dist:
                    neighbors.append(j)
                # Vertical neighbors
                elif abs(dist - self.layer_spacing) < 0.2:
                    neighbors.append(j)
            self._blueprint_adjacency[i] = neighbors
        
        # Set up structures
        self.positions_np = self._blueprint_positions
        self.positions = torch.tensor(self._blueprint_positions, device=self.device, dtype=torch.float32)
        self.adjacency = self._blueprint_adjacency
        self._blueprint_built = True
        
        print(f"  Blueprint built: {self._total_cells} cells")
    
    def _save_blueprint(self):
        """Save blueprint to disk."""
        blueprint_path = self._get_blueprint_path()
        
        data = {
            'layers': self.layers,
            'hex_radius': self.hex_radius,
            'layer_spacing': self.layer_spacing,
            'hex_spacing': self.hex_spacing,
            'total_cells': self._total_cells,
            'positions': self._blueprint_positions.tolist(),
            'adjacency': {str(k): v for k, v in self._blueprint_adjacency.items()}
        }
        
        with open(blueprint_path, 'w') as f:
            json.dump(data, f)
        
        print(f"  Blueprint saved to {blueprint_path}")
    
    def _generate_cell(self, idx: int) -> SparseCell:
        """Generate a single cell on-demand."""
        position = self._blueprint_positions[idx]
        alternate_mode = 'closer' if idx % 2 == 0 else 'further'
        
        # Random initialization matching LatticeVisualizer
        if alternate_mode == 'closer':
            mean = (np.random.rand() * 2 - 1)
            std_dev = np.random.rand() * 0.5 + 2.0  # Range: 2.0 to 2.5
            split_prob = np.random.rand() * 0.2
            join_prob = np.random.rand() * 0.3 + 0.2
            bounce_angles = np.random.rand(3) * 0.5
        else:
            mean = (np.random.rand() * 2 - 1)
            std_dev = np.random.rand() * 1.0 + 2.5  # Range: 2.5 to 3.5
            split_prob = np.random.rand() * 0.3 + 0.2
            join_prob = np.random.rand() * 0.2
            bounce_angles = np.random.rand(3) * 1.0 + 1.0
        
        return SparseCell(
            idx=idx,
            position=position,
            mean=mean,
            std_dev=max(2.0, std_dev),
            bounce_angles=bounce_angles,
            split_prob=min(1.0, max(0.0, split_prob)),
            join_prob=min(1.0, max(0.0, join_prob)),
            alternate_mode=alternate_mode,
            touched=True,
            modified=True
        )
    
    def _get_or_create_cell(self, idx: int) -> SparseCell:
        """Get cell from sparse storage or create it."""
        if idx not in self.sparse_cells:
            self.sparse_cells[idx] = self._generate_cell(idx)
            self.modified_cell_indices.add(idx)
        
        self.active_cell_indices.add(idx)
        self.sparse_cells[idx].touched = True
        return self.sparse_cells[idx]
    
    def _get_neighbor_indices(self, idx: int) -> List[int]:
        """Get neighbor indices for a cell from the blueprint adjacency."""
        if self._blueprint_adjacency is None:
            return []
        return self._blueprint_adjacency.get(idx, [])
    
    def _sparse_cell_to_full(self, sparse: SparseCell) -> Cell:
        """Convert SparseCell to full Cell for computation."""
        return Cell(
            mean=sparse.mean,
            std_dev=sparse.std_dev,
            bounce_angles=sparse.bounce_angles,
            split_prob=sparse.split_prob,
            join_prob=sparse.join_prob,
            state=sparse.idx,
            device=self.device,
            alternate_mode=sparse.alternate_mode
        )
    
    def _update_sparse_from_full(self, sparse: SparseCell, full: Cell):
        """Update SparseCell from full Cell after computation."""
        new_mean = full.mean.item()
        new_std_dev = full.std_dev.item()
        new_bounce = full.bounce_angles.detach().cpu().numpy()
        new_split = full.split_prob.item()
        new_join = full.join_prob.item()
        
        # Check if modified
        if (abs(new_mean - sparse.mean) > 1e-6 or
            abs(new_std_dev - sparse.std_dev) > 1e-6 or
            np.any(np.abs(new_bounce - sparse.bounce_angles) > 1e-6) or
            abs(new_split - sparse.split_prob) > 1e-6 or
            abs(new_join - sparse.join_prob) > 1e-6):
            sparse.mean = new_mean
            sparse.std_dev = new_std_dev
            sparse.bounce_angles = new_bounce
            sparse.split_prob = new_split
            sparse.join_prob = new_join
            sparse.modified = True
            self.modified_cell_indices.add(sparse.idx)
    
    def propagate_pulse_sparse(
        self,
        start_cell: int,
        initial_direction: Union[np.ndarray, torch.Tensor],
        initial_power: float = 1.0,
        max_steps: int = 10,
        enable_split_join: bool = True
    ) -> List[PulseRecord]:
        """
        Propagate pulse through lattice, generating cells on-demand.
        
        Only cells touched during propagation are loaded and computed.
        
        Args:
            start_cell: Index of starting cell
            initial_direction: Initial pulse direction
            initial_power: Initial pulse power
            max_steps: Maximum propagation steps
            enable_split_join: Whether to enable split/join behavior
            
        Returns:
            List of PulseRecord objects tracking the pulse path
        """
        if not self._blueprint_built:
            self._load_or_build_blueprint()
        
        pulse_path = []
        
        # Convert direction
        if isinstance(initial_direction, np.ndarray):
            direction = torch.tensor(initial_direction, device=self.device, dtype=torch.float32)
        else:
            direction = initial_direction.to(self.device)
        direction = direction / torch.norm(direction)
        
        # Active pulses: (cell_idx, direction, power, split_weight, split_occurred)
        active_pulses = [(start_cell, direction.clone(), initial_power, 1.0, False)]
        
        for step in range(max_steps):
            if not active_pulses:
                break
            
            new_pulses = []
            cell_incoming: Dict[int, List[Tuple[torch.Tensor, float]]] = {}
            
            for current_cell, pulse_dir, power, split_weight, split_occurred in active_pulses:
                if power <= 0.01:
                    continue
                
                # Get or create cell (lazy generation)
                sparse_cell = self._get_or_create_cell(current_cell)
                full_cell = self._sparse_cell_to_full(sparse_cell)
                
                pos = torch.tensor(sparse_cell.position, device=self.device, dtype=torch.float32)
                bounce_angles = full_cell.get_bounce_angles()
                
                # Sample for bounce decision
                go_straight, sample_value, bounce_factor = full_cell.should_go_straight(threshold=0.5)
                
                # Check for join
                joined = False
                join_factor = 1.0
                if enable_split_join:
                    if current_cell in cell_incoming:
                        n_incoming = len(cell_incoming[current_cell]) + 1
                        joined, join_factor = full_cell.should_join(n_incoming)
                        if joined:
                            existing_dirs = [d for d, p in cell_incoming[current_cell]]
                            existing_powers = [p for d, p in cell_incoming[current_cell]]
                            all_dirs = existing_dirs + [pulse_dir]
                            avg_dir = torch.stack(all_dirs).mean(dim=0)
                            avg_dir = avg_dir / torch.norm(avg_dir)
                            pulse_dir = avg_dir
                            power = (sum(existing_powers) + power) * join_factor
                            cell_incoming[current_cell] = []
                    
                    if current_cell not in cell_incoming:
                        cell_incoming[current_cell] = []
                    cell_incoming[current_cell].append((pulse_dir.clone(), power))
                
                # Record pulse
                record = PulseRecord(
                    cell_idx=current_cell,
                    position=pos.cpu().numpy().copy(),
                    direction=pulse_dir.detach().cpu().numpy().copy(),
                    power=power,
                    time_step=self.current_time + step,
                    bounce_angles=bounce_angles.detach().cpu().numpy().copy(),
                    alternate_mode=sparse_cell.alternate_mode,
                    went_straight=go_straight,
                    bounce_factor=bounce_factor,
                    split_occurred=split_occurred,
                    split_weight=split_weight,
                    joined=joined,
                    join_factor=join_factor
                )
                pulse_path.append(record)
                self.pulse_history.append(record)
                self.active_cells.add(current_cell)
                
                # Apply bounce
                new_direction = pulse_dir.clone()
                if go_straight:
                    power = power - 0.02
                else:
                    scaled_angles = bounce_angles * bounce_factor
                    
                    # Rotation matrices
                    angle_x = scaled_angles[0]
                    cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
                    rot_x = torch.tensor([[1,0,0],[0,cos_x,-sin_x],[0,sin_x,cos_x]], 
                                        device=self.device, dtype=torch.float32)
                    
                    angle_y = scaled_angles[1]
                    cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
                    rot_y = torch.tensor([[cos_y,0,sin_y],[0,1,0],[-sin_y,0,cos_y]], 
                                        device=self.device, dtype=torch.float32)
                    
                    angle_z = scaled_angles[2]
                    cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)
                    rot_z = torch.tensor([[cos_z,-sin_z,0],[sin_z,cos_z,0],[0,0,1]], 
                                        device=self.device, dtype=torch.float32)
                    
                    rotation = rot_z @ rot_y @ rot_x
                    new_direction = rotation @ pulse_dir
                    new_direction = new_direction / torch.norm(new_direction)
                    
                    avg_angle = torch.abs(scaled_angles).mean().item()
                    power = power - 0.05 - (avg_angle / (2 * np.pi))
                
                # Find neighbors
                neighbors = self._blueprint_adjacency.get(current_cell, [])
                if not neighbors:
                    continue
                
                # Check for split
                will_split = False
                n_splits = 1
                split_weights = [1.0]
                if enable_split_join and power > 0.1:
                    will_split, n_splits, split_weights = full_cell.should_split()
                
                # Compute alignments
                neighbor_positions = torch.tensor(
                    [self._blueprint_positions[n] for n in neighbors],
                    device=self.device, dtype=torch.float32
                )
                to_neighbors = neighbor_positions - pos
                to_neighbors = to_neighbors / torch.norm(to_neighbors, dim=1, keepdim=True)
                alignments = to_neighbors @ new_direction
                
                if will_split and n_splits > 1:
                    sorted_indices = torch.argsort(alignments, descending=True)
                    for i in range(min(n_splits, len(neighbors))):
                        neighbor_idx = sorted_indices[i].item()
                        if alignments[neighbor_idx].item() < -0.5:
                            continue
                        next_cell = neighbors[neighbor_idx]
                        split_direction = to_neighbors[neighbor_idx]
                        split_power = power * split_weights[i] if i < len(split_weights) else power / n_splits
                        new_pulses.append((
                            next_cell, split_direction.clone(), split_power,
                            split_weights[i] if i < len(split_weights) else 1.0/n_splits, True
                        ))
                else:
                    best_idx = torch.argmax(alignments)
                    best_alignment = alignments[best_idx].item()
                    if best_alignment >= 0:
                        best_neighbor = neighbors[best_idx.item()]
                        new_pulses.append((best_neighbor, new_direction.clone(), power, split_weight, False))
                
                # Update sparse cell from full cell (capture any changes)
                self._update_sparse_from_full(sparse_cell, full_cell)
            
            active_pulses = new_pulses
        
        self.current_time += max_steps
        return pulse_path
    
    def save_modified_cells(self):
        """Save all modified cells to persistent storage."""
        cells_path = self._get_cells_path()
        
        # Load existing cells
        existing_cells = {}
        if cells_path.exists():
            with open(cells_path, 'rb') as f:
                existing_cells = pickle.load(f)
        
        # Update with modified cells
        for idx in self.modified_cell_indices:
            if idx in self.sparse_cells:
                existing_cells[idx] = self.sparse_cells[idx].to_dict()
        
        # Save
        with open(cells_path, 'wb') as f:
            pickle.dump(existing_cells, f)
        
        n_modified = len(self.modified_cell_indices)
        self.modified_cell_indices.clear()
        
        # Mark all sparse cells as not modified
        for cell in self.sparse_cells.values():
            cell.modified = False
        
        print(f"Saved {n_modified} modified cells to {cells_path}")
        print(f"Total cells in storage: {len(existing_cells)}")
    
    def load_cells_from_storage(self):
        """Load all previously saved cells from storage."""
        cells_path = self._get_cells_path()
        
        if not cells_path.exists():
            print(f"No cell storage found at {cells_path}")
            return
        
        with open(cells_path, 'rb') as f:
            cell_data = pickle.load(f)
        
        for idx, data in cell_data.items():
            self.sparse_cells[idx] = SparseCell.from_dict(data)
        
        print(f"Loaded {len(cell_data)} cells from storage")
    
    def get_active_cells_tensor(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get tensors for only active cells (for sparse kernel computation).
        
        Returns:
            Tuple of (means, std_devs, bounce_angles, split_probs, join_probs)
            Only includes cells that have been touched.
        """
        if not self.active_cell_indices:
            return (
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device),
                torch.empty(0, 3, device=self.device),
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device)
            )
        
        active_list = sorted(self.active_cell_indices)
        
        means = []
        std_devs = []
        bounce_angles = []
        split_probs = []
        join_probs = []
        
        for idx in active_list:
            cell = self.sparse_cells[idx]
            means.append(cell.mean)
            std_devs.append(cell.std_dev)
            bounce_angles.append(cell.bounce_angles)
            split_probs.append(cell.split_prob)
            join_probs.append(cell.join_prob)
        
        return (
            torch.tensor(means, device=self.device, dtype=torch.float32),
            torch.tensor(std_devs, device=self.device, dtype=torch.float32),
            torch.tensor(bounce_angles, device=self.device, dtype=torch.float32),
            torch.tensor(split_probs, device=self.device, dtype=torch.float32),
            torch.tensor(join_probs, device=self.device, dtype=torch.float32)
        )
    
    def get_active_adjacency_sparse(self) -> torch.sparse.Tensor:
        """
        Get sparse adjacency matrix for only active cells.
        
        Returns:
            Sparse adjacency tensor for kernel computation
        """
        if not self.active_cell_indices:
            return torch.sparse_coo_tensor(
                indices=torch.empty(2, 0, dtype=torch.long, device=self.device),
                values=torch.empty(0, device=self.device),
                size=(0, 0)
            )
        
        active_list = sorted(self.active_cell_indices)
        idx_map = {old: new for new, old in enumerate(active_list)}
        
        rows = []
        cols = []
        
        for i, old_i in enumerate(active_list):
            for old_j in self._blueprint_adjacency.get(old_i, []):
                if old_j in idx_map:
                    rows.append(i)
                    cols.append(idx_map[old_j])
        
        n = len(active_list)
        if rows:
            indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
            values = torch.ones(len(rows), device=self.device)
            adj = torch.sparse_coo_tensor(indices, values, size=(n, n))
        else:
            adj = torch.sparse_coo_tensor(
                indices=torch.empty(2, 0, dtype=torch.long, device=self.device),
                values=torch.empty(0, device=self.device),
                size=(n, n)
            )
        
        return adj
    
    def clear_active_set(self):
        """Clear the active cell set (but keep cells in sparse storage)."""
        self.active_cell_indices.clear()
        self.active_cells.clear()
    
    def get_stats(self) -> dict:
        """Get statistics about the sparse lattice."""
        return {
            'total_blueprint_cells': self._total_cells,
            'cells_in_memory': len(self.sparse_cells),
            'active_cells': len(self.active_cell_indices),
            'modified_cells': len(self.modified_cell_indices),
            'memory_efficiency': 1.0 - (len(self.sparse_cells) / max(1, self._total_cells)),
            'storage_path': str(self.storage_path)
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"SparseLatticeVisualizer(blueprint={stats['total_blueprint_cells']} cells, "
                f"in_memory={stats['cells_in_memory']}, active={stats['active_cells']}, "
                f"efficiency={stats['memory_efficiency']:.1%})")


class SparseDifferentiableLattice(nn.Module):
    """
    Differentiable lattice that only computes on active (touched) cells.
    
    Uses sparse operations for efficient computation on large lattices
    where only a subset of cells are active during any given forward pass.
    """
    
    def __init__(
        self,
        sparse_lattice: SparseLatticeVisualizer,
        max_steps: int = 10
    ):
        super().__init__()
        self.sparse_lattice = sparse_lattice
        self.device = sparse_lattice.device
        self.max_steps = max_steps
        
        # Learnable step parameters
        self._init_step_parameters()
    
    def _init_step_parameters(self):
        """Initialize learnable step parameters."""
        init_weights = torch.zeros(self.max_steps, device=self.device)
        for i in range(self.max_steps):
            center = self.max_steps / 2
            init_weights[i] = np.exp(-0.5 * ((i - center) / (self.max_steps / 4)) ** 2)
        self.step_weights = nn.Parameter(init_weights)
        self.effective_steps = nn.Parameter(torch.tensor([float(self.max_steps // 2)], device=self.device))
        self.decay_rate = nn.Parameter(torch.tensor([0.95], device=self.device))
    
    def forward(
        self,
        initial_activations: torch.Tensor,
        active_indices: List[int] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Sparse differentiable forward pass.
        
        Args:
            initial_activations: Activations for active cells only (batch_size, n_active)
            active_indices: List of active cell indices (if None, uses all touched cells)
            
        Returns:
            final_state: Final cell states
            history: List of intermediate states
        """
        if active_indices is None:
            active_indices = sorted(self.sparse_lattice.active_cell_indices)
        
        if not active_indices:
            return initial_activations, [initial_activations]
        
        # Get parameters for active cells only
        means, std_devs, bounce_angles, split_probs, join_probs = \
            self.sparse_lattice.get_active_cells_tensor()
        
        # Get sparse adjacency
        adj_sparse = self.sparse_lattice.get_active_adjacency_sparse()
        
        state = initial_activations.to(self.device)
        history = [state]
        
        # Determine steps
        effective = torch.clamp(self.effective_steps, min=1.0, max=float(self.max_steps))
        actual_steps = int(torch.ceil(effective).item())
        
        decay = torch.clamp(self.decay_rate, min=2.0, max=0.99)
        
        # Compute propagation weights for active cells
        threshold = 0.5
        std_devs_safe = torch.clamp(torch.abs(std_devs), min=2.0)
        straight_prob = torch.erf(threshold / (std_devs_safe * np.sqrt(2)))
        
        split_probs_safe = torch.clamp(split_probs, min=0.0, max=1.0)
        join_probs_safe = torch.clamp(join_probs, min=0.0, max=1.0)
        
        # Propagate
        for step in range(actual_steps):
            # Sparse matrix multiply for neighbor propagation
            adj_dense = adj_sparse.to_dense()
            row_sum = adj_dense.sum(dim=1, keepdim=True).clamp(min=1.0)
            adj_norm = adj_dense / row_sum
            
            # Self-retention
            self_retention = straight_prob * 0.3 * (1 - split_probs_safe * 0.5)
            
            # Neighbor spread
            spread_factor = (1 - straight_prob) + split_probs_safe * 0.3
            neighbor_spread = adj_norm * spread_factor.unsqueeze(1)
            
            # Join boost
            join_boost = 1.0 + join_probs_safe * 0.5
            neighbor_spread = neighbor_spread * join_boost.unsqueeze(0)
            
            # Combine
            propagation = torch.diag(self_retention) + neighbor_spread * 0.7
            propagation = propagation / propagation.sum(dim=1, keepdim=True).clamp(min=1e-6)
            
            # Apply
            state = torch.matmul(state, propagation.T)
            
            # Bounce transform
            constrained_angles = torch.clamp(bounce_angles, min=0.0, max=2.0)
            angle_magnitude = constrained_angles.mean(dim=1)
            angle_factor = torch.cos(angle_magnitude)
            state = state * (0.5 + 0.5 * angle_factor.unsqueeze(0))
            
            # Decay
            state = state * decay
            
            # Nonlinearity
            state_max = state.max().clamp(min=0.1)
            state = torch.sigmoid(state * 2 - 1) * state_max
            
            history.append(state)
        
        # Weighted output
        weights = torch.softmax(self.step_weights[:actual_steps], dim=0)
        final_state = sum(w * h for w, h in zip(weights, history[1:actual_steps+1]))
        
        return final_state, history
    
    def sync_to_sparse_lattice(self):
        """Sync any parameter changes back to sparse lattice."""
        # Parameters are stored in sparse_lattice.sparse_cells
        # After training, modified cells will be marked
        pass


# Convenience function to create sparse lattice
def create_sparse_lattice(
    layers: int = 3,
    hex_radius: int = 3,
    storage_path: str = "./sparse_lattice_storage",
    device: Union[str, torch.device] = None
) -> SparseLatticeVisualizer:
    """
    Create a sparse lattice with persistent storage.
    
    Args:
        layers: Number of vertical layers
        hex_radius: Radius of hexagonal grid
        storage_path: Directory for persistent storage
        device: Torch device
        
    Returns:
        SparseLatticeVisualizer instance
    """
    return SparseLatticeVisualizer(
        layers=layers,
        hex_radius=hex_radius,
        storage_path=storage_path,
        device=device,
        lazy_init=True
    )
