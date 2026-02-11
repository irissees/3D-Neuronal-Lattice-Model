"""
Toroidal Lattice: 3D torus-shaped arrangement of cells.

A torus has wrap-around connectivity - cells on opposite edges connect,
meaning there are no "boundary" cells. Every cell has the same number of neighbors.

Key properties:
- Parameterized by major radius R and minor radius r
- Cells arranged on torus surface using (θ, φ) coordinates
- θ: angle around the tube (minor circle)
- φ: angle around the torus (major circle)
- Wrap-around connectivity in both directions
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import pickle
import json
import plotly.graph_objects as go

from cell import Cell, DEVICE


@dataclass
class ToroidalCell:
    """Lightweight cell data for sparse storage on torus."""
    idx: int
    position: np.ndarray  # (x, y, z) in 3D space
    theta_idx: int  # Index around tube
    phi_idx: int    # Index around torus
    mean: float
    std_dev: float
    bounce_angles: np.ndarray
    split_prob: float
    join_prob: float
    alternate_mode: str
    touched: bool = False
    modified: bool = False
    
    def to_dict(self) -> dict:
        return {
            'idx': self.idx,
            'position': self.position.tolist(),
            'theta_idx': self.theta_idx,
            'phi_idx': self.phi_idx,
            'mean': self.mean,
            'std_dev': self.std_dev,
            'bounce_angles': self.bounce_angles.tolist(),
            'split_prob': self.split_prob,
            'join_prob': self.join_prob,
            'alternate_mode': self.alternate_mode,
            'touched': self.touched,
            'modified': self.modified
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ToroidalCell':
        return cls(
            idx=d['idx'],
            position=np.array(d['position']),
            theta_idx=d['theta_idx'],
            phi_idx=d['phi_idx'],
            mean=d['mean'],
            std_dev=d['std_dev'],
            bounce_angles=np.array(d['bounce_angles']),
            split_prob=d['split_prob'],
            join_prob=d['join_prob'],
            alternate_mode=d['alternate_mode'],
            touched=d.get('touched', False),
            modified=d.get('modified', False)
        )


class ToroidalLatticeVisualizer:
    """
    3D Toroidal lattice with on-demand cell creation.
    
    Cells are arranged on the surface of a torus, with wrap-around connectivity
    in both directions (θ and φ). This means:
    - No edge effects - every cell has exactly the same number of neighbors
    - Natural for representing cyclic/periodic data
    - Smooth, continuous surface topology
    
    Parameters:
    - n_theta: Number of cells around the tube (minor circle)
    - n_phi: Number of cells around the torus (major circle)
    - major_radius: Distance from torus center to tube center (R)
    - minor_radius: Radius of the tube (r)
    """
    
    def __init__(
        self,
        n_theta: int = 16,
        n_phi: int = 32,
        major_radius: float = 3.0,
        minor_radius: float = 1.0,
        storage_path: str = None,
        lazy_init: bool = True,
        device: str = DEVICE
    ):
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.storage_path = Path(storage_path) if storage_path else None
        self.lazy_init = lazy_init
        self.device = device
        
        # Total cells on torus surface
        self._total_cells = n_theta * n_phi
        
        # Sparse cell storage
        self.sparse_cells: Dict[int, ToroidalCell] = {}
        self.active_cell_indices: Set[int] = set()
        self.modified_cell_indices: Set[int] = set()
        
        # Blueprint storage
        self._blueprint_positions: Optional[np.ndarray] = None
        self._blueprint_adjacency: Optional[Dict[int, List[int]]] = None
        self._theta_phi_to_idx: Optional[Dict[Tuple[int, int], int]] = None
        self._idx_to_theta_phi: Optional[Dict[int, Tuple[int, int]]] = None
        
        # Ensure storage directory exists before loading/building blueprint
        if storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Build or load blueprint
        self._load_or_build_blueprint()
    
    def _load_or_build_blueprint(self):
        """Load blueprint from disk or build it."""
        blueprint_file = None
        if self.storage_path:
            blueprint_file = self.storage_path / f"torus_blueprint_T{self.n_theta}_P{self.n_phi}.json"
        
        if blueprint_file and blueprint_file.exists():
            print(f"Loading torus blueprint from {blueprint_file}")
            with open(blueprint_file, 'r') as f:
                data = json.load(f)
            self._blueprint_positions = np.array(data['positions'])
            self._blueprint_adjacency = {int(k): v for k, v in data['adjacency'].items()}
            self._theta_phi_to_idx = {tuple(k.split(',')): v for k, v in data['theta_phi_to_idx'].items()}
            self._theta_phi_to_idx = {(int(k[0]), int(k[1])): v for k, v in self._theta_phi_to_idx.items()}
            self._idx_to_theta_phi = {v: k for k, v in self._theta_phi_to_idx.items()}
        else:
            print("Building torus blueprint...")
            self._build_blueprint()
            if blueprint_file:
                self._save_blueprint(blueprint_file)
    
    def _build_blueprint(self):
        """Build the torus lattice blueprint (positions and adjacency)."""
        positions = []
        self._theta_phi_to_idx = {}
        self._idx_to_theta_phi = {}
        
        idx = 0
        for ti in range(self.n_theta):
            theta = 2 * np.pi * ti / self.n_theta
            for pi in range(self.n_phi):
                phi = 2 * np.pi * pi / self.n_phi
                
                # Torus parametric equations
                x = (self.major_radius + self.minor_radius * np.cos(theta)) * np.cos(phi)
                y = (self.major_radius + self.minor_radius * np.cos(theta)) * np.sin(phi)
                z = self.minor_radius * np.sin(theta)
                
                positions.append([x, y, z])
                self._theta_phi_to_idx[(ti, pi)] = idx
                self._idx_to_theta_phi[idx] = (ti, pi)
                idx += 1
        
        self._blueprint_positions = np.array(positions)
        
        # Build adjacency with wrap-around
        self._blueprint_adjacency = {}
        for idx in range(self._total_cells):
            ti, pi = self._idx_to_theta_phi[idx]
            neighbors = []
            
            # 4-connected neighbors on torus surface (with wrap-around)
            # Neighbor in theta direction (around tube)
            neighbors.append(self._theta_phi_to_idx[((ti + 1) % self.n_theta, pi)])
            neighbors.append(self._theta_phi_to_idx[((ti - 1) % self.n_theta, pi)])
            # Neighbor in phi direction (around torus)
            neighbors.append(self._theta_phi_to_idx[(ti, (pi + 1) % self.n_phi)])
            neighbors.append(self._theta_phi_to_idx[(ti, (pi - 1) % self.n_phi)])
            
            # Optional: 8-connected (add diagonals)
            neighbors.append(self._theta_phi_to_idx[((ti + 1) % self.n_theta, (pi + 1) % self.n_phi)])
            neighbors.append(self._theta_phi_to_idx[((ti + 1) % self.n_theta, (pi - 1) % self.n_phi)])
            neighbors.append(self._theta_phi_to_idx[((ti - 1) % self.n_theta, (pi + 1) % self.n_phi)])
            neighbors.append(self._theta_phi_to_idx[((ti - 1) % self.n_theta, (pi - 1) % self.n_phi)])
            
            self._blueprint_adjacency[idx] = neighbors
        
        print(f"Torus blueprint built: {self._total_cells} cells, "
              f"{self.n_theta} around tube × {self.n_phi} around torus")
    
    def _save_blueprint(self, filepath: Path):
        """Save blueprint to disk."""
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'n_theta': self.n_theta,
            'n_phi': self.n_phi,
            'major_radius': self.major_radius,
            'minor_radius': self.minor_radius,
            'positions': self._blueprint_positions.tolist(),
            'adjacency': {str(k): v for k, v in self._blueprint_adjacency.items()},
            'theta_phi_to_idx': {f"{k[0]},{k[1]}": v for k, v in self._theta_phi_to_idx.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Torus blueprint saved to {filepath}")
    
    def _generate_cell(self, idx: int) -> ToroidalCell:
        """Generate a single cell on-demand with random parameters."""
        if idx < 0 or idx >= self._total_cells:
            raise ValueError(f"Cell index {idx} out of range [0, {self._total_cells})")
        
        position = self._blueprint_positions[idx]
        ti, pi = self._idx_to_theta_phi[idx]
        
        # Alternate mode based on position on torus
        alternate_mode = 'closer' if (ti + pi) % 2 == 0 else 'further'
        
        # Random initialization
        if alternate_mode == 'closer':
            mean = np.random.rand() * 2 - 1
            std_dev = np.random.rand() * 0.5 + 2.0
            split_prob = np.random.rand() * 0.2
            join_prob = np.random.rand() * 0.3 + 0.2
            bounce_angles = np.random.rand(3) * 0.5
        else:
            mean = np.random.rand() * 2 - 1
            std_dev = np.random.rand() * 1.0 + 2.5
            split_prob = np.random.rand() * 0.3 + 0.2
            join_prob = np.random.rand() * 0.2
            bounce_angles = np.random.rand(3) * 1.0 + 1.0
        
        return ToroidalCell(
            idx=idx,
            position=position,
            theta_idx=ti,
            phi_idx=pi,
            mean=mean,
            std_dev=max(2.0, std_dev),
            bounce_angles=bounce_angles,
            split_prob=min(1.0, max(0.0, split_prob)),
            join_prob=min(1.0, max(0.0, join_prob)),
            alternate_mode=alternate_mode,
            touched=True,
            modified=True
        )
    
    def _get_or_create_cell(self, idx: int) -> ToroidalCell:
        """Get cell from sparse storage or create it on-demand."""
        if idx not in self.sparse_cells:
            self.sparse_cells[idx] = self._generate_cell(idx)
            self.modified_cell_indices.add(idx)
        
        self.active_cell_indices.add(idx)
        self.sparse_cells[idx].touched = True
        return self.sparse_cells[idx]
    
    def _get_neighbor_indices(self, idx: int) -> List[int]:
        """Get neighbor indices for a cell (wrap-around on torus)."""
        if self._blueprint_adjacency is None:
            return []
        return self._blueprint_adjacency.get(idx, [])
    
    def get_theta_phi(self, idx: int) -> Tuple[int, int]:
        """Get (theta_idx, phi_idx) for a cell index."""
        return self._idx_to_theta_phi.get(idx, (0, 0))
    
    def get_idx_from_theta_phi(self, theta_idx: int, phi_idx: int) -> int:
        """Get cell index from (theta_idx, phi_idx), with wrap-around."""
        ti = theta_idx % self.n_theta
        pi = phi_idx % self.n_phi
        return self._theta_phi_to_idx.get((ti, pi), 0)
    
    def save_modified_cells(self):
        """Save modified cells to disk."""
        if not self.storage_path or not self.modified_cell_indices:
            return
        
        cells_file = self.storage_path / "torus_cells.pkl"
        
        # Load existing cells
        existing = {}
        if cells_file.exists():
            with open(cells_file, 'rb') as f:
                existing = pickle.load(f)
        
        # Update with modified cells
        for idx in self.modified_cell_indices:
            if idx in self.sparse_cells:
                existing[idx] = self.sparse_cells[idx].to_dict()
        
        # Save
        with open(cells_file, 'wb') as f:
            pickle.dump(existing, f)
        
        n_saved = len(self.modified_cell_indices)
        self.modified_cell_indices.clear()
        print(f"Saved {n_saved} modified torus cells")
    
    def load_cells_from_storage(self):
        """Load all saved cells from storage."""
        if not self.storage_path:
            return
        
        cells_file = self.storage_path / "torus_cells.pkl"
        if not cells_file.exists():
            return
        
        with open(cells_file, 'rb') as f:
            data = pickle.load(f)
        
        for idx, cell_dict in data.items():
            self.sparse_cells[int(idx)] = ToroidalCell.from_dict(cell_dict)
        
        print(f"Loaded {len(data)} torus cells from storage")
    
    def get_stats(self) -> Dict:
        """Get statistics about the toroidal lattice."""
        return {
            'total_cells_blueprint': self._total_cells,
            'cells_in_memory': len(self.sparse_cells),
            'active_cells': len(self.active_cell_indices),
            'modified_cells': len(self.modified_cell_indices),
            'n_theta': self.n_theta,
            'n_phi': self.n_phi,
            'major_radius': self.major_radius,
            'minor_radius': self.minor_radius,
            'memory_efficiency': 1 - (len(self.sparse_cells) / self._total_cells) if self._total_cells > 0 else 1.0
        }
    
    def clear_active_set(self):
        """Clear the active cell tracking (keeps cells in memory)."""
        self.active_cell_indices.clear()
    
    def visualize_plotly(
        self,
        show_all_positions: bool = True,
        show_active_only: bool = False,
        cell_size: float = 5,
        opacity: float = 0.7,
        colorscale: str = 'Viridis',
        title: str = None
    ) -> go.Figure:
        """
        Create interactive 3D visualization of the torus lattice.
        """
        fig = go.Figure()
        
        # Show blueprint positions (all potential cells)
        if show_all_positions and not show_active_only:
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
                name='Blueprint (inactive)',
                hovertemplate='Cell %{customdata}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}',
                customdata=list(range(self._total_cells))
            ))
        
        # Show active/created cells
        if self.sparse_cells:
            active_indices = list(self.sparse_cells.keys()) if not show_active_only else list(self.active_cell_indices)
            
            if active_indices:
                positions = np.array([self.sparse_cells[i].position for i in active_indices if i in self.sparse_cells])
                means = np.array([self.sparse_cells[i].mean for i in active_indices if i in self.sparse_cells])
                
                if len(positions) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=positions[:, 0],
                        y=positions[:, 1],
                        z=positions[:, 2],
                        mode='markers',
                        marker=dict(
                            size=cell_size,
                            color=means,
                            colorscale=colorscale,
                            opacity=opacity,
                            colorbar=dict(title='Mean')
                        ),
                        name='Active cells',
                        hovertemplate=(
                            'Cell %{customdata[0]}<br>'
                            'θ: %{customdata[1]}, φ: %{customdata[2]}<br>'
                            'Mean: %{customdata[3]:.3f}<br>'
                            'Std: %{customdata[4]:.3f}'
                        ),
                        customdata=[
                            [i, self.sparse_cells[i].theta_idx, self.sparse_cells[i].phi_idx,
                             self.sparse_cells[i].mean, self.sparse_cells[i].std_dev]
                            for i in active_indices if i in self.sparse_cells
                        ]
                    ))
        
        # Add torus wireframe for reference
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, 2 * np.pi, 30)
        U, V = np.meshgrid(u, v)
        X = (self.major_radius + self.minor_radius * np.cos(V)) * np.cos(U)
        Y = (self.major_radius + self.minor_radius * np.cos(V)) * np.sin(U)
        Z = self.minor_radius * np.sin(V)
        
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.1,
            colorscale='Blues',
            showscale=False,
            name='Torus surface'
        ))
        
        # Layout
        fig.update_layout(
            title=title or f'Toroidal Lattice ({self.n_theta}×{self.n_phi} = {self._total_cells} cells)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            showlegend=True,
            width=900,
            height=700
        )
        
        return fig


class ToroidalDifferentiableLattice(nn.Module):
    """
    Differentiable toroidal lattice for training.
    
    Similar to OnDemandDifferentiableLattice but specialized for torus topology.
    """
    
    def __init__(self, torus_lattice: ToroidalLatticeVisualizer, max_steps: int = 10):
        super().__init__()
        self.torus_lattice = torus_lattice
        self.max_steps = max_steps
        self.device = torus_lattice.device
        
        # Learnable parameters
        self.step_weights = nn.Parameter(torch.ones(max_steps) / max_steps)
        self.decay_rate = nn.Parameter(torch.tensor(0.85))
        self.propagation_threshold = nn.Parameter(torch.tensor(0.01))
    
    def forward(
        self,
        entry_probs: torch.Tensor,
        entry_strengths: torch.Tensor,
        entry_indices: List[List[int]]
    ) -> Tuple[torch.Tensor, List[int], Dict]:
        """
        Propagate signal through torus, creating cells on-demand.
        
        The torus wrap-around means propagation can loop back on itself!
        """
        batch_size = entry_probs.shape[0]
        device = entry_probs.device
        
        # Collect entry points
        all_entry_indices = set()
        for indices in entry_indices:
            all_entry_indices.update(indices)
        
        # Create entry cells
        touched_cells = set(all_entry_indices)
        for idx in all_entry_indices:
            self.torus_lattice._get_or_create_cell(idx)
        
        touched_list = sorted(touched_cells)
        idx_to_pos = {idx: i for i, idx in enumerate(touched_list)}
        
        # Initialize activations
        current_activation = torch.zeros(batch_size, len(touched_list), device=device)
        for i, idx in enumerate(touched_list):
            current_activation[:, i] = entry_probs[:, idx]
        
        history = {'steps': [], 'cells_created': [len(touched_cells)]}
        step_outputs = []
        
        decay = torch.clamp(self.decay_rate, min=0.5, max=0.99)
        threshold = torch.clamp(self.propagation_threshold, min=0.001, max=0.1).item()
        
        for step in range(self.max_steps):
            with torch.no_grad():
                max_activation = current_activation.max(dim=0)[0]
                active_mask = max_activation > threshold
                active_positions = torch.where(active_mask)[0].tolist()
            
            if not active_positions:
                step_outputs.append(current_activation.clone())
                continue
            
            # Get neighbors (with torus wrap-around!)
            new_cells = set()
            for pos in active_positions:
                cell_idx = touched_list[pos]
                neighbors = self.torus_lattice._get_neighbor_indices(cell_idx)
                for neighbor_idx in neighbors:
                    if neighbor_idx not in touched_cells:
                        new_cells.add(neighbor_idx)
                        self.torus_lattice._get_or_create_cell(neighbor_idx)
            
            if new_cells:
                touched_cells.update(new_cells)
                touched_list = sorted(touched_cells)
                idx_to_pos = {idx: i for i, idx in enumerate(touched_list)}
                
                old_activation = current_activation
                current_activation = torch.zeros(batch_size, len(touched_list), device=device)
                for old_pos, old_idx in enumerate(sorted(touched_cells - new_cells)):
                    new_pos = idx_to_pos[old_idx]
                    current_activation[:, new_pos] = old_activation[:, old_pos]
            
            history['cells_created'].append(len(new_cells))
            
            # Build adjacency
            n_touched = len(touched_list)
            adjacency = torch.zeros(n_touched, n_touched, device=device)
            
            for i, idx in enumerate(touched_list):
                neighbors = self.torus_lattice._get_neighbor_indices(idx)
                for neighbor_idx in neighbors:
                    if neighbor_idx in idx_to_pos:
                        j = idx_to_pos[neighbor_idx]
                        adjacency[i, j] = 1.0
            
            row_sum = adjacency.sum(dim=1, keepdim=True) + 1e-8
            adjacency = adjacency / row_sum
            
            # Propagate
            next_activation = (
                0.3 * current_activation +
                0.7 * torch.matmul(current_activation, adjacency)
            ) * decay
            
            for i, idx in enumerate(touched_list):
                if idx in all_entry_indices:
                    next_activation[:, i] = next_activation[:, i] + entry_probs[:, idx] * 0.1
            
            current_activation = next_activation
            step_outputs.append(current_activation.clone())
        
        # Combine steps
        weights = torch.softmax(self.step_weights, dim=0)
        max_cells = len(touched_list)
        final_state = torch.zeros(batch_size, max_cells, device=device)
        
        for i, step_out in enumerate(step_outputs):
            if step_out.shape[1] < max_cells:
                padded = torch.zeros(batch_size, max_cells, device=device)
                padded[:, :step_out.shape[1]] = step_out
                step_out = padded
            final_state = final_state + weights[i] * step_out
        
        history['touched_indices'] = touched_list
        
        return final_state, touched_list, history


def create_toroidal_lattice(
    n_theta: int = 16,
    n_phi: int = 32,
    major_radius: float = 3.0,
    minor_radius: float = 1.0,
    storage_path: str = None
) -> ToroidalLatticeVisualizer:
    """Factory function to create a toroidal lattice."""
    return ToroidalLatticeVisualizer(
        n_theta=n_theta,
        n_phi=n_phi,
        major_radius=major_radius,
        minor_radius=minor_radius,
        storage_path=storage_path,
        lazy_init=True
    )


if __name__ == "__main__":
    # Quick test
    torus = create_toroidal_lattice(n_theta=12, n_phi=24)
    print(f"Created torus lattice with {torus._total_cells} cells")
    print(f"Stats: {torus.get_stats()}")
    
    # Create some cells
    for i in range(10):
        torus._get_or_create_cell(i * 10)
    
    print(f"After creating 10 cells: {torus.get_stats()}")
