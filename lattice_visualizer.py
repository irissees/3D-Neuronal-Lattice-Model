import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from cell import Cell, DEVICE


class UnionFind:
    """Union-Find (Disjoint Set Union) data structure for Kruskal's algorithm."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if x and y were in different sets."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


@dataclass
class PulseRecord:
    """Record of an activation pulse at a specific time."""
    cell_idx: int
    position: np.ndarray
    direction: np.ndarray  # Unit vector of pulse direction
    power: float
    time_step: int
    bounce_angles: np.ndarray  # 3-axis bounce angles [X, Y, Z] (with -1 applied)
    alternate_mode: str  # 'closer' or 'further'
    went_straight: bool = True  # Whether pulse went straight or bounced
    bounce_factor: float = 0.0  # How much bounce was applied (0 = straight)
    split_occurred: bool = False  # Whether this pulse was the result of a split
    split_weight: float = 1.0  # Power fraction from split (1.0 = no split)
    joined: bool = False  # Whether this pulse resulted from a join
    join_factor: float = 1.0  # Boost from joining (1.0 = no boost)


@dataclass 
class OriginPoint:
    """Origin point for pulse generation (Section 4 of paper)."""
    position: np.ndarray
    direction: np.ndarray  # Initial outward direction


class LatticeVisualizer:
    """
    Visualizes cells in a 3D hexagonal lattice and tracks activation pulse propagation.
    Uses PyTorch with MPS (Apple Silicon GPU) when available.
    
    Based on the Cross-Dimension Toroidal Pulse Architecture:
    - Cells embedded in 3D lattice (stacked hexagonal grids)
    - Pulses propagate through the space with tracked direction
    - Cell states correspond to bounce angles (0-360 degrees)
    - Supports toroidal wave pattern visualization
    """
    
    def __init__(
        self,
        layers: int = 3,
        hex_radius: int = 3,
        layer_spacing: float = 1.0,
        hex_spacing: float = 0.9,
        device: Union[str, torch.device] = None
    ):
        """
        Initialize the lattice visualizer.
        
        Args:
            layers: Number of vertical layers in the lattice
            hex_radius: Radius of hexagonal grid per layer (in cells)
            layer_spacing: Vertical distance between layers
            hex_spacing: Horizontal distance between adjacent cells (default 0.9)
            device: Torch device ('mps', 'cuda', 'cpu') - defaults to MPS if available
        """
        self.layers = layers
        self.hex_radius = hex_radius
        self.layer_spacing = layer_spacing
        self.hex_spacing = hex_spacing
        
        # Set up device
        if device is None:
            self.device = DEVICE
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        print(f"LatticeVisualizer using device: {self.device}")
        
        # Cell storage
        self.cells: List[Cell] = []
        self.positions: torch.Tensor = None  # Store positions as torch tensor
        self.positions_np: np.ndarray = None  # Numpy version for visualization
        self.adjacency: Dict[int, List[int]] = {}  # Cell index -> neighbor indices
        
        # Pulse tracking
        self.pulse_history: List[PulseRecord] = []
        self.current_time: int = 0
        self.active_cells: set = set()
        
        # Origin points (Section 4: three origin points for out-and-back flow)
        self.origin_points: List[OriginPoint] = []
        
        # Build the lattice
        self._build_hexagonal_lattice()
        self._compute_adjacency()
        self._initialize_origin_points()
    
    def _hex_to_cartesian(self, q: int, r: int, layer: int) -> torch.Tensor:
        """Convert axial hex coordinates to 3D Cartesian coordinates."""
        x = self.hex_spacing * (3/2 * q)
        y = self.hex_spacing * (np.sqrt(3) * (r + q/2))
        z = layer * self.layer_spacing
        return torch.tensor([x, y, z], device=self.device, dtype=torch.float32)
    
    def _build_hexagonal_lattice(self):
        """Build the 3D hexagonal lattice structure using PyTorch tensors."""
        positions = []
        cell_idx = 0
        
        for layer in range(self.layers):
            for q in range(-self.hex_radius, self.hex_radius + 1):
                r1 = max(-self.hex_radius, -q - self.hex_radius)
                r2 = min(self.hex_radius, -q + self.hex_radius)
                for r in range(r1, r2 + 1):
                    pos = self._hex_to_cartesian(q, r, layer)
                    positions.append(pos)
                    
                    # Alternate between 'closer' and 'further' modes
                    # 'closer' = low std_dev, pulse more likely to go straight
                    # 'further' = high std_dev, pulse more likely to bounce
                    alternate_mode = 'closer' if cell_idx % 2 == 0 else 'further'
                    
                    # Randomized initialization for mean (weight) and std_dev (bias)
                    # mean: random in [-1, 1] - shifts the center of direction probability
                    random_mean = (torch.rand(1, device=self.device) * 2 - 1).item()
                    
                    # std_dev: random but influenced by alternate_mode (min 0.5)
                    # 'closer' mode: lower std_dev range [0.5, 0.7]
                    # 'further' mode: higher std_dev range [0.7, 1.0]
                    if alternate_mode == 'closer':
                        random_std = (torch.rand(1, device=self.device) * 0.2 + 0.5).item()
                    else:
                        random_std = (torch.rand(1, device=self.device) * 0.3 + 0.7).item()
                    
                    # Randomized bounce angles [0, 2] radians for each axis
                    random_bounce = torch.rand(3, device=self.device) * 2.0
                    
                    # Create cell with randomized parameters
                    cell = Cell(
                        mean=random_mean,
                        std_dev=random_std,
                        bounce_angles=random_bounce,
                        state=cell_idx,
                        device=self.device,
                        alternate_mode=alternate_mode
                    )
                    self.cells.append(cell)
                    cell_idx += 1
        
        # Stack positions into a single tensor
        self.positions = torch.stack(positions)
        # Keep numpy version for visualization
        self.positions_np = self.positions.cpu().numpy()
    
    def _compute_adjacency(self):
        """Compute neighbor relationships using torch for GPU acceleration."""
        n_cells = len(self.cells)
        
        # Compute pairwise distances efficiently using torch
        # Using broadcasting: (n, 1, 3) - (1, n, 3) -> (n, n, 3)
        diff = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)
        distances = torch.norm(diff, dim=2)  # (n, n)
        
        # In axial hex coordinates, adjacent cells are sqrt(3) * hex_spacing apart
        # (due to the coordinate conversion: x = 1.5*q, y = sqrt(3)*(r + q/2))
        hex_adjacent_dist = self.hex_spacing * np.sqrt(3) * 1.1  # ~1.71 for spacing=0.9
        
        for i in range(n_cells):
            self.adjacency[i] = []
            for j in range(n_cells):
                if i != j:
                    dist = distances[i, j].item()
                    # Same layer neighbors (hex adjacency) - horizontal connections
                    if dist < hex_adjacent_dist:
                        self.adjacency[i].append(j)
                    # Cross-layer neighbors (vertical adjacency)
                    elif abs(dist - self.layer_spacing) < 0.2:
                        self.adjacency[i].append(j)
    
    def _initialize_origin_points(self, num_pulses: int = 3):
        """
        Initialize origin points for pulse projection (Section 4).
        
        All pulses start from the center with random directions across 360 degrees.
        
        Args:
            num_pulses: Number of pulses to create (default 3)
        """
        # All pulses originate from the center of the lattice
        center = self.positions.mean(dim=0).cpu().numpy()
        
        # Random directions across full 360 degrees
        for _ in range(num_pulses):
            angle = torch.rand(1, device=self.device).item() * 2 * np.pi
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            self.origin_points.append(OriginPoint(position=center.copy(), direction=direction))
    
    def get_cell_bounce_angles(self, cell_idx: int) -> torch.Tensor:
        """
        Get the 3-axis bounce angles for a cell (Section 5.2).
        
        Returns bounce angles [X, Y, Z] multiplied by -1.
        """
        cell = self.cells[cell_idx]
        return cell.get_bounce_angles()  # Returns angles * -1
    
    def get_cell_bounce_angles_raw(self, cell_idx: int) -> torch.Tensor:
        """Get the raw 3-axis bounce angles without -1 multiplication."""
        cell = self.cells[cell_idx]
        return cell.get_bounce_angles_raw()
    
    def set_cell_bounce_angles(self, cell_idx: int, angles: Union[list, torch.Tensor]):
        """
        Set the 3-axis bounce angles for a cell.
        
        Args:
            cell_idx: Index of the cell
            angles: 3-element list/tensor for [X, Y, Z] angles
        """
        cell = self.cells[cell_idx]
        cell.set_bounce_angles(angles)
    
    def propagate_pulse(
        self,
        start_cell: int,
        initial_direction: Union[np.ndarray, torch.Tensor],
        initial_power: float = 1.0,
        max_steps: int = 10,
        enable_split_join: bool = True
    ) -> List[PulseRecord]:
        """
        Propagate an activation pulse through the lattice using torch for GPU acceleration.
        Supports pulse splitting (one pulse becomes many) and joining (many become one).
        
        Args:
            start_cell: Index of starting cell
            initial_direction: Initial pulse direction (unit vector)
            initial_power: Initial pulse power
            max_steps: Maximum propagation steps
            enable_split_join: Whether to enable split/join behavior (default True)
            
        Returns:
            List of PulseRecord objects tracking the pulse path
        """
        pulse_path = []
        
        # Convert direction to torch tensor
        if isinstance(initial_direction, np.ndarray):
            direction = torch.tensor(initial_direction, device=self.device, dtype=torch.float32)
        else:
            direction = initial_direction.to(self.device)
        direction = direction / torch.norm(direction)
        
        # Active pulses: list of (cell_idx, direction, power, split_weight, split_occurred)
        active_pulses = [(start_cell, direction.clone(), initial_power, 1.0, False)]
        
        for step in range(max_steps):
            if not active_pulses:
                break
            
            # Track pulses arriving at each cell this step (for join logic)
            cell_incoming: Dict[int, List[Tuple[torch.Tensor, float]]] = {}  # cell -> [(direction, power), ...]
            
            new_pulses = []
            
            for current_cell, pulse_dir, power, split_weight, split_occurred in active_pulses:
                if power <= 0.01:  # Power threshold
                    continue
                
                pos = self.positions[current_cell]
                cell = self.cells[current_cell]
                
                # Get 3-axis bounce angles (with -1 applied)
                bounce_angles = self.get_cell_bounce_angles(current_cell)
                
                # Sample from cell's normal distribution to determine if ray bounces
                go_straight, sample_value, bounce_factor = cell.should_go_straight(threshold=0.5)
                
                # Check for join with other incoming pulses at this cell
                joined = False
                join_factor = 1.0
                if enable_split_join:
                    if current_cell in cell_incoming:
                        # There are already pulses here - check if we should join
                        n_incoming = len(cell_incoming[current_cell]) + 1
                        joined, join_factor = cell.should_join(n_incoming)
                        if joined:
                            # Combine with existing pulses
                            existing_dirs = [d for d, p in cell_incoming[current_cell]]
                            existing_powers = [p for d, p in cell_incoming[current_cell]]
                            # Average directions and sum powers with join boost
                            all_dirs = existing_dirs + [pulse_dir]
                            avg_dir = torch.stack(all_dirs).mean(dim=0)
                            avg_dir = avg_dir / torch.norm(avg_dir)
                            pulse_dir = avg_dir
                            power = (sum(existing_powers) + power) * join_factor
                            # Clear the cell_incoming so these pulses don't get processed again
                            cell_incoming[current_cell] = []
                    
                    # Track this pulse for potential joining
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
                    alternate_mode=cell.alternate_mode,
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
                
                # Apply bounce if needed
                new_direction = pulse_dir.clone()
                if go_straight:
                    # Ray continues straight - minimal power loss
                    power = power - 0.02
                else:
                    # Ray bounces - apply 3D rotation based on X, Y, Z angles (already in radians)
                    scaled_angles = bounce_angles * bounce_factor
                    
                    # X-axis rotation: deflects ray in the Y-Z plane
                    angle_x = scaled_angles[0]
                    cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
                    rot_x = torch.tensor([
                        [1, 0, 0],
                        [0, cos_x, -sin_x],
                        [0, sin_x, cos_x]
                    ], device=self.device, dtype=torch.float32)
                    
                    # Y-axis rotation: deflects ray in the X-Z plane
                    angle_y = scaled_angles[1]
                    cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
                    rot_y = torch.tensor([
                        [cos_y, 0, sin_y],
                        [0, 1, 0],
                        [-sin_y, 0, cos_y]
                    ], device=self.device, dtype=torch.float32)
                    
                    # Z-axis rotation: deflects ray in the X-Y plane
                    angle_z = scaled_angles[2]
                    cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)
                    rot_z = torch.tensor([
                        [cos_z, -sin_z, 0],
                        [sin_z, cos_z, 0],
                        [0, 0, 1]
                    ], device=self.device, dtype=torch.float32)
                    
                    # Combined rotation: R = Rz @ Ry @ Rx
                    rotation = rot_z @ rot_y @ rot_x
                    new_direction = rotation @ pulse_dir
                    new_direction = new_direction / torch.norm(new_direction)
                    
                    # More power loss when bouncing
                    avg_angle = torch.abs(scaled_angles).mean().item()
                    power = power - 0.05 - (avg_angle / (2 * np.pi))
                
                # Find neighbors for next step
                neighbors = self.adjacency.get(current_cell, [])
                if not neighbors:
                    continue
                
                # Check for split
                will_split = False
                n_splits = 1
                split_weights = [1.0]
                if enable_split_join and power > 0.1:  # Only split if there's enough power
                    will_split, n_splits, split_weights = cell.should_split()
                
                # Batch compute alignments for all neighbors
                neighbor_positions = self.positions[neighbors]
                to_neighbors = neighbor_positions - pos
                to_neighbors = to_neighbors / torch.norm(to_neighbors, dim=1, keepdim=True)
                alignments = to_neighbors @ new_direction
                
                if will_split and n_splits > 1:
                    # Split: send pulses to multiple neighbors
                    sorted_indices = torch.argsort(alignments, descending=True)
                    
                    for i in range(min(n_splits, len(neighbors))):
                        neighbor_idx = sorted_indices[i].item()
                        if alignments[neighbor_idx].item() < -0.5:  # Don't go backwards
                            continue
                        
                        next_cell = neighbors[neighbor_idx]
                        split_direction = to_neighbors[neighbor_idx]
                        split_power = power * split_weights[i] if i < len(split_weights) else power / n_splits
                        
                        new_pulses.append((
                            next_cell, 
                            split_direction.clone(), 
                            split_power, 
                            split_weights[i] if i < len(split_weights) else 1.0 / n_splits,
                            True  # Mark as split
                        ))
                else:
                    # No split: continue with best alignment
                    best_idx = torch.argmax(alignments)
                    best_alignment = alignments[best_idx].item()
                    
                    if best_alignment >= 0:
                        best_neighbor = neighbors[best_idx.item()]
                        new_pulses.append((
                            best_neighbor, 
                            new_direction.clone(), 
                            power, 
                            split_weight,
                            False
                        ))
            
            active_pulses = new_pulses
        
        self.current_time += max_steps
        return pulse_path
    
    def propagate_from_origins(self, power: float = 1.0, max_steps: int = 10) -> List[List[PulseRecord]]:
        """
        Propagate pulses from all three origin points (Section 4).
        Uses torch for GPU-accelerated distance calculations.
        
        Returns:
            List of pulse paths from each origin
        """
        all_paths = []
        
        for origin in self.origin_points:
            # Find nearest cell to origin using torch
            origin_pos = torch.tensor(origin.position, device=self.device, dtype=torch.float32)
            distances = torch.norm(self.positions - origin_pos, dim=1)
            start_cell = int(torch.argmin(distances).item())
            
            path = self.propagate_pulse(
                start_cell=start_cell,
                initial_direction=origin.direction,
                initial_power=power,
                max_steps=max_steps
            )
            all_paths.append(path)
        
        return all_paths
    
    def plot_lattice(
        self,
        ax: Optional[Axes3D] = None,
        show_cells: bool = True,
        show_connections: bool = True,
        show_origins: bool = True,
        cell_size: float = 50,
        alpha: float = 0.6
    ) -> Axes3D:
        """
        Plot the 3D lattice structure.
        
        Args:
            ax: Matplotlib 3D axes (created if None)
            show_cells: Whether to show cell positions
            show_connections: Whether to show adjacency connections
            show_origins: Whether to show origin points
            cell_size: Size of cell markers
            alpha: Transparency
            
        Returns:
            Matplotlib 3D axes
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot cells (using numpy array for matplotlib)
        if show_cells:
            colors = ['lightblue' if i not in self.active_cells else 'orange' 
                     for i in range(len(self.cells))]
            ax.scatter(
                self.positions_np[:, 0],
                self.positions_np[:, 1],
                self.positions_np[:, 2],
                c=colors,
                s=cell_size,
                alpha=alpha,
                edgecolors='black',
                linewidths=0.5
            )
        
        # Plot connections
        if show_connections:
            lines = []
            for i, neighbors in self.adjacency.items():
                for j in neighbors:
                    if i < j:  # Avoid duplicates
                        lines.append([self.positions_np[i], self.positions_np[j]])
            
            if lines:
                lc = Line3DCollection(lines, colors='gray', alpha=0.2, linewidths=0.5)
                ax.add_collection(lc)
        
        # Plot origin points
        if show_origins:
            for origin in self.origin_points:
                ax.scatter(*origin.position, c='red', s=100, marker='^', label='Origin')
                # Draw direction arrow
                end = origin.position + origin.direction * self.hex_spacing
                ax.quiver(
                    origin.position[0], origin.position[1], origin.position[2],
                    origin.direction[0], origin.direction[1], origin.direction[2],
                    color='red', alpha=0.8, arrow_length_ratio=0.3
                )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Layer)')
        ax.set_title('3D Hexagonal Lattice')
        
        return ax
    
    def plot_pulse_path(
        self,
        pulse_path: List[PulseRecord],
        ax: Optional[Axes3D] = None,
        color: str = 'blue',
        show_directions: bool = True,
        show_power: bool = True
    ) -> Axes3D:
        """
        Plot a pulse propagation path.
        
        Args:
            pulse_path: List of PulseRecord objects
            ax: Matplotlib 3D axes
            color: Color for the path
            show_directions: Whether to show direction arrows
            show_power: Whether to scale by power
            
        Returns:
            Matplotlib 3D axes
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        if not pulse_path:
            return ax
        
        # Plot path line
        positions = np.array([p.position for p in pulse_path])
        powers = np.array([p.power for p in pulse_path])
        
        # Color gradient based on time
        for i in range(len(positions) - 1):
            alpha = 0.3 + 0.7 * (i / len(positions))
            linewidth = 1 + 3 * powers[i] if show_power else 2
            ax.plot(
                positions[i:i+2, 0],
                positions[i:i+2, 1],
                positions[i:i+2, 2],
                color=color,
                alpha=alpha,
                linewidth=linewidth
            )
        
        # Plot direction arrows
        if show_directions:
            for record in pulse_path:
                scale = 0.3 * (record.power if show_power else 1)
                ax.quiver(
                    record.position[0], record.position[1], record.position[2],
                    record.direction[0] * scale,
                    record.direction[1] * scale,
                    record.direction[2] * scale,
                    color=color,
                    alpha=0.7,
                    arrow_length_ratio=0.3
                )
        
        # Mark start and end
        ax.scatter(*positions[0], c='green', s=100, marker='o', label='Start')
        ax.scatter(*positions[-1], c='red', s=100, marker='x', label='End')
        
        return ax
    
    def plot_pulse_history(
        self,
        ax: Optional[Axes3D] = None,
        cmap: str = 'viridis'
    ) -> Axes3D:
        """
        Plot all recorded pulses colored by time.
        
        Args:
            ax: Matplotlib 3D axes
            cmap: Colormap name
            
        Returns:
            Matplotlib 3D axes
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        if not self.pulse_history:
            return ax
        
        # Get time range
        times = np.array([p.time_step for p in self.pulse_history])
        positions = np.array([p.position for p in self.pulse_history])
        powers = np.array([p.power for p in self.pulse_history])
        
        # Normalize times for colormap
        norm_times = (times - times.min()) / (times.max() - times.min() + 1e-8)
        
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=norm_times,
            cmap=cmap,
            s=50 * powers,
            alpha=0.7
        )
        
        plt.colorbar(scatter, ax=ax, label='Time (normalized)')
        ax.set_title('Pulse History')
        
        return ax
    
    def visualize_full(
        self,
        pulse_paths: List[List[PulseRecord]] = None,
        figsize: Tuple[int, int] = (16, 6)
    ):
        """
        Create a comprehensive visualization with lattice and pulses.
        
        Args:
            pulse_paths: Optional list of pulse paths to display
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        
        # Lattice structure
        ax1 = fig.add_subplot(131, projection='3d')
        self.plot_lattice(ax1)
        ax1.set_title('Lattice Structure')
        
        # Pulse paths
        ax2 = fig.add_subplot(132, projection='3d')
        self.plot_lattice(ax2, show_connections=False, alpha=0.3)
        
        if pulse_paths:
            colors = ['blue', 'green', 'purple']
            for i, path in enumerate(pulse_paths):
                self.plot_pulse_path(path, ax2, color=colors[i % len(colors)])
        ax2.set_title('Pulse Propagation')
        
        # Pulse history heatmap
        ax3 = fig.add_subplot(133, projection='3d')
        self.plot_lattice(ax3, show_connections=False, alpha=0.2, show_origins=False)
        self.plot_pulse_history(ax3)
        ax3.set_title('Activation History')
        
        plt.tight_layout()
        return fig
    
    # ==================== Plotly Visualization Methods ====================
    
    def plot_lattice_plotly(
        self,
        show_cells: bool = True,
        show_connections: bool = True,
        show_origins: bool = True,
        show_pulse_paths: bool = True,
        pulse_paths: List[List[PulseRecord]] = None,
        cell_size: float = 8,
        title: str = "3D Hexagonal Lattice"
    ) -> go.Figure:
        """
        Plot the 3D lattice structure using Plotly.
        
        Args:
            show_cells: Whether to show cell positions
            show_connections: Whether to show adjacency connections
            show_origins: Whether to show origin points
            show_pulse_paths: Whether to show pulse ray paths
            pulse_paths: List of pulse paths to display
            cell_size: Size of cell markers
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Plot cells
        if show_cells:
            colors = ['orange' if i in self.active_cells else 'lightblue' 
                     for i in range(len(self.cells))]
            
            # Gather cell data for hover info
            cell_data = []
            for cell in self.cells:
                cell_data.append([
                    cell.get_straight_probability(),
                    cell.get_split_prob(),
                    cell.get_join_prob(),
                    cell.bounce_angles[0].item(),
                    cell.bounce_angles[1].item(),
                    cell.bounce_angles[2].item()
                ])
            cell_data = np.array(cell_data)
            
            fig.add_trace(go.Scatter3d(
                x=self.positions_np[:, 0],
                y=self.positions_np[:, 1],
                z=self.positions_np[:, 2],
                mode='markers',
                marker=dict(
                    size=cell_size,
                    color=colors,
                    line=dict(width=1, color='black'),
                    opacity=0.8
                ),
                name='Cells',
                hovertemplate=(
                    'Cell %{text}<br>'
                    'X: %{x:.2f}, Y: %{y:.2f}, Z: %{z:.2f}<br>'
                    'P(straight): %{customdata[0]:.1%}<br>'
                    'P(split): %{customdata[1]:.1%}<br>'
                    'P(join): %{customdata[2]:.1%}<br>'
                    'Bounce: [%{customdata[3]:.2f}, %{customdata[4]:.2f}, %{customdata[5]:.2f}] rad'
                ),
                text=[str(i) for i in range(len(self.cells))],
                customdata=cell_data
            ))
        
        # Plot connections
        if show_connections:
            edge_x, edge_y, edge_z = [], [], []
            for i, neighbors in self.adjacency.items():
                for j in neighbors:
                    if i < j:  # Avoid duplicates
                        edge_x.extend([self.positions_np[i, 0], self.positions_np[j, 0], None])
                        edge_y.extend([self.positions_np[i, 1], self.positions_np[j, 1], None])
                        edge_z.extend([self.positions_np[i, 2], self.positions_np[j, 2], None])
            
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='gray', width=1),
                opacity=0.3,
                name='Connections',
                hoverinfo='skip'
            ))
        
        # Plot pulse paths as rays
        if show_pulse_paths and pulse_paths:
            ray_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            
            for idx, path in enumerate(pulse_paths):
                if not path:
                    continue
                
                color = ray_colors[idx % len(ray_colors)]
                positions = np.array([p.position for p in path])
                powers = np.array([p.power for p in path])
                went_straight = [p.went_straight for p in path]
                split_occurred = [p.split_occurred for p in path]
                joined = [p.joined for p in path]
                split_weights = [p.split_weight for p in path]
                join_factors = [p.join_factor for p in path]
                
                # Determine marker colors based on state:
                # Normal (straight) = original color
                # Bounced = yellow
                # Split = orange
                # Joined = cyan
                marker_colors = []
                for ws, sp, jn in zip(went_straight, split_occurred, joined):
                    if jn:
                        marker_colors.append('cyan')
                    elif sp:
                        marker_colors.append('orange')
                    elif not ws:
                        marker_colors.append('yellow')
                    else:
                        marker_colors.append(color)
                
                # Draw ray line (small, subtle)
                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='lines+markers',
                    line=dict(color=color, width=1),
                    marker=dict(
                        size=3,
                        color=marker_colors,
                        symbol=['circle' if ws else 'diamond' for ws in went_straight],
                        opacity=0.7
                    ),
                    name=f'Ray {idx + 1}',
                    hovertemplate=(
                        'Step %{text}<br>'
                        'Power: %{customdata[0]:.2f}<br>'
                        'Straight: %{customdata[1]}<br>'
                        'Split: %{customdata[2]} (weight: %{customdata[3]:.2f})<br>'
                        'Joined: %{customdata[4]} (factor: %{customdata[5]:.2f})'
                    ),
                    text=[str(i) for i in range(len(path))],
                    customdata=np.column_stack([
                        powers, went_straight, split_occurred, 
                        split_weights, joined, join_factors
                    ])
                ))
                
                # Start marker (green, small)
                fig.add_trace(go.Scatter3d(
                    x=[positions[0, 0]],
                    y=[positions[0, 1]],
                    z=[positions[0, 2]],
                    mode='markers',
                    marker=dict(size=4, color='lime', symbol='circle', 
                               line=dict(width=1, color='darkgreen')),
                    name=f'Start {idx + 1}',
                    showlegend=False,
                    hovertemplate='Start Ray %{text}' + f' {idx + 1}',
                    text=['']
                ))
                
                # End marker (red, small)
                fig.add_trace(go.Scatter3d(
                    x=[positions[-1, 0]],
                    y=[positions[-1, 1]],
                    z=[positions[-1, 2]],
                    mode='markers',
                    marker=dict(size=4, color='red', symbol='x',
                               line=dict(width=1, color='darkred')),
                    name=f'End {idx + 1}',
                    showlegend=False,
                    hovertemplate='End Ray %{text}' + f' {idx + 1}',
                    text=['']
                ))
        
        # Plot origin points and direction arrows (small)
        if show_origins and self.origin_points:
            origin_positions = np.array([o.position for o in self.origin_points])
            
            fig.add_trace(go.Scatter3d(
                x=origin_positions[:, 0],
                y=origin_positions[:, 1],
                z=origin_positions[:, 2],
                mode='markers',
                marker=dict(size=5, color='red', symbol='diamond'),
                name='Origin'
            ))
            
            # Direction arrows using cones (small)
            for origin in self.origin_points:
                fig.add_trace(go.Cone(
                    x=[origin.position[0]], 
                    y=[origin.position[1]], 
                    z=[origin.position[2]],
                    u=[origin.direction[0]], 
                    v=[origin.direction[1]], 
                    w=[origin.direction[2]],
                    sizemode='absolute',
                    sizeref=0.15,
                    colorscale=[[0, 'red'], [1, 'red']],
                    showscale=False,
                    name='Direction'
                ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z (Layer)',
                aspectmode='data'
            ),
            showlegend=True,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def plot_pulse_paths_plotly(
        self,
        pulse_paths: List[List[PulseRecord]] = None,
        show_lattice: bool = True,
        show_directions: bool = True,
        title: str = "Pulse Propagation"
    ) -> go.Figure:
        """
        Plot pulse propagation paths using Plotly.
        
        Args:
            pulse_paths: List of pulse paths to display
            show_lattice: Whether to show the lattice cells
            show_directions: Whether to show direction cones
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Show lattice cells (faded)
        if show_lattice:
            fig.add_trace(go.Scatter3d(
                x=self.positions_np[:, 0],
                y=self.positions_np[:, 1],
                z=self.positions_np[:, 2],
                mode='markers',
                marker=dict(size=5, color='lightgray', opacity=0.4),
                name='Lattice',
                hoverinfo='skip'
            ))
        
        # Plot each pulse path
        colors = px.colors.qualitative.Plotly
        
        if pulse_paths:
            for idx, path in enumerate(pulse_paths):
                if not path:
                    continue
                
                color = colors[idx % len(colors)]
                positions = np.array([p.position for p in path])
                powers = np.array([p.power for p in path])
                
                # Path line (small, scaled with cells)
                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='lines+markers',
                    line=dict(color=color, width=1),
                    marker=dict(
                        size=3,
                        color=color,
                        opacity=0.7
                    ),
                    name=f'Pulse {idx + 1}',
                    hovertemplate='Step %{text}<br>Power: %{customdata:.2f}',
                    text=[str(i) for i in range(len(path))],
                    customdata=powers
                ))
                
                # Direction cones (small)
                if show_directions:
                    for record in path:
                        fig.add_trace(go.Cone(
                            x=[record.position[0]],
                            y=[record.position[1]],
                            z=[record.position[2]],
                            u=[record.direction[0] * 0.15],
                            v=[record.direction[1] * 0.15],
                            w=[record.direction[2] * 0.15],
                            sizemode='absolute',
                            sizeref=0.08,
                            colorscale=[[0, color], [1, color]],
                            showscale=False,
                            opacity=0.5,
                            hoverinfo='skip'
                        ))
                
                # Start marker (small)
                fig.add_trace(go.Scatter3d(
                    x=[positions[0, 0]],
                    y=[positions[0, 1]],
                    z=[positions[0, 2]],
                    mode='markers',
                    marker=dict(size=4, color='green', symbol='circle'),
                    name=f'Start {idx + 1}',
                    showlegend=False
                ))
                
                # End marker (small)
                fig.add_trace(go.Scatter3d(
                    x=[positions[-1, 0]],
                    y=[positions[-1, 1]],
                    z=[positions[-1, 2]],
                    mode='markers',
                    marker=dict(size=4, color='red', symbol='x'),
                    name=f'End {idx + 1}',
                    showlegend=False
                ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z (Layer)',
                aspectmode='data'
            ),
            showlegend=True,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def plot_pulse_history_plotly(
        self,
        colorscale: str = 'Viridis',
        title: str = "Pulse History"
    ) -> go.Figure:
        """
        Plot all recorded pulses colored by time using Plotly.
        
        Args:
            colorscale: Plotly colorscale name
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Show lattice cells (faded)
        fig.add_trace(go.Scatter3d(
            x=self.positions_np[:, 0],
            y=self.positions_np[:, 1],
            z=self.positions_np[:, 2],
            mode='markers',
            marker=dict(size=4, color='lightgray', opacity=0.3),
            name='Lattice',
            hoverinfo='skip'
        ))
        
        if self.pulse_history:
            times = np.array([p.time_step for p in self.pulse_history])
            positions = np.array([p.position for p in self.pulse_history])
            powers = np.array([p.power for p in self.pulse_history])
            bounce_angles = np.array([p.bounce_angles for p in self.pulse_history])  # 3-axis
            modes = np.array([p.alternate_mode for p in self.pulse_history])
            
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=times,
                    colorscale=colorscale,
                    colorbar=dict(title='Time Step'),
                    opacity=0.7
                ),
                name='Activations',
                hovertemplate=(
                    'Time: %{marker.color}<br>'
                    'Power: %{customdata[0]:.2f}<br>'
                    'X: %{customdata[1]:.1f}° Y: %{customdata[2]:.1f}° Z: %{customdata[3]:.1f}°'
                ),
                customdata=np.column_stack([powers, bounce_angles])
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z (Layer)',
                aspectmode='data'
            ),
            showlegend=True,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def visualize_full_plotly(
        self,
        pulse_paths: List[List[PulseRecord]] = None
    ) -> go.Figure:
        """
        Create comprehensive interactive 3D visualization using Plotly.
        
        Args:
            pulse_paths: Optional list of pulse paths to display
            
        Returns:
            Plotly Figure with dropdown to switch views
        """
        # Create all three visualizations - include pulse paths in lattice view
        fig_lattice = self.plot_lattice_plotly(
            pulse_paths=pulse_paths, 
            show_pulse_paths=True,
            title="3D Hexagonal Lattice with Ray Paths"
        )
        fig_pulses = self.plot_pulse_paths_plotly(pulse_paths)
        fig_history = self.plot_pulse_history_plotly()
        
        # Combine into single figure with dropdown
        fig = go.Figure()
        
        # Add all traces from lattice view (visible by default)
        for trace in fig_lattice.data:
            trace.visible = True
            fig.add_trace(trace)
        n_lattice = len(fig_lattice.data)
        
        # Add traces from pulse view (hidden)
        for trace in fig_pulses.data:
            trace.visible = False
            fig.add_trace(trace)
        n_pulses = len(fig_pulses.data)
        
        # Add traces from history view (hidden)
        for trace in fig_history.data:
            trace.visible = False
            fig.add_trace(trace)
        n_history = len(fig_history.data)
        
        total_traces = n_lattice + n_pulses + n_history
        
        # Create dropdown buttons
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=[
                        dict(
                            label="Lattice + Rays",
                            method="update",
                            args=[
                                {"visible": [i < n_lattice for i in range(total_traces)]},
                                {"title": "3D Hexagonal Lattice with Ray Paths"}
                            ]
                        ),
                        dict(
                            label="Pulse Propagation",
                            method="update",
                            args=[
                                {"visible": [n_lattice <= i < n_lattice + n_pulses for i in range(total_traces)]},
                                {"title": "Pulse Propagation"}
                            ]
                        ),
                        dict(
                            label="Activation History",
                            method="update",
                            args=[
                                {"visible": [i >= n_lattice + n_pulses for i in range(total_traces)]},
                                {"title": "Pulse History"}
                            ]
                        ),
                    ],
                    direction="down",
                    showactive=True,
                    x=0.1,
                    y=1.15
                )
            ],
            title="3D Hexagonal Lattice",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z (Layer)',
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        return fig
    
    def reset_pulses(self):
        """Clear pulse history and active cells."""
        self.pulse_history.clear()
        self.active_cells.clear()
        self.current_time = 0
    
    def get_toroidal_pattern(self) -> np.ndarray:
        """
        Analyze pulse history for toroidal wave patterns (Section 2.1).
        
        Returns:
            Array of activation frequencies per cell
        """
        activation_counts = np.zeros(len(self.cells))
        
        for record in self.pulse_history:
            activation_counts[record.cell_idx] += record.power
        
        return activation_counts
    
    def to(self, device: Union[str, torch.device]) -> 'LatticeVisualizer':
        """
        Move lattice to specified device.
        
        Args:
            device: Target device ('mps', 'cuda', 'cpu')
            
        Returns:
            Self for chaining
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.positions = self.positions.to(device)
        
        for cell in self.cells:
            cell.to(device)
        
        print(f"Moved lattice to device: {device}")
        return self
    
    def cpu(self) -> 'LatticeVisualizer':
        """Move lattice to CPU."""
        return self.to(torch.device("cpu"))
    
    def mps(self) -> 'LatticeVisualizer':
        """Move lattice to MPS (Apple Silicon GPU)."""
        return self.to(torch.device("mps"))
    
    def __repr__(self) -> str:
        return (f"LatticeVisualizer(layers={self.layers}, hex_radius={self.hex_radius}, "
                f"n_cells={len(self.cells)}, n_pulses={len(self.pulse_history)}, "
                f"device={self.device})")


class MazeLatticeVisualizer(LatticeVisualizer):
    """
    A lattice visualizer that initializes cell states using Kruskal's algorithm
    to create maze-like patterns in the X and Y bounce angle axes.
    
    Higher bounce angle values represent "walls" (cells that deflect pulses more)
    Lower bounce angle values represent "pathways" (cells that let pulses through)
    
    The Z axis bounce angle remains random.
    
    All angles are in RADIANS (max 2 radians).
    """
    
    def __init__(
        self,
        layers: int = 3,
        hex_radius: int = 3,
        layer_spacing: float = 1.0,
        hex_spacing: float = 0.9,
        device: Union[str, torch.device] = None,
        wall_angle: float = 1.5,
        pathway_angle: float = 0.2,
        seed: Optional[int] = None
    ):
        """
        Initialize the maze lattice visualizer.
        
        Args:
            layers: Number of vertical layers in the lattice
            hex_radius: Radius of hexagonal grid per layer (in cells)
            layer_spacing: Vertical distance between layers
            hex_spacing: Horizontal distance between adjacent cells
            device: Torch device ('mps', 'cuda', 'cpu')
            wall_angle: Bounce angle for "wall" cells in radians (high deflection, default 1.5)
            pathway_angle: Bounce angle for "pathway" cells in radians (low deflection, default 0.2)
            seed: Random seed for reproducible maze generation
        """
        self.wall_angle = wall_angle
        self.pathway_angle = pathway_angle
        self.seed = seed
        self.maze_edges = set()  # Stores (i, j) tuples for MST edges (pathways)
        self.wall_cells = set()  # Cells that are primarily walls
        self.pathway_cells = set()  # Cells that are primarily pathways
        
        # Call parent init (which will call _build_hexagonal_lattice)
        super().__init__(
            layers=layers,
            hex_radius=hex_radius,
            layer_spacing=layer_spacing,
            hex_spacing=hex_spacing,
            device=device
        )
    
    def _build_hexagonal_lattice(self):
        """
        Build the 3D hexagonal lattice structure with maze-initialized states.
        Uses Kruskal's algorithm on X and Y axes for bounce angles.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        positions = []
        cell_idx = 0
        
        # First pass: build positions and basic cell structure
        for layer in range(self.layers):
            for q in range(-self.hex_radius, self.hex_radius + 1):
                r1 = max(-self.hex_radius, -q - self.hex_radius)
                r2 = min(self.hex_radius, -q + self.hex_radius)
                for r in range(r1, r2 + 1):
                    pos = self._hex_to_cartesian(q, r, layer)
                    positions.append(pos)
                    cell_idx += 1
        
        # Stack positions into a single tensor
        self.positions = torch.stack(positions)
        self.positions_np = self.positions.cpu().numpy()
        
        n_cells = len(positions)
        print(f"Building maze lattice with {n_cells} cells...")
        
        # Compute adjacency for the maze (within same layer for X-Y maze)
        layer_adjacency = self._compute_layer_adjacency()
        
        # Run Kruskal's algorithm to generate maze spanning tree
        self._generate_maze_kruskal(layer_adjacency, n_cells)
        
        # Classify cells based on how many pathway edges they connect to
        cell_pathway_count = np.zeros(n_cells)
        for (i, j) in self.maze_edges:
            cell_pathway_count[i] += 1
            cell_pathway_count[j] += 1
        
        # Cells with more pathway connections are pathway cells
        # Cells with fewer connections are wall cells
        max_connections = max(cell_pathway_count.max(), 1)
        
        # Second pass: create cells with maze-based bounce angles
        self.cells = []
        for idx in range(n_cells):
            # Alternate between 'closer' and 'further' modes
            alternate_mode = 'closer' if idx % 2 == 0 else 'further'
            
            # Randomized mean (weight)
            random_mean = (torch.rand(1, device=self.device) * 2 - 1).item()
            
            # std_dev influenced by alternate_mode
            if alternate_mode == 'closer':
                random_std = (torch.rand(1, device=self.device) * 0.2 + 0.5).item()
            else:
                random_std = (torch.rand(1, device=self.device) * 0.3 + 0.7).item()
            
            # Compute bounce angles based on maze structure (in RADIANS)
            # Pathway ratio: 0 = wall, 1 = pathway
            pathway_ratio = cell_pathway_count[idx] / max_connections
            
            # X and Y axes: interpolate between wall and pathway angles
            # Pathway cells have low bounce angles (pulses go straight)
            # Wall cells have high bounce angles (pulses deflect)
            x_angle = self.wall_angle * (1 - pathway_ratio) + self.pathway_angle * pathway_ratio
            y_angle = self.wall_angle * (1 - pathway_ratio) + self.pathway_angle * pathway_ratio
            
            # Add some noise to make it more natural (±0.1 radians noise)
            noise = torch.rand(2, device=self.device) * 0.2 - 0.1
            x_angle = max(0, min(2.0, x_angle + noise[0].item()))
            y_angle = max(0, min(2.0, y_angle + noise[1].item()))
            
            # Z axis: random (not part of maze) - max 2 radians
            z_angle = torch.rand(1, device=self.device).item() * 2.0
            
            bounce_angles = torch.tensor([x_angle, y_angle, z_angle], device=self.device)
            
            # Track cell type
            if pathway_ratio > 0.5:
                self.pathway_cells.add(idx)
            else:
                self.wall_cells.add(idx)
            
            # Create cell
            cell = Cell(
                mean=random_mean,
                std_dev=random_std,
                bounce_angles=bounce_angles,
                state=idx,
                device=self.device,
                alternate_mode=alternate_mode
            )
            self.cells.append(cell)
        
        print(f"Maze generated: {len(self.pathway_cells)} pathway cells, {len(self.wall_cells)} wall cells")
        print(f"MST edges (pathways): {len(self.maze_edges)}")
    
    def _compute_layer_adjacency(self) -> List[Tuple[int, int, float]]:
        """
        Compute adjacency within layers for maze generation.
        Returns list of (cell_i, cell_j, random_weight) for Kruskal's.
        """
        n_cells = len(self.positions)
        edges = []
        
        # Compute pairwise distances
        diff = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)
        distances = torch.norm(diff, dim=2)
        
        # In axial hex coordinates, adjacent cells are sqrt(3) * hex_spacing apart
        hex_adjacent_dist = self.hex_spacing * np.sqrt(3) * 1.1
        
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                dist = distances[i, j].item()
                # Same layer neighbors (hex adjacency) - horizontal connections
                if dist < hex_adjacent_dist:
                    # Random weight for Kruskal's (gives random maze)
                    weight = np.random.random()
                    edges.append((i, j, weight))
        
        return edges
    
    def _generate_maze_kruskal(self, edges: List[Tuple[int, int, float]], n_cells: int):
        """
        Generate maze using Kruskal's algorithm (minimum spanning tree).
        
        The MST edges become "pathways" - cells connected by these edges
        will have lower bounce angles.
        """
        # Sort edges by random weight
        edges.sort(key=lambda x: x[2])
        
        # Union-Find for Kruskal's
        uf = UnionFind(n_cells)
        
        self.maze_edges = set()
        
        for i, j, weight in edges:
            # If these cells aren't already connected, add edge to MST
            if uf.union(i, j):
                self.maze_edges.add((min(i, j), max(i, j)))
    
    def get_maze_visualization_data(self) -> Dict:
        """
        Get data for visualizing the maze structure.
        
        Returns:
            Dictionary with pathway/wall cell indices and edge data
        """
        return {
            'pathway_cells': list(self.pathway_cells),
            'wall_cells': list(self.wall_cells),
            'maze_edges': list(self.maze_edges),
            'n_pathway_cells': len(self.pathway_cells),
            'n_wall_cells': len(self.wall_cells),
            'wall_angle': self.wall_angle,
            'pathway_angle': self.pathway_angle
        }
    
    def plot_maze_structure_plotly(
        self,
        show_edges: bool = True,
        title: str = "Maze Lattice Structure"
    ) -> go.Figure:
        """
        Plot the maze structure with pathway/wall cells colored differently.
        
        Args:
            show_edges: Whether to show MST edges (pathways)
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Get bounce angles for color mapping
        x_angles = np.array([self.cells[i].bounce_angles[0].item() for i in range(len(self.cells))])
        y_angles = np.array([self.cells[i].bounce_angles[1].item() for i in range(len(self.cells))])
        avg_xy_angles = (x_angles + y_angles) / 2
        
        # Color cells by their X-Y average bounce angle (pathway vs wall)
        # Lower angles = pathway (blue), Higher angles = wall (red)
        fig.add_trace(go.Scatter3d(
            x=self.positions_np[:, 0],
            y=self.positions_np[:, 1],
            z=self.positions_np[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                color=avg_xy_angles,
                colorscale='RdYlBu_r',  # Red=high (wall), Blue=low (pathway)
                colorbar=dict(title='Bounce Angle<br>(X-Y avg, rad)'),
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            name='Cells',
            hovertemplate=(
                'Cell %{text}<br>'
                'X angle: %{customdata[0]:.3f} rad<br>'
                'Y angle: %{customdata[1]:.3f} rad<br>'
                'Type: %{customdata[2]}'
            ),
            text=[str(i) for i in range(len(self.cells))],
            customdata=np.column_stack([
                x_angles, 
                y_angles,
                ['Pathway' if i in self.pathway_cells else 'Wall' for i in range(len(self.cells))]
            ])
        ))
        
        # Plot MST edges (pathways) in green
        if show_edges:
            edge_x, edge_y, edge_z = [], [], []
            for (i, j) in self.maze_edges:
                edge_x.extend([self.positions_np[i, 0], self.positions_np[j, 0], None])
                edge_y.extend([self.positions_np[i, 1], self.positions_np[j, 1], None])
                edge_z.extend([self.positions_np[i, 2], self.positions_np[j, 2], None])
            
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='limegreen', width=3),
                opacity=0.7,
                name='Pathways (MST)',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z (Layer)',
                aspectmode='data'
            ),
            showlegend=True,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def plot_maze_2d_slice(
        self,
        layer: int = 0,
        figsize: Tuple[int, int] = (10, 10)
    ) -> plt.Figure:
        """
        Plot a 2D slice of the maze at a specific layer.
        
        Args:
            layer: Which layer to visualize (0-indexed)
            figsize: Figure size
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get cells in this layer
        cells_per_layer = len(self.cells) // self.layers
        layer_start = layer * cells_per_layer
        layer_end = (layer + 1) * cells_per_layer
        
        # Get X-Y angles for cells in this layer
        x_angles = np.array([
            self.cells[i].bounce_angles[0].item() 
            for i in range(layer_start, min(layer_end, len(self.cells)))
        ])
        y_angles = np.array([
            self.cells[i].bounce_angles[1].item() 
            for i in range(layer_start, min(layer_end, len(self.cells)))
        ])
        avg_angles = (x_angles + y_angles) / 2
        
        # Plot cells
        positions_layer = self.positions_np[layer_start:layer_end]
        scatter = ax.scatter(
            positions_layer[:, 0],
            positions_layer[:, 1],
            c=avg_angles,
            cmap='RdYlBu_r',
            s=200,
            edgecolors='black',
            linewidths=1
        )
        plt.colorbar(scatter, ax=ax, label='Bounce Angle (X-Y avg, rad)')
        
        # Plot MST edges for this layer
        for (i, j) in self.maze_edges:
            if layer_start <= i < layer_end and layer_start <= j < layer_end:
                local_i = i - layer_start
                local_j = j - layer_start
                ax.plot(
                    [positions_layer[local_i, 0], positions_layer[local_j, 0]],
                    [positions_layer[local_i, 1], positions_layer[local_j, 1]],
                    'g-', linewidth=2, alpha=0.7
                )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Maze Structure - Layer {layer}\n(Blue=Pathway, Red=Wall, angles in radians)')
        ax.set_aspect('equal')
        
        return fig
    
    def __repr__(self) -> str:
        return (f"MazeLatticeVisualizer(layers={self.layers}, hex_radius={self.hex_radius}, "
                f"n_cells={len(self.cells)}, pathways={len(self.pathway_cells)}, "
                f"walls={len(self.wall_cells)}, device={self.device})")
