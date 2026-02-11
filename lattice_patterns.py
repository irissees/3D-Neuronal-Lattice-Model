"""
Lattice Pattern Generators: Programmable 3D shapes and patterns for cell weight initialization.

Provides geometric primitives, mathematical patterns, and composition tools
for setting cell weights in 3D lattice structures.

Usage:
    from lattice_patterns import PatternGenerator, Patterns
    
    # Create pattern for a 32x32x32 lattice
    pg = PatternGenerator(shape=(32, 32, 32))
    
    # Single pattern
    sphere = pg.sphere(center=(16, 16, 16), radius=10, falloff='smooth')
    
    # Composite patterns
    pattern = pg.compose([
        ('sphere', {'center': (16, 16, 16), 'radius': 8}, 1.0),
        ('gradient', {'axis': 'z', 'direction': 'ascending'}, 0.5),
        ('noise', {'scale': 0.1, 'octaves': 3}, 0.2)
    ])
    
    # Apply to lattice
    lattice.set_pattern(pattern)
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional, Union, Callable, Literal
from dataclasses import dataclass
from enum import Enum


class FalloffType(Enum):
    """Falloff functions for shape boundaries."""
    HARD = 'hard'           # Binary: 1 inside, 0 outside
    LINEAR = 'linear'       # Linear falloff
    SMOOTH = 'smooth'       # Smoothstep falloff
    GAUSSIAN = 'gaussian'   # Gaussian falloff
    EXPONENTIAL = 'exp'     # Exponential falloff


@dataclass
class Pattern3D:
    """Container for a 3D pattern."""
    data: np.ndarray
    name: str
    params: Dict
    
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        return torch.tensor(self.data, dtype=torch.float32, device=device)
    
    def __add__(self, other: 'Pattern3D') -> 'Pattern3D':
        return Pattern3D(
            data=self.data + other.data,
            name=f"({self.name} + {other.name})",
            params={'op': 'add', 'patterns': [self.params, other.params]}
        )
    
    def __mul__(self, other: Union['Pattern3D', float]) -> 'Pattern3D':
        if isinstance(other, Pattern3D):
            return Pattern3D(
                data=self.data * other.data,
                name=f"({self.name} * {other.name})",
                params={'op': 'mul', 'patterns': [self.params, other.params]}
            )
        return Pattern3D(
            data=self.data * other,
            name=f"({self.name} * {other})",
            params={'op': 'scale', 'factor': other, 'pattern': self.params}
        )
    
    def __rmul__(self, other: float) -> 'Pattern3D':
        return self.__mul__(other)
    
    def normalize(self, min_val: float = 0.0, max_val: float = 1.0) -> 'Pattern3D':
        """Normalize pattern to range [min_val, max_val]."""
        data = self.data.copy()
        data_min, data_max = data.min(), data.max()
        if data_max - data_min > 1e-8:
            data = (data - data_min) / (data_max - data_min)
            data = data * (max_val - min_val) + min_val
        return Pattern3D(data=data, name=f"norm({self.name})", params=self.params)
    
    def clip(self, min_val: float = 0.0, max_val: float = 1.0) -> 'Pattern3D':
        """Clip pattern values to range."""
        return Pattern3D(
            data=np.clip(self.data, min_val, max_val),
            name=f"clip({self.name})",
            params=self.params
        )
    
    def invert(self) -> 'Pattern3D':
        """Invert pattern (1 - x for normalized patterns)."""
        return Pattern3D(
            data=1.0 - self.data,
            name=f"invert({self.name})",
            params=self.params
        )
    
    def threshold(self, value: float, above: float = 1.0, below: float = 0.0) -> 'Pattern3D':
        """Apply threshold to pattern."""
        data = np.where(self.data >= value, above, below)
        return Pattern3D(data=data, name=f"thresh({self.name})", params=self.params)


class PatternGenerator:
    """
    Generator for 3D patterns and shapes.
    
    All patterns are generated as numpy arrays matching the lattice shape.
    """
    
    def __init__(self, shape: Tuple[int, int, int]):
        """
        Initialize pattern generator.
        
        Args:
            shape: (D1, D2, D3) shape of the target lattice
        """
        self.shape = shape
        self.d1, self.d2, self.d3 = shape
        
        # Pre-compute coordinate grids (normalized 0-1)
        self._build_coordinate_grids()
    
    def _build_coordinate_grids(self):
        """Build normalized coordinate grids."""
        # Integer coordinates
        i = np.arange(self.d1)
        j = np.arange(self.d2)
        k = np.arange(self.d3)
        self.I, self.J, self.K = np.meshgrid(i, j, k, indexing='ij')
        
        # Normalized coordinates [0, 1]
        self.X = self.I / max(1, self.d1 - 1)
        self.Y = self.J / max(1, self.d2 - 1)
        self.Z = self.K / max(1, self.d3 - 1)
        
        # Centered coordinates [-1, 1]
        self.Xc = 2 * self.X - 1
        self.Yc = 2 * self.Y - 1
        self.Zc = 2 * self.Z - 1
    
    def _apply_falloff(self, distance: np.ndarray, radius: float, 
                       falloff: Union[str, FalloffType], falloff_width: float = 0.2) -> np.ndarray:
        """Apply falloff function based on distance from boundary."""
        if isinstance(falloff, str):
            falloff = FalloffType(falloff)
        
        if falloff == FalloffType.HARD:
            return (distance <= radius).astype(float)
        
        # Normalized distance (0 at center, 1 at radius)
        t = distance / radius
        
        if falloff == FalloffType.LINEAR:
            return np.clip(1 - t, 0, 1)
        
        elif falloff == FalloffType.SMOOTH:
            # Smoothstep
            t = np.clip(t, 0, 1)
            return 1 - (3 * t**2 - 2 * t**3)
        
        elif falloff == FalloffType.GAUSSIAN:
            sigma = radius * falloff_width
            return np.exp(-0.5 * (distance / sigma) ** 2)
        
        elif falloff == FalloffType.EXPONENTIAL:
            return np.exp(-t / falloff_width)
        
        return (distance <= radius).astype(float)
    
    # ========================================================================
    # Primitive Shapes
    # ========================================================================
    
    def sphere(
        self,
        center: Tuple[float, float, float] = None,
        radius: float = None,
        falloff: str = 'smooth',
        falloff_width: float = 0.3,
        value: float = 1.0
    ) -> Pattern3D:
        """
        Generate a spherical pattern.
        
        Args:
            center: (x, y, z) center in grid coordinates. Default: lattice center
            radius: Sphere radius in grid units. Default: min(shape)/4
            falloff: 'hard', 'linear', 'smooth', 'gaussian', 'exp'
            value: Value inside the sphere
        """
        if center is None:
            center = (self.d1 / 2, self.d2 / 2, self.d3 / 2)
        if radius is None:
            radius = min(self.shape) / 4
        
        # Distance from center
        dist = np.sqrt(
            (self.I - center[0])**2 + 
            (self.J - center[1])**2 + 
            (self.K - center[2])**2
        )
        
        data = self._apply_falloff(dist, radius, falloff, falloff_width) * value
        
        return Pattern3D(
            data=data,
            name='sphere',
            params={'center': center, 'radius': radius, 'falloff': falloff}
        )
    
    def cube(
        self,
        center: Tuple[float, float, float] = None,
        size: Union[float, Tuple[float, float, float]] = None,
        falloff: str = 'hard',
        falloff_width: float = 0.2,
        value: float = 1.0
    ) -> Pattern3D:
        """Generate a cubic/box pattern."""
        if center is None:
            center = (self.d1 / 2, self.d2 / 2, self.d3 / 2)
        if size is None:
            size = min(self.shape) / 4
        if isinstance(size, (int, float)):
            size = (size, size, size)
        
        # Chebyshev distance (infinity norm)
        dist = np.maximum.reduce([
            np.abs(self.I - center[0]) / (size[0] / 2),
            np.abs(self.J - center[1]) / (size[1] / 2),
            np.abs(self.K - center[2]) / (size[2] / 2)
        ])
        
        data = self._apply_falloff(dist, 1.0, falloff, falloff_width) * value
        
        return Pattern3D(
            data=data,
            name='cube',
            params={'center': center, 'size': size, 'falloff': falloff}
        )
    
    def cylinder(
        self,
        center: Tuple[float, float] = None,
        radius: float = None,
        axis: str = 'z',
        z_range: Tuple[float, float] = None,
        falloff: str = 'smooth',
        value: float = 1.0
    ) -> Pattern3D:
        """Generate a cylindrical pattern along specified axis."""
        if center is None:
            if axis == 'x':
                center = (self.d2 / 2, self.d3 / 2)
            elif axis == 'y':
                center = (self.d1 / 2, self.d3 / 2)
            else:  # z
                center = (self.d1 / 2, self.d2 / 2)
        
        if radius is None:
            if axis == 'x':
                radius = min(self.d2, self.d3) / 4
            elif axis == 'y':
                radius = min(self.d1, self.d3) / 4
            else:
                radius = min(self.d1, self.d2) / 4
        
        # 2D distance in perpendicular plane
        if axis == 'x':
            dist = np.sqrt((self.J - center[0])**2 + (self.K - center[1])**2)
        elif axis == 'y':
            dist = np.sqrt((self.I - center[0])**2 + (self.K - center[1])**2)
        else:  # z
            dist = np.sqrt((self.I - center[0])**2 + (self.J - center[1])**2)
        
        data = self._apply_falloff(dist, radius, falloff) * value
        
        # Apply z-range if specified
        if z_range is not None:
            if axis == 'x':
                mask = (self.I >= z_range[0]) & (self.I <= z_range[1])
            elif axis == 'y':
                mask = (self.J >= z_range[0]) & (self.J <= z_range[1])
            else:
                mask = (self.K >= z_range[0]) & (self.K <= z_range[1])
            data = data * mask
        
        return Pattern3D(data=data, name='cylinder', params={'axis': axis, 'radius': radius})
    
    def cone(
        self,
        apex: Tuple[float, float, float] = None,
        base_center: Tuple[float, float, float] = None,
        base_radius: float = None,
        falloff: str = 'smooth',
        value: float = 1.0
    ) -> Pattern3D:
        """Generate a conical pattern."""
        if apex is None:
            apex = (self.d1 / 2, self.d2 / 2, self.d3 - 1)
        if base_center is None:
            base_center = (self.d1 / 2, self.d2 / 2, 0)
        if base_radius is None:
            base_radius = min(self.d1, self.d2) / 3
        
        apex = np.array(apex)
        base_center = np.array(base_center)
        axis = base_center - apex
        height = np.linalg.norm(axis)
        axis_norm = axis / height
        
        # Vector from apex to each point
        points = np.stack([self.I, self.J, self.K], axis=-1)
        to_point = points - apex
        
        # Project onto axis
        proj_length = np.sum(to_point * axis_norm, axis=-1)
        proj_length = np.clip(proj_length, 0, height)
        
        # Radius at this height
        t = proj_length / height
        radius_at_height = base_radius * t
        
        # Distance from axis
        proj_point = apex + proj_length[..., np.newaxis] * axis_norm
        dist_from_axis = np.linalg.norm(points - proj_point, axis=-1)
        
        # Inside cone if distance < radius at height
        relative_dist = np.where(radius_at_height > 0, dist_from_axis / radius_at_height, 1.0)
        data = self._apply_falloff(relative_dist, 1.0, falloff) * value
        
        # Mask outside height range
        data = data * (proj_length >= 0) * (proj_length <= height)
        
        return Pattern3D(data=data, name='cone', params={'apex': tuple(apex), 'base_radius': base_radius})
    
    def torus(
        self,
        center: Tuple[float, float, float] = None,
        major_radius: float = None,
        minor_radius: float = None,
        axis: str = 'z',
        falloff: str = 'smooth',
        value: float = 1.0
    ) -> Pattern3D:
        """Generate a toroidal pattern."""
        if center is None:
            center = (self.d1 / 2, self.d2 / 2, self.d3 / 2)
        if major_radius is None:
            major_radius = min(self.d1, self.d2) / 3
        if minor_radius is None:
            minor_radius = major_radius / 3
        
        center = np.array(center)
        
        if axis == 'z':
            # Distance from center in xy plane
            dist_xy = np.sqrt((self.I - center[0])**2 + (self.J - center[1])**2)
            # Distance from tube center
            dist_torus = np.sqrt((dist_xy - major_radius)**2 + (self.K - center[2])**2)
        elif axis == 'y':
            dist_xz = np.sqrt((self.I - center[0])**2 + (self.K - center[2])**2)
            dist_torus = np.sqrt((dist_xz - major_radius)**2 + (self.J - center[1])**2)
        else:  # x
            dist_yz = np.sqrt((self.J - center[1])**2 + (self.K - center[2])**2)
            dist_torus = np.sqrt((dist_yz - major_radius)**2 + (self.I - center[0])**2)
        
        data = self._apply_falloff(dist_torus, minor_radius, falloff) * value
        
        return Pattern3D(
            data=data, name='torus',
            params={'major_radius': major_radius, 'minor_radius': minor_radius}
        )
    
    def plane(
        self,
        normal: Tuple[float, float, float] = (0, 0, 1),
        point: Tuple[float, float, float] = None,
        thickness: float = 1.0,
        falloff: str = 'smooth',
        value: float = 1.0
    ) -> Pattern3D:
        """Generate a planar slice pattern."""
        if point is None:
            point = (self.d1 / 2, self.d2 / 2, self.d3 / 2)
        
        normal = np.array(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        point = np.array(point)
        
        # Signed distance to plane
        dist = (
            normal[0] * (self.I - point[0]) +
            normal[1] * (self.J - point[1]) +
            normal[2] * (self.K - point[2])
        )
        
        data = self._apply_falloff(np.abs(dist), thickness / 2, falloff) * value
        
        return Pattern3D(data=data, name='plane', params={'normal': tuple(normal), 'thickness': thickness})
    
    # ========================================================================
    # Gradients
    # ========================================================================
    
    def gradient(
        self,
        axis: str = 'z',
        direction: str = 'ascending',
        start: float = 0.0,
        end: float = 1.0
    ) -> Pattern3D:
        """Generate a linear gradient along an axis."""
        if axis == 'x':
            t = self.X
        elif axis == 'y':
            t = self.Y
        else:  # z
            t = self.Z
        
        if direction == 'descending':
            t = 1 - t
        
        data = t * (end - start) + start
        
        return Pattern3D(data=data, name='gradient', params={'axis': axis, 'direction': direction})
    
    def radial_gradient(
        self,
        center: Tuple[float, float, float] = None,
        direction: str = 'outward',
        max_radius: float = None
    ) -> Pattern3D:
        """Generate a radial gradient from center."""
        if center is None:
            center = (self.d1 / 2, self.d2 / 2, self.d3 / 2)
        if max_radius is None:
            max_radius = np.sqrt(sum(d**2 for d in self.shape)) / 2
        
        dist = np.sqrt(
            (self.I - center[0])**2 +
            (self.J - center[1])**2 +
            (self.K - center[2])**2
        )
        
        t = np.clip(dist / max_radius, 0, 1)
        
        if direction == 'inward':
            t = 1 - t
        
        return Pattern3D(data=t, name='radial_gradient', params={'center': center})
    
    # ========================================================================
    # Wave Patterns
    # ========================================================================
    
    def sine_wave(
        self,
        frequency: Tuple[float, float, float] = (1, 1, 1),
        phase: Tuple[float, float, float] = (0, 0, 0),
        amplitude: float = 1.0,
        offset: float = 0.0
    ) -> Pattern3D:
        """Generate a 3D sine wave pattern."""
        freq = np.array(frequency)
        ph = np.array(phase)
        
        wave = np.sin(
            2 * np.pi * (freq[0] * self.X + freq[1] * self.Y + freq[2] * self.Z) + 
            ph[0] + ph[1] + ph[2]
        )
        
        data = wave * amplitude + offset
        
        return Pattern3D(data=data, name='sine_wave', params={'frequency': frequency})
    
    def spherical_wave(
        self,
        center: Tuple[float, float, float] = None,
        frequency: float = 2.0,
        phase: float = 0.0,
        decay: float = 0.0
    ) -> Pattern3D:
        """Generate concentric spherical waves."""
        if center is None:
            center = (self.d1 / 2, self.d2 / 2, self.d3 / 2)
        
        dist = np.sqrt(
            (self.I - center[0])**2 +
            (self.J - center[1])**2 +
            (self.K - center[2])**2
        )
        
        wave = np.sin(2 * np.pi * frequency * dist / max(self.shape) + phase)
        
        if decay > 0:
            wave = wave * np.exp(-decay * dist / max(self.shape))
        
        # Normalize to [0, 1]
        data = (wave + 1) / 2
        
        return Pattern3D(data=data, name='spherical_wave', params={'frequency': frequency})
    
    def spiral(
        self,
        center: Tuple[float, float, float] = None,
        axis: str = 'z',
        turns: float = 3.0,
        thickness: float = 2.0,
        direction: int = 1  # 1 or -1
    ) -> Pattern3D:
        """Generate a spiral pattern."""
        if center is None:
            center = (self.d1 / 2, self.d2 / 2, self.d3 / 2)
        
        if axis == 'z':
            r = np.sqrt((self.I - center[0])**2 + (self.J - center[1])**2)
            theta = np.arctan2(self.J - center[1], self.I - center[0])
            h = self.K - center[2]
        elif axis == 'y':
            r = np.sqrt((self.I - center[0])**2 + (self.K - center[2])**2)
            theta = np.arctan2(self.K - center[2], self.I - center[0])
            h = self.J - center[1]
        else:  # x
            r = np.sqrt((self.J - center[1])**2 + (self.K - center[2])**2)
            theta = np.arctan2(self.K - center[2], self.J - center[1])
            h = self.I - center[0]
        
        # Spiral: h = theta * turns / (2*pi)
        target_theta = h * 2 * np.pi / (self.shape[{'x': 0, 'y': 1, 'z': 2}[axis]] / turns)
        angle_diff = np.abs(np.mod(direction * theta - target_theta + np.pi, 2 * np.pi) - np.pi)
        
        # Distance from spiral surface
        dist = angle_diff * r / (2 * np.pi)
        data = np.exp(-0.5 * (dist / thickness)**2)
        
        return Pattern3D(data=data, name='spiral', params={'turns': turns, 'axis': axis})
    
    # ========================================================================
    # Noise Patterns
    # ========================================================================
    
    def noise(
        self,
        scale: float = 0.1,
        seed: int = None
    ) -> Pattern3D:
        """Generate uniform random noise."""
        if seed is not None:
            np.random.seed(seed)
        data = np.random.rand(*self.shape) * scale
        return Pattern3D(data=data, name='noise', params={'scale': scale})
    
    def perlin_noise(
        self,
        scale: float = 4.0,
        octaves: int = 4,
        persistence: float = 0.5,
        seed: int = None
    ) -> Pattern3D:
        """Generate Perlin-like noise (simplified implementation)."""
        if seed is not None:
            np.random.seed(seed)
        
        data = np.zeros(self.shape)
        amplitude = 1.0
        frequency = 1.0
        
        for _ in range(octaves):
            # Generate smooth noise at this frequency
            grid_size = max(2, int(min(self.shape) / (scale * frequency)))
            small = np.random.randn(grid_size, grid_size, grid_size)
            
            # Upsample with interpolation
            from scipy.ndimage import zoom
            factors = (self.d1 / grid_size, self.d2 / grid_size, self.d3 / grid_size)
            upsampled = zoom(small, factors, order=3)
            
            # Crop to exact shape
            upsampled = upsampled[:self.d1, :self.d2, :self.d3]
            
            data += upsampled * amplitude
            amplitude *= persistence
            frequency *= 2
        
        # Normalize to [0, 1]
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        return Pattern3D(data=data, name='perlin', params={'scale': scale, 'octaves': octaves})
    
    # ========================================================================
    # Lattice Patterns
    # ========================================================================
    
    def checkerboard(
        self,
        cell_size: Union[int, Tuple[int, int, int]] = 4,
        value1: float = 0.0,
        value2: float = 1.0
    ) -> Pattern3D:
        """Generate a 3D checkerboard pattern."""
        if isinstance(cell_size, int):
            cell_size = (cell_size, cell_size, cell_size)
        
        pattern = (
            (self.I // cell_size[0]) + 
            (self.J // cell_size[1]) + 
            (self.K // cell_size[2])
        ) % 2
        
        data = np.where(pattern == 0, value1, value2)
        
        return Pattern3D(data=data, name='checkerboard', params={'cell_size': cell_size})
    
    def grid_lines(
        self,
        spacing: Union[int, Tuple[int, int, int]] = 8,
        thickness: int = 1,
        value: float = 1.0
    ) -> Pattern3D:
        """Generate a 3D grid of lines."""
        if isinstance(spacing, int):
            spacing = (spacing, spacing, spacing)
        
        lines_i = (self.I % spacing[0]) < thickness
        lines_j = (self.J % spacing[1]) < thickness
        lines_k = (self.K % spacing[2]) < thickness
        
        data = (lines_i | lines_j | lines_k).astype(float) * value
        
        return Pattern3D(data=data, name='grid_lines', params={'spacing': spacing})
    
    def dots(
        self,
        spacing: Union[int, Tuple[int, int, int]] = 8,
        radius: float = 1.5,
        falloff: str = 'smooth'
    ) -> Pattern3D:
        """Generate a pattern of spherical dots on a grid."""
        if isinstance(spacing, int):
            spacing = (spacing, spacing, spacing)
        
        # Distance to nearest grid point
        dist_i = np.minimum(self.I % spacing[0], spacing[0] - self.I % spacing[0])
        dist_j = np.minimum(self.J % spacing[1], spacing[1] - self.J % spacing[1])
        dist_k = np.minimum(self.K % spacing[2], spacing[2] - self.K % spacing[2])
        
        dist = np.sqrt(dist_i**2 + dist_j**2 + dist_k**2)
        data = self._apply_falloff(dist, radius, falloff)
        
        return Pattern3D(data=data, name='dots', params={'spacing': spacing, 'radius': radius})
    
    # ========================================================================
    # Special Patterns
    # ========================================================================
    
    def shell(
        self,
        center: Tuple[float, float, float] = None,
        inner_radius: float = None,
        outer_radius: float = None,
        falloff: str = 'smooth'
    ) -> Pattern3D:
        """Generate a spherical shell pattern."""
        if center is None:
            center = (self.d1 / 2, self.d2 / 2, self.d3 / 2)
        if outer_radius is None:
            outer_radius = min(self.shape) / 3
        if inner_radius is None:
            inner_radius = outer_radius * 0.6
        
        dist = np.sqrt(
            (self.I - center[0])**2 +
            (self.J - center[1])**2 +
            (self.K - center[2])**2
        )
        
        mid_radius = (inner_radius + outer_radius) / 2
        thickness = outer_radius - inner_radius
        
        dist_from_shell = np.abs(dist - mid_radius)
        data = self._apply_falloff(dist_from_shell, thickness / 2, falloff)
        
        return Pattern3D(data=data, name='shell', params={'inner': inner_radius, 'outer': outer_radius})
    
    def gyroid(
        self,
        scale: float = 4.0,
        threshold: float = 0.0,
        smooth: bool = True
    ) -> Pattern3D:
        """Generate a gyroid minimal surface pattern."""
        x = self.X * scale * 2 * np.pi
        y = self.Y * scale * 2 * np.pi
        z = self.Z * scale * 2 * np.pi
        
        # Gyroid equation: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
        gyroid = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
        
        if smooth:
            data = (gyroid + 1.5) / 3  # Normalize roughly to [0, 1]
        else:
            data = (gyroid > threshold).astype(float)
        
        return Pattern3D(data=data, name='gyroid', params={'scale': scale})
    
    def voronoi(
        self,
        n_points: int = 20,
        seed: int = None,
        mode: str = 'distance'  # 'distance', 'cell_id', 'edge'
    ) -> Pattern3D:
        """Generate a 3D Voronoi pattern."""
        if seed is not None:
            np.random.seed(seed)
        
        # Random points
        points = np.random.rand(n_points, 3) * np.array(self.shape)
        
        # Compute distance to each point for every cell
        coords = np.stack([self.I, self.J, self.K], axis=-1)
        
        distances = np.zeros((n_points, *self.shape))
        for i, p in enumerate(points):
            distances[i] = np.sqrt(np.sum((coords - p)**2, axis=-1))
        
        if mode == 'distance':
            # Distance to nearest point
            data = np.min(distances, axis=0)
            data = data / data.max()
        elif mode == 'cell_id':
            # Cell ID (which point is closest)
            data = np.argmin(distances, axis=0).astype(float)
            data = data / n_points
        else:  # edge
            # Edge detection (difference between 1st and 2nd closest)
            sorted_dist = np.sort(distances, axis=0)
            data = sorted_dist[1] - sorted_dist[0]
            data = np.exp(-data / (data.max() * 0.1))
        
        return Pattern3D(data=data, name='voronoi', params={'n_points': n_points, 'mode': mode})
    
    # ========================================================================
    # Composition
    # ========================================================================
    
    def compose(
        self,
        patterns: List[Tuple[str, Dict, float]],
        mode: str = 'add'
    ) -> Pattern3D:
        """
        Compose multiple patterns.
        
        Args:
            patterns: List of (pattern_name, params_dict, weight) tuples
            mode: 'add', 'multiply', 'max', 'min', 'average'
        
        Example:
            pg.compose([
                ('sphere', {'radius': 10}, 1.0),
                ('noise', {'scale': 0.1}, 0.2),
                ('gradient', {'axis': 'z'}, 0.3)
            ])
        """
        result = None
        
        for pattern_name, params, weight in patterns:
            # Get pattern method
            method = getattr(self, pattern_name, None)
            if method is None:
                raise ValueError(f"Unknown pattern: {pattern_name}")
            
            pattern = method(**params)
            weighted = pattern.data * weight
            
            if result is None:
                result = weighted
            elif mode == 'add':
                result = result + weighted
            elif mode == 'multiply':
                result = result * weighted
            elif mode == 'max':
                result = np.maximum(result, weighted)
            elif mode == 'min':
                result = np.minimum(result, weighted)
            elif mode == 'average':
                result = (result + weighted) / 2
        
        return Pattern3D(
            data=result,
            name='composite',
            params={'patterns': patterns, 'mode': mode}
        )
    
    def mask(self, pattern: Pattern3D, mask: Pattern3D) -> Pattern3D:
        """Apply a mask pattern to another pattern."""
        return Pattern3D(
            data=pattern.data * mask.data,
            name=f"masked({pattern.name})",
            params={'pattern': pattern.params, 'mask': mask.params}
        )
    
    def blend(
        self,
        pattern1: Pattern3D,
        pattern2: Pattern3D,
        blend_factor: Union[float, Pattern3D] = 0.5
    ) -> Pattern3D:
        """Blend two patterns together."""
        if isinstance(blend_factor, Pattern3D):
            t = blend_factor.data
        else:
            t = blend_factor
        
        data = pattern1.data * (1 - t) + pattern2.data * t
        
        return Pattern3D(
            data=data,
            name=f"blend({pattern1.name}, {pattern2.name})",
            params={'factor': blend_factor if isinstance(blend_factor, float) else 'pattern'}
        )
    
    # ========================================================================
    # Utility
    # ========================================================================
    
    def zeros(self) -> Pattern3D:
        """Generate a zero pattern."""
        return Pattern3D(data=np.zeros(self.shape), name='zeros', params={})
    
    def ones(self, value: float = 1.0) -> Pattern3D:
        """Generate a constant pattern."""
        return Pattern3D(data=np.full(self.shape, value), name='ones', params={'value': value})
    
    def from_function(
        self,
        func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        name: str = 'custom'
    ) -> Pattern3D:
        """
        Generate pattern from custom function.
        
        Args:
            func: Function(I, J, K) -> values
                  I, J, K are meshgrid arrays of integer coordinates
        """
        data = func(self.I, self.J, self.K)
        return Pattern3D(data=data, name=name, params={'function': 'custom'})
    
    def from_tensor(self, tensor: Union[np.ndarray, torch.Tensor]) -> Pattern3D:
        """Create pattern from existing tensor."""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        if tensor.shape != self.shape:
            raise ValueError(f"Tensor shape {tensor.shape} doesn't match lattice shape {self.shape}")
        
        return Pattern3D(data=tensor.copy(), name='from_tensor', params={})


# Convenience alias
Patterns = PatternGenerator


if __name__ == "__main__":
    # Demo patterns
    print("=" * 60)
    print("Pattern Generator Demo")
    print("=" * 60)
    
    pg = PatternGenerator(shape=(32, 32, 32))
    
    # Test various patterns
    patterns = [
        ("sphere", pg.sphere(radius=12, falloff='smooth')),
        ("cube", pg.cube(size=16)),
        ("cylinder", pg.cylinder(radius=10, axis='z')),
        ("torus", pg.torus(major_radius=10, minor_radius=4)),
        ("gradient", pg.gradient(axis='z')),
        ("sine_wave", pg.sine_wave(frequency=(2, 2, 2))),
        ("checkerboard", pg.checkerboard(cell_size=4)),
        ("shell", pg.shell(inner_radius=8, outer_radius=12)),
        ("gyroid", pg.gyroid(scale=2)),
    ]
    
    print("\nGenerated patterns:")
    for name, pattern in patterns:
        print(f"  {name}: shape={pattern.data.shape}, "
              f"range=[{pattern.data.min():.3f}, {pattern.data.max():.3f}]")
    
    # Test composition
    composite = pg.compose([
        ('sphere', {'radius': 12}, 1.0),
        ('noise', {'scale': 0.2}, 0.3),
        ('gradient', {'axis': 'z'}, 0.2)
    ])
    print(f"\nComposite: range=[{composite.data.min():.3f}, {composite.data.max():.3f}]")
    
    # Test pattern operations
    inverted = patterns[0][1].invert()
    normalized = composite.normalize(0, 1)
    print(f"Inverted sphere: range=[{inverted.data.min():.3f}, {inverted.data.max():.3f}]")
    print(f"Normalized composite: range=[{normalized.data.min():.3f}, {normalized.data.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
