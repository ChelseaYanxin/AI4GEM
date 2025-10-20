from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class Material:
    eps: float  # permittivity (epsilon) 介电常数
    mu: float   # permeability (mu) 磁导率
    sigma: float = 0.0  # electric conductivity 电导率

class Grid:
    """
    构建一个 nx × nz 的二维网格，存储每个点的材料参数和场分量。
    Ey, Hx, Hz 分别是电场和磁场的二维数组，初始化为零。
    """
    def __init__(self, nx: int, nz: int, dx: float, dz: float,
                 eps_map: np.ndarray, mu_map: np.ndarray, sigma_map: np.ndarray):
        assert eps_map.shape == (nx, nz)
        assert mu_map.shape == (nx, nz)
        assert sigma_map.shape == (nx, nz)
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.eps = eps_map.astype(np.float32)
        self.mu = mu_map.astype(np.float32)
        self.sigma = sigma_map.astype(np.float32)

        # Field arrays (staggered positions approximated on same lattice for simplicity):
        self.Ey = np.zeros((nx, nz), dtype=np.float32)
        self.Hx = np.zeros((nx, nz), dtype=np.float32)
        self.Hz = np.zeros((nx, nz), dtype=np.float32)

    @staticmethod
    # 快速生成一个均匀材料的网格（所有点参数一样）
    def homogeneous(nx: int, nz: int, dx: float, dz: float,
                    eps: float, mu: float, sigma: float = 0.0) -> "Grid":
        eps_map = np.full((nx, nz), eps, dtype=np.float32)
        mu_map = np.full((nx, nz), mu, dtype=np.float32)
        sigma_map = np.full((nx, nz), sigma, dtype=np.float32)
        return Grid(nx, nz, dx, dz, eps_map, mu_map, sigma_map)
