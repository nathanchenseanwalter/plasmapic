"""Electric field solvers for the PIC method."""

import numpy as np
from scipy.sparse.linalg import spsolve
from .grid import make_array

MU_0 = 4 * np.pi * 1e-7
EPS_0 = 8.854187817e-12
Q_E = 1.60217662e-19
M_I = 1.6726219e-27

class ElectricField:
    """Class representing the electric field in a PIC simulation."""

    def __init__(self, grid):
        """Initialize the electric field object."""
        self.grid = grid
        
        self.V = self.solve_V()
        self.Ex, self.Ey = self.solve_E()

    def solve_V(self):
        """Solve for the electric potential."""
        A = self.grid.get_A()
        b = self.grid.get_b()
        V = spsolve(A, b)
        return make_array(V, self.grid.Nx, self.grid.Ny)

    def solve_E(self):
        """Solve for the electric field."""
        Ex, Ey = np.gradient(self.V)
        Ex = -Ex
        Ey = -Ey
        return Ex, Ey
