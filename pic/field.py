"""Electric field solvers for the PIC method."""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from .grid import make_array
import time


class ElectricField:
    """Class representing the electric field in a PIC simulation."""

    def __init__(self, grid):
        """Initialize the electric field object.

        Args
        ----
        grid (Grid) : a Grid object containing the mesh grid and potential information
                    of the environment (the inlet, outlet, and the walls).

        """
        self.grid = grid

        t0 = time.time()
        self.V = self.solve_V()
        print(f"Time to solve V:  {(time.time() - t0):.5f} seconds")
        t0 = time.time()
        self.Ex, self.Ey = self.solve_E()
        print(f"Time to find E:  {(time.time() - t0):.5f} seconds")
        t0 = time.time()
        self.fEx = interp2d(self.grid.Xs[0], self.grid.Ys[:, 0], self.Ex, kind="linear")
        self.fEy = interp2d(self.grid.Xs[0], self.grid.Ys[:, 0], self.Ey, kind="linear")
        print(f"Time to interpolate E:  {(time.time() - t0):.5f} seconds")

    def solve_V(self):
        """Solve for the electric potential."""
        A = self.grid.get_A()
        b = self.grid.get_b()
        V = spsolve(A, b)
        return make_array(V, self.grid.Nx, self.grid.Ny)

    def solve_E(self):
        """Solve for the electric field."""
        Ey, Ex = np.gradient(self.V, self.grid.h)
        Ey[0, :] = 0
        Ey[-1, :] = 0
        Ex = -Ex
        Ey = -Ey
        return Ex, Ey

    def get_field_at(self, x):
        """Return the electric field at a given position."""
        return np.array([self.fEx(x[0], x[1]), self.fEy(x[0], x[1])]).flatten()

    def plot_E_field(self):
        """Plot the electric field."""
        plt.figure()
        plt.quiver(self.grid.Xs, self.grid.Ys, self.Ex, self.Ey, color="b")
        plt.title("Electric Field")
        plt.xlabel("x")
        plt.ylabel("y")

    def plot_contour_V(self):
        plt.figure()
        plt.contourf(self.grid.Xs, self.grid.Ys, self.V)
        plt.colorbar()
        plt.title("Electric Potential")
        plt.xlabel("x")
        plt.ylabel("y")
