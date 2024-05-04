"""Finite difference grid and matrix generation functions."""
import numpy as np
from scipy.sparse import diags

class Grid:
    """Class representing a finite difference grid for the PIC method."""
    def __init__(self, N, length):
        """
        Initialize the grid object.

        Args:
            N (int): Number of grid points.
            length (float): Length of the grid.
        """
        self.N = N
        self.length = length
        self.dx = length / (N - 1)

    def get_dx(self):
        """Return the grid spacing."""
        return self.dx

    def get_grid_res(self):
        """Return the number of grid points."""
        return self.N

    def get_length(self):
        """Return the length of the grid."""
        return self.length

    def get_laplacian(self):
        """Return the 1D Laplacian operator."""
        laplacian = np.zeros((self.N, self.N))
        for i in range(1, self.N - 1):
            laplacian[i, i - 1] = 1
            laplacian[i, i] = -2
            laplacian[i, i + 1] = 1
        return laplacian

    def get_laplacian_sparse(self):
        """Return the 1D Laplacian operator as a sparse matrix."""
        from scipy.sparse import diags
        diagonals = [[1], [-2], [1]]
        laplacian = diags(diagonals, [-1, 0, 1], shape=(self.N, self.N))
        return laplacian

    def get_laplacian_2d(self):
        """Return the 2D Laplacian operator."""
        laplacian_x = self.get_laplacian_sparse()
        laplacian_y = self.get_laplacian_sparse()
        return laplacian_x, laplacian_y

    def get_laplacian_2d_sparse(self):
        """Return the 2D Laplacian operator as a sparse matrix."""
        laplacian_x = self.get_laplacian_sparse()
        laplacian_y = self.get_laplacian_sparse()
        return laplacian_x, laplacian_y