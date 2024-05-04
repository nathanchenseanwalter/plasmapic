"""Finite difference grid and matrix generation functions."""
import numpy as np
from scipy.sparse import csr_matrix

class Grid:
    """Class representing a finite difference grid for the PIC method."""
    def __init__(self, h, length, height, h_wall, w_wall, x_wall, Vin, Vout, Vwall):
        """
        Initialize the grid object.

        Args:
            N (int): Number of grid points.
            length (float): Length of the grid.
        """
        self.h = h
        self.length = length
        self.height = height
        self.h_wall = h_wall
        self.w_wall = w_wall
        self.x_wall = x_wall
        
        self.Vin = Vin
        self.Vout = Vout
        self.Vwall = Vwall
        
        self.Xs, self.Ys = np.meshgrid(np.arange(0, length, h), np.arange(0, height, h))
        
        self.Nx = int(length / h)
        self.Ny = int(height / h)
        
        self._A, self._b = self.get_laplacian()

    def get_h(self):
        """Return the grid spacing."""
        return self.h

    def get_length(self):
        """Return the length of the grid."""
        return self.length
    
    def get_height(self):
        """Return the height of the grid."""
        return self.height
    
    def get_x_wall(self):
        """Return the distance x of the wall to the inlet."""
        return self.x_wall
    
    def get_w_wall(self):
        """Return the width of the wall."""
        return self.w_wall
    
    def get_h_wall(self):
        """Return the height of the wall."""
        return self.h_wall
    
    def get_A(self):
        """Return the coefficient matrix A."""
        return self._A
    
    def get_b(self):
        """Return the RHS vector b."""
        return self._b

    def get_laplacian(self):
        """Return the 2D Laplacian operator using sparse matrix."""
        data = []
        b = np.zeros(self.Nx * self.Ny)
        h = self.h
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                id = i * self.Ny + j
                
                # Inlet Boundary Condition
                if i == 0:
                    data.append(np.array([id, id, 1]))
                    b[id] = self.Vin
                # Outlet Boundary Condition
                elif i == self.Nx - 1:
                    data.append(np.array([id, id, 1]))
                    b[id] = self.Vout
                # Wall Boundary Condition
                elif (i >= int(self.x_wall / h) and i <= int((self.x_wall + self.w_wall) / h)) and (j <= int(self.h_wall / h)):
                    data.append(np.array([id, id, 1]))
                    b[id] = self.Vwall
                # Dirichlet Boundary Condition at bottom
                elif j == 0:
                    data.append(np.array([id, id, 1]))
                    data.append(np.array([id, id+self.Ny, -1]))
                # Dirichlet Boundary Condition at top
                elif j == self.Ny - 1:
                    data.append(np.array([id, id, 1]))
                    data.append(np.array([id, id-self.Ny, -1]))
                # Rest of the domain with standar Laplacian
                else:
                    data.append(np.array([id, id, -4/h**2]))
                    data.append(np.array([id, id+self.Ny, 1/h**2]))
                    data.append(np.array([id, id-self.Ny, 1/h**2]))
                    data.append(np.array([id, id+1, 1/h**2]))
                    data.append(np.array([id, id-1, 1/h**2]))
                    
        data = np.array(data)
        A = csr_matrix((data[:, 2], (data[:, 0], data[:, 1])), shape=(self.Nx * self.Ny, self.Nx * self.Ny))
        return A, b
    
def make_vector(x, Nx, Ny):
    """Return a vector from a 2D array."""
    vec = np.zeros(Nx * Ny)
    for i in range(Nx):
        for j in range(Ny):
            vec[i * Ny + j] = x[i, j]
            
    return vec

def make_array(x, Nx, Ny):
    """Return a 2D array from a vector."""
    arr = np.zeros([Ny, Nx])
    for i in range(Ny):
        for j in range(Nx):
            arr[i, j] = x[j*Ny + i]
            
    return arr