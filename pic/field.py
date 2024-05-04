"""Electric field solvers for the PIC method."""

import numpy as np

class ElectricField:
    """Class representing the electric field in a PIC simulation."""

    def __init__(self, grid, charge_density):
        """Initialize the electric field object."""
        self.charge_density = charge_density
        self.grid = grid
        self.field = None

    def get_electric_field(self, x):
        """
        Return the electric field at a given position.

        Args:
            x (numpy.ndarray): Position vector.

        Returns:
            numpy.ndarray: Electric field vector at the given position.
        """
        return self.field

    def set_electric_field(self, field):
        """
        Set the electric field.

        Args:
            field (numpy.ndarray): Electric field vector.
        """
        self.field = field
