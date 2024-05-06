"""Particle class and functions for PIC simulation."""

import numpy as np

# Single Xenon ion charge and mass
Q = 1.60217657e-19
M = 131.293 * 1.66053892 * 1e-27


class Particles:
    """Class representing a collection of particles in a PIC simulation."""

    def __init__(self, n_particles, height):
        """
        Initialize the particle object with random positions and velocities.

        Args:
            n_particles (int): Number of particles to create.
        """

        self.num = n_particles
        self.positions = np.random.rand(n_particles, 2)
        self.positions[:, 0] = 0
        self.positions[:, 1] *= height
        self.velocities = np.random.rand(n_particles, 2)
        self.velocities[:, 0] = abs(self.velocities[:, 0])
        self.velocities[:, 1] = 0

    def push(self, pusher, electric_field, dt, grid):
        """
        Push particles using the specified pusher function.

        Args:
            pusher (callable): Particle pusher function.
            electric_field (callable): Electric field function.
            magnetic_field (callable): Magnetic field function.
            dt (float): Time step.
            grid (Grid): Grid class.
        """
        for i in range(self.num):
            x = self.positions[i]
            v = self.velocities[i]
            # print("x = ", x)
            # print("v = ", v)
            a = lambda pos: Q * electric_field.get_field_at(pos) / M
            x_new, v_new = pusher(x, v, a, dt)

            bottom_boundary = (x_new[1] <= 0 and x_new[0] <= grid.x_wall) or (x_new[1] <= 0 and x >= (grid.x_wall + grid.w_wall))
            top_boundary = x_new[1] >= grid.height
            right_boundary = x_new[0] >= grid.length

            left_wall = x_new[0] >= grid.x_wall
            right_wall = x_new[0] <= (grid.x_wall + grid.w_wall)
            top_wall = x_new[1] <= grid.h_wall
            wall = left_wall and right_wall and top_wall


            if top_boundary or bottom_boundary:
                v_new[0] = - v_new[0]
            elif right_boundary:
                v_new = np.array([0, 0])
                x_new = x.copy()
            elif wall:
                # if the particle passed throught the top wall
                if (x[1] >= grid.h_wall and x_new[1] <= grid.h_wall):
                    v_new[1] = - v_new[1]
                else:
                    v_new[0] = - v_new[0]
                
            self.positions[i] = x_new
            self.velocities[i] = v_new

    def get_positions(self):
        """Return the particle positions."""
        return self.positions

    def get_velocities(self):
        """Return the particle velocities."""
        return self.velocities

    def get_position(self, i):
        """Return the particle position."""
        if i >= self.num:
            raise IndexError("Particle index out of range.")
        else:
            return self.positions[i]

    def get_velocity(self, i):
        """Return the particle velocity."""
        if i >= self.num:
            raise IndexError("Particle index out of range.")
        else:
            return self.velocities[i]
