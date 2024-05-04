"""Particle class and functions for PIC simulation."""

import numpy as np

class Particles:
    """Class representing a collection of particles in a PIC simulation."""
    def __init__(self, n_particles):
        """
        Initialize the particle object with random positions and velocities.

        Args:
            n_particles (int): Number of particles to create.
        """
        self.n_particles = n_particles
        self.positions = np.random.rand(n_particles, 2)
        self.velocities = np.random.rand(n_particles, 2)

    def push(self, pusher, electric_field, magnetic_field, dt):
        """
        Push particles using the specified pusher function.

        Args:
            pusher (callable): Particle pusher function.
            electric_field (callable): Electric field function.
            magnetic_field (callable): Magnetic field function.
            dt (float): Time step.
        """
        for i in range(self.n_particles):
            x = self.positions[i]
            v = self.velocities[i]
            a = lambda x, v: electric_field(x) + np.cross(v, magnetic_field(x))
            x_new, v_new = pusher(x, v, a, dt)
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
        if i >= self.n_particles:
            raise IndexError("Particle index out of range.")
        else:
            return self.positions[i]

    def get_velocity(self, i):
        """Return the particle velocity."""
        if i >= self.n_particles:
            raise IndexError("Particle index out of range.")
        else:
            return self.velocities[i]