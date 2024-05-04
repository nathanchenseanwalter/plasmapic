"""Main particle in cell simulator for plasma simulations."""

if __name__ == "__main__":
    print("Running main script...")
    # Import the necessary modules
    import numpy as np
    import matplotlib.pyplot as plt
    
    from .particle import Particles
    from pic.field import ElectricField
    from pic.integrator import euler, rk4, leapfrog
    
    # Set up the simulation parameters
    n_particles = 1000
    n_steps = 100
    dt = 0.01
    pusher = leapfrog

    # Initialize the particle and field objects
    particles = Particles(n_particles)
    fields = ElectricField()

    # Run the simulation loop
    for i in range(n_steps):
        # Push particles using the leapfrog method
        particles.push(pusher, fields.get_electric_field, fields.get_magnetic_field, dt)


    # Show the plots
    plt.show()