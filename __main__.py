"""Main particle in cell simulator for plasma simulations."""

if __name__ == "__main__":
    print("Running main script...")
    # Import the necessary modules
    import numpy as np
    import matplotlib.pyplot as plt
    
    from pic.grid import Grid, make_array, make_vector
    from pic.field import ElectricField
    # from pic.particle import Particles
    from pic.integrator import euler, rk4, leapfrog
    
    # Set up the simulation parameters
    n_particles = 10
    n_steps = 100
    dt = 0.01
    pusher = leapfrog
    
    # Set up grid parameters
    h = 0.01
    length = 4.0
    height = 2.0
    h_wall = height / 5
    w_wall = length / 5
    x_wall = 1
    
    # Set electric potentials
    Vin = 1000
    Vout = 0
    Vwall = 3000

    # Initialize the objects
    grid = Grid(h, length, height, h_wall, w_wall, x_wall, Vin, Vout, Vwall)
    # particles = Particles(n_particles)
    fields = ElectricField(grid)

    plt.figure()
    plt.contourf(grid.Xs, grid.Ys, make_array(grid.get_b(), grid.Nx, grid.Ny))
    plt.colorbar()
    
    plt.figure()
    plt.contourf(grid.Xs, grid.Ys, fields.V)
    plt.colorbar()
    
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Surface(z=fields.V, x=grid.Xs, y=grid.Ys)])
    fig.update_layout(width=700, height=700)
    
    fig.show()
    plt.show()