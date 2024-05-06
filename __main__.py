"""Main particle in cell simulator for plasma simulations."""

if __name__ == "__main__":
    print("Running main script...")
    # Import the necessary modules
    import numpy as np
    np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=100000)
    import matplotlib.pyplot as plt
    
    from pic.grid import Grid, make_array, make_vector
    from pic.field import ElectricField
    from pic.particle import Particles
    from pic.integrator import euler, rk4, leapfrog
    
    # Set up the simulation parameters
    n_particles = 100
    n_steps = 100
    dt = 1e-7
    pusher = euler
    
    # Set up grid parameters
    h = 1e-4
    length = 0.05
    height = 0.02
    h_wall = height / 5
    w_wall = length / 5
    x_wall = 0.01
    
    # Set electric potentials
    Vin = 1100
    Vout = -100
    Vwall = 1000

    # Initialize the objects
    grid = Grid(h, length, height, h_wall, w_wall, x_wall, Vin, Vout, Vwall)
    particles = Particles(n_particles, height)
    fields = ElectricField(grid)
    plt.scatter(particles.positions[:, 0], particles.velocities[:, 0])
    
    # path = []
    # for _ in range(n_steps):
    #     particles.push(pusher, fields, dt, grid)
    #     path.append(np.array(particles.get_position(0)))
    # path = np.array(path)
    # plt.figure()
    # plt.contourf(grid.Xs, grid.Ys, make_array(grid.get_b(), grid.Nx, grid.Ny))
    # plt.plot(path[:, 0], path[:, 1], linewidth=3, color='r')    
    
    fields.plot_E_field()
    fields.plot_contour_V()
    
    # plt.figure()
    # plt.contourf(grid.Xs, grid.Ys, make_array(grid.get_b(), grid.Nx, grid.Ny))
    # plt.colorbar()
    
    plt.show()