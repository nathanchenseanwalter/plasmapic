"""Main particle in cell simulator for plasma simulations."""

if __name__ == "__main__":
    print("Running simulation...")
    # Import the necessary modules
    import numpy as np

    np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=100000)
    import matplotlib.pyplot as plt

    from pic.grid import Grid, make_array
    from pic.field import ElectricField
    from pic.particle import Particles
    from pic.integrator import euler, rk4, leapfrog
    from pic.particle import Q, M

    # Set up the simulation parameters
    n_particles = 1
    n_steps = 10000
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

    dt = h / np.sqrt(2 * Q * (Vin - Vout) / M)
    print("dt = ", dt)

    # Initialize the objects
    grid = Grid(h, length, height, h_wall, w_wall, x_wall, Vin, Vout, Vwall)
    particles = Particles(n_particles, height)
    fields = ElectricField(grid)



    path = []
    for _ in range(n_steps):
        particles.push(pusher, fields, dt, grid)
        path.append(np.array(particles.get_position(0)))
    path = np.array(path)
    plt.figure()
    plt.contourf(grid.Xs, grid.Ys, make_array(grid.get_b(), grid.Nx, grid.Ny))
    plt.plot(path[:, 0], path[:, 1], linewidth=1, color='r', linestyle="--")
    plt.contour(grid.Xs, grid.Ys, fields.V)

    fields.plot_E_field()
    fields.plot_contour_V()

    # plt.figure()
    # plt.contourf(grid.Xs, grid.Ys, make_array(grid.get_b(), grid.Nx, grid.Ny))
    # plt.colorbar()

    plt.show()