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
    n_particles = 5
    n_steps = 1000
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

    plt.figure()
    paths = {}
    energies = {}
    for k in range(n_steps):
        if k == 0:
            for j in range(n_particles):
                paths[j] = [np.array(particles.get_position(j))]
                r = particles.get_position(j)
                v = particles.get_velocity(j)
                K = 0.5 * particles.M * np.linalg.norm(v)
                U = particles.Q * fields.get_field_at(r)
                energies[j] = [K + U]

        particles.push(pusher, fields, dt, grid)
        for j in range(n_particles):
            paths[j].append(np.array(particles.get_position(j)))
            r = particles.get_position(j)
            v = particles.get_velocity(j)
            K = 0.5 * particles.M * np.linalg.norm(v)
            U = particles.Q * fields.get_field_at(r)
            energies[j].append(K + U)
    for j in range(n_particles):
        paths[j] = np.array(paths[j])
        plt.plot(paths[j][:, 0], paths[j][:, 1], linewidth=3, color="r")
    fields.plot_contour_V(new_fig=False)
    plt.gca().add_patch(
        plt.Rectangle(
            (grid.x_wall, 0),
            grid.w_wall,
            grid.h_wall,
            edgecolor="k",
            facecolor="none",
        )
    )
    plt.savefig("trajectories.png")

    fields.plot_E_field()
    plt.savefig("electric_field.png")

    fields.plot_contour_V()
    plt.savefig("potential.png")

    plt.figure()
    for j in range(n_particles):
        E = energies[j]
        plt.plot(np.arange(len(E)), E, label=f"particle {j}")
        plt.xlabel("step number")
        plt.ylabel("Total Energy")
        plt.legend()

    plt.show()
