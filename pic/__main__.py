"""Main particle in cell simulator for plasma simulations."""

if __name__ == "__main__":
    print("Running simulation...")
    # Import the necessary modules
    import sys

    methods = []
    if len(sys.argv) > 1:
        methods = [str(sys.argv[i]) for i in range(1, len(sys.argv))]
    else:
        methods = ["rk4"]

    import numpy as np

    np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=100000)
    import matplotlib.pyplot as plt

    from pic.grid import Grid
    from pic.field import ElectricField
    from pic.particle import Particles
    from pic.integrator import euler, rk4, leapfrog
    from pic.particle import Q, M

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
    v0 = 100  # initial velocity of the ions (m/s)

    dt = h * 1 / np.sqrt(2 * Q * (Vin - Vout) / M)
    print("dt = ", dt)

    # Set up the simulation parameters
    n_particles = 1
    n_steps = 10000

    pushers = {"euler": euler, "rk4": rk4, "leapfrog": leapfrog}
    print(f"Using {methods} method for integration")


    fig, ax = plt.subplots()
    colors = ["red", "blue", "green"]
    plt.figure()
    for i, method in enumerate(methods):
        pusher = pushers[method]
        # Initialize the objects
        grid = Grid(h, length, height, h_wall, w_wall, x_wall, Vin, Vout, Vwall)
        particles = Particles(n_particles, height, v0)
        fields = ElectricField(grid)

        paths = {}
        energies = {}
        for k in range(n_steps):
            if k == 0:
                for j in range(n_particles):
                    paths[j] = [np.array(particles.get_position(j))]
                    r = particles.get_position(j)
                    v = particles.get_velocity(j)
                    K = 0.5 * particles.M * np.linalg.norm(v) ** 2
                    U = particles.Q * fields.get_potential_at(r)
                    energies[j] = [K + U]

            particles.push(pusher, fields, dt, grid)
            for j in range(n_particles):
                paths[j].append(np.array(particles.get_position(j)))
                r = particles.get_position(j)
                v = particles.get_velocity(j)
                K = 0.5 * particles.M * np.linalg.norm(v) ** 2
                U = particles.Q * fields.get_potential_at(r)
                energies[j].append(K + U)

            if r[0] >= length:
                break

        for j in range(n_particles):
            E = energies[j]
            print(np.shape(E))
            ax.plot(np.arange(len(E[:-2])), E[:-2], label=method)
            # ax.vlines(ymin=)
            # ax.plot(np.arange(len(E)), E)
            ax.set_xlabel("step number")
            ax.set_ylabel("Total Energy")
            ax.legend()

        for j in range(n_particles):
            paths[j] = np.array(paths[j])
            plt.plot(paths[j][:-2, 0], paths[j][:-2, 1], linewidth=3, color=colors[i])
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
        # plt.gca().set_xlim([0, length])
        # plt.gca().set_ylim([0, height])
        # plt.savefig(f"trajectories_{method}.png")

        # fields.plot_E_field()
        # plt.savefig("electric_field.png")

        # fields.plot_contour_V()
        # plt.savefig("potential.png")

        

    plt.show()
