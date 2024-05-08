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
    length = 0.1
    height = 0.05
    h_wall = height / 5
    w_wall = length / 5
    x_wall = 0.2

    # Set electric potentials
    Vin = 1100
    Vout = -100
    Vwall = 1000
    v0 = 100  # initial velocity of the ions (m/s)

    dt = 100 * h * 1 / np.sqrt(2 * Q * (Vin - Vout) / M)
    print("dt = ", dt)

    # Set up the simulation parameters
    n_particles = 1
    n_steps = 10000

    pushers = {
        "euler": euler,
        "rk4": rk4,
        "leapfrog": leapfrog,
        "verlet": lambda x, v, a, dt: leapfrog(x, v, a, dt, use_verlet=True),
    }
    print(f"Using {methods} method for integration")

    if len(methods) > 1 or methods[0] != "euler":
        raise ValueError("please only run with euler on this branch!")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    colors = ["red", "blue", "green"]
    linestyles = ["-", "-", "--"]
    fig_conv, axs_conv = plt.subplots(1, 1, figsize=(5, 5))
    E_diffs = []
    dts = []
    for dt in [dt, dt / 10, dt / 100]:
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
                axs[0].plot(
                    np.arange(len(E[:-2])),
                    E[:-2],
                    label=method,
                    linestyle=linestyles[i],
                )
                axs[0].set_xlabel("step number", fontsize=13)
                axs[0].set_ylabel("Total Energy", fontsize=13)
                axs[0].set_title("Energy with the wall", fontsize=14)
                axs[0].legend()
                plt.tight_layout()

                E_diffs.append(E[0] - E[-3])
                dts.append(dt)

            for j in range(n_particles):
                paths[j] = np.array(paths[j])
                axs[1].plot(
                    paths[j][:-2, 0], paths[j][:-2, 1], linewidth=3, color=colors[i]
                )

    axs_conv.scatter(dts, E_diffs)
    axs_conv.set_xscale("log")
    axs_conv.set_yscale("log")
    p = np.polyfit(x=dts, y=E_diffs, deg=2)
    print(p)
    fit_y = np.polyval(p, np.array([dts[0], dts[-1]]))
    axs_conv.plot(
        [dts[0], dts[-1]],
        fit_y,
        "--",
        label=r"degree 2 polynomial fit $\sim \Delta t^2$",
    )
    axs_conv.set_xscale("log")
    axs_conv.set_yscale("log")
    axs_conv.legend()
    axs_conv.set_xlabel("dt (s)")
    axs_conv.set_ylabel("Initial E - Final E (J))")
    axs_conv.set_title("Euler Energy Loss versus timestep size")
    fig_conv.savefig("Euler_energy_loss_co")

    p = axs[1].contourf(fields.grid.Xs, fields.grid.Ys, fields.V, 20)
    plt.gca().add_patch(
        plt.Rectangle(
            (grid.x_wall, 0),
            grid.w_wall,
            grid.h_wall,
            edgecolor="k",
            facecolor="none",
        )
    )
    plt.colorbar(p)
    axs[1].set_title("Electric Potential (V)", fontsize=14)
    axs[1].set_xlabel("x", fontsize=13)
    axs[1].set_ylabel("y", fontsize=13)

    # fields.plot_E_field()
    # plt.savefig("electric_field.png")

    # fields.plot_contour_V()
    # plt.savefig("potential.png")
    plt.tight_layout()
    plt.show()
