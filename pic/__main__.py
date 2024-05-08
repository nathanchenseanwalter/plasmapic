"""Main particle in cell simulator for plasma simulations."""

if __name__ == "__main__":
    print("Running simulation...")
    # Import the necessary modules
    import sys

    if len(sys.argv) > 1:
        method = str(sys.argv[1])
    else:
        method = "euler"
    import numpy as np

    np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=100000)
    import matplotlib.pyplot as plt

    from pic.grid import Grid
    from pic.field import ElectricField
    from pic.particle import Particles
    from pic.integrator import euler, rk4, leapfrog
    from pic.particle import Q, M

    def analytical(q, E, t, m, v_0, h_0):
        x = 0.5 * q * E * t**2 / m + np.array([v_0 * t, 0]) + np.array([0, h_0])
        return x

    def l2_error(f, g):
        error = np.mean((f - g) ** 2)
        return error

    def max_error(f, g):
        return np.max(np.abs(f - g))

    # Set up the simulation parameters
    n_particles = 1
    n_steps = 20000

    pusher = {"euler": euler, "rk4": rk4, "leapfrog": leapfrog}[method]
    print(f"Using {method} method for integration")

    # Set up grid parameters
    h = 1e-3
    length = 0.5
    height = 0.02
    h_wall = height / 5
    w_wall = length / 5
    x_wall = 0.01

    # Set electric potentials
    Vin = 11
    Vout = -1
    Vwall = 1000
    v0 = 1  # initial velocity of the ions (m/s)

    dt = h / np.sqrt(2 * Q * (Vin - Vout) / M)
    print("dt = ", dt)

    # Initialize the objects

    # plt.figure()

    dt_arr = np.array(
        [
            h * scale / np.sqrt(2 * Q * (Vin - Vout) / M)
            for scale in [
                1e-3,
                1e-2,
                1e-1,
            ]
        ]
    )
    fig, ax = plt.subplots()
    analytic_plotted = False
    fig2, ax2 = plt.subplots()

    for method in ["euler", "rk4", "leapfrog"]:
        pusher = {"euler": euler, "rk4": rk4, "leapfrog": leapfrog}[method]
        grid = Grid(h, length, height, h_wall, w_wall, x_wall, Vin, Vout, Vwall)
        particles = Particles(n_particles, height, v0)
        fields = ElectricField(grid)
        paths = {}
        energies = {}
        analytical_sol = []
        errors = []
        t = 0
        for dt in dt_arr:
            for k in range(n_steps):

                if k == 0:
                    for j in range(n_particles):
                        paths[j] = [np.array(particles.get_position(j))]
                        r = particles.get_position(j)
                        v = particles.get_velocity(j)
                        K = 0.5 * particles.M * np.linalg.norm(v) ** 2
                        U = particles.Q * fields.get_potential_at(r)
                        analytical_sol = [
                            analytical(Q, fields.get_field_at(r), t, M, v0, r[1])
                        ]
                        energies[j] = [K + U]
                        # analytical_sol[j, k] = analytical(Q, fields.get_field_at(r), t, M, v0, r[1])
                        # print(analytical(Q, fields.get_field_at(r), t, M, v0, r[1]))

                particles.push(pusher, fields, dt, grid)
                for j in range(n_particles):
                    paths[j].append(np.array(particles.get_position(j)))
                    r = particles.get_position(j)
                    v = particles.get_velocity(j)
                    K = 0.5 * particles.M * np.linalg.norm(v) ** 2
                    U = particles.Q * fields.get_potential_at(r)
                    energies[j].append(K + U)

                    analytical_sol.append(
                        analytical(Q, fields.get_field_at(r), t, M, v0, r[1])
                    )
                t += dt

                if r[0] >= length:
                    break

            analytical_sol = np.array(analytical_sol)
            errors.append(l2_error(analytical_sol[:, 0], np.array(paths[0])[:, 0]))
            if not analytic_plotted:
                for j in range(n_particles):
                    paths[j] = np.array(paths[j])
                    ax.plot(
                        paths[j][:, 0],
                        paths[j][:, 1],
                        linewidth=3,
                        color="r",
                        label="euler solution",
                    )
                    ax.plot(
                        analytical_sol[:, 0],
                        analytical_sol[:, 1],
                        label="exact solution",
                    )
                    ax.set_ylim([5e-3 - 1e-5, 5e-3 + 1e-5])
                    ax.legend()

                ax.set_xlabel("x", fontsize=13)
                ax.set_ylabel("y", fontsize=13)
                ax.set_title("Exact vs Euler Solution", fontsize=14)
                fig.tight_layout()
                fig.savefig("analytic.png")
                analytic_plotted = True
        print(dt_arr)
        print(method)
        print(errors)
        p = np.polyfit(x=np.log(dt_arr), y=np.log(errors), deg=1)
        print(p)

        # ax2.loglog(dt_arr, np.abs(errors), "o", label=method, alpha=0.5)
        # ax2.set_xlabel("dt", fontsize=14)
        # ax2.set_ylabel("max error", fontsize=14)
    ax2.legend()

    # fields.plot_contour_V(new_fig=False)
    # plt.gca().add_patch(
    #     plt.Rectangle(
    #         (grid.x_wall, 0),
    #         grid.w_wall,
    #         grid.h_wall,
    #         edgecolor="k",
    #         facecolor="none",
    #     )
    # )

    plt.show()
