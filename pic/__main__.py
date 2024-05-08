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
        x = 0.5 * q * E * t ** 2 / m + np.array([v_0 * t, 0]) + np.array([0, h_0])
        return x
    
    def l2_error(f, g):
        error = np.sqrt(((f - g) ** 2).mean())
        return error
    
    def max_error(f, g):
        return np.max(f - g)

    # Set up the simulation parameters
    n_particles = 1
    n_steps = 2000

    pusher = {"euler": euler, "rk4": rk4, "leapfrog": leapfrog}[method]
    print(f"Using {method} method for integration")

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

    dt = h / np.sqrt(2 * Q * (Vin - Vout) / M)
    print("dt = ", dt)
    

    # Initialize the objects
    

    # plt.figure()
    
    dt_arr = np.array([h * scale / np.sqrt(2 * Q * (Vin - Vout) / M) for scale in np.arange(10, 10000, 1000)])

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
                        analytical_sol = [analytical(Q, fields.get_field_at(r), t, M, v0, r[1])]
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
                
                    analytical_sol.append(analytical(Q, fields.get_field_at(r), t, M, v0, r[1]))
                t += dt

                if r[0] >= length:
                    break
            
            analytical_sol = np.array(analytical_sol)
            errors.append(max_error(analytical_sol[:, 0], np.array(paths[0])[:, 0]))
            # for j in range(n_particles):
            #     paths[j] = np.array(paths[j])
            #     plt.plot(paths[j][:, 0], paths[j][:, 1], linewidth=3, color="r", label="euler solution")
            #     plt.plot(analytical_sol[:, 0], analytical_sol[:, 1], label="exact solution")

            # plt.xlabel("x", fontsize=13)
            # plt.ylabel("y", fontsize=13)
            # plt.title("Exact vs Euler Solution", fontsize=14)

        plt.loglog(dt_arr[2:], errors[2:], '--o', label=method, alpha=0.5)
        plt.xlabel("dt", fontsize=14)
        plt.ylabel("max error", fontsize=14)
    plt.legend()

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
