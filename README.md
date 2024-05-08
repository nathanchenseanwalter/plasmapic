# Particle in Cell Simulation for Plasma

2024 Spring APC 523 Final Project

![Problem](./problem.png)
Positively charge particles enter the simulation domain from the inlet and exposed to an electric field due to imposed boundary conditions. Particles feel some electric force and their trajectory is tracked until they leave the simulation domain. All of the walls are assumed to reflect the particles.

How to run
---
After cloning the repo simply run following command,
`python -m pic`
For testing different integrators, use following syntax,
`python -m pic rk4`
Available integrators are `euler`, `rk4` and `leapfrog`.

checkout each of the branches to create the figures in the report:
- `energy-euler-scaling` for the euler energy loss scaling plot
- `analytical` for the comparison to analytic solution
- `energy-plots` for the energy plots

Group Members
---

- Nathaniel CHEN
- Yigit Gunsur ELMACIOGLU
- Kian ORR
- Dario PANICI

