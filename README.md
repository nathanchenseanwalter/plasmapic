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

On this branch, run `python -m pic euler` in order to make the euler energy loss plot shown in the report.

Group Members
---

- Nathaniel CHEN
- Yigit Gunsur ELMACIOGLU
- Kian ORR
- Dario PANICI
