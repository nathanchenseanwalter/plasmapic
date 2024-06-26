import numpy as np


def B_update(x):
    return np.array([0., 0., np.sqrt(x[0]**2 + x[1]**2)]);

def E_update(x):
    return np.array([x[0], x[1], 0])/np.sqrt(x[0]**2 + x[1]**2)*1e-2;

def euler(x, v, a, dt):
    """
    Implements the Euler forward method for particle pushing.

    Args:
        x (numpy.ndarray): Initial position of the particle (3D vector).
        v (numpy.ndarray): Initial velocity of the particle (3D vector).
        a (callable): Acceleration function that takes position and velocity as arguments.
        dt (float): Time step.

    Returns:
        tuple: Updated position and velocity of the particle after the push.
    """
    # Update velocity using acceleration
    v_new = v + a(x, v) * dt

    # Update position using updated velocity
    x_new = x + v_new * dt

    return x_new, v_new


def rk4(x, v, a, dt):
    """
    Implements the fourth-order Runge-Kutta (RK4) method for particle pushing.

    Args:
        x (numpy.ndarray): Initial position of the particle (3D vector).
        v (numpy.ndarray): Initial velocity of the particle (3D vector).
        a (callable): Acceleration function that takes position and velocity as arguments.
        dt (float): Time step.

    Returns:
        tuple: Updated position and velocity of the particle after the push.
    """
    # Define the RK4 coefficients
    k1_v = a(x, v)
    k1_x = v

    k2_v = a(x + 0.5 * dt * k1_x, v + 0.5 * dt * k1_v)
    k2_x = v + 0.5 * dt * k1_v

    k3_v = a(x + 0.5 * dt * k2_x, v + 0.5 * dt * k2_v)
    k3_x = v + 0.5 * dt * k2_v

    k4_v = a(x + dt * k3_x, v + dt * k3_v)
    k4_x = v + dt * k3_v

    # Update position and velocity using RK4
    x_new = x + (dt / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
    v_new = v + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return x_new, v_new

def leapfrog(x, v, a, dt, use_verlet=False):
    """
    Implements the leapfrog algorithm for particle pushing.

    Args:
        x (numpy.ndarray): Initial position of the particle (3D vector).
        v (numpy.ndarray): Initial velocity of the particle (3D vector).
        a (numpy.ndarray): Acceleration of the particle (3D vector).
        dt (float): Time step.
        use_verlet (bool): Flag to use velocity Verlet algorithm (default: False).

    Returns:
        tuple: Updated position and velocity of the particle after the push.
    """
    if use_verlet:
        # Velocity Verlet algorithm
        x_new = x + v * dt + 0.5 * a * dt**2
        v_new = v + 0.5 * (a + a) * dt
    else:
        # Leapfrog algorithm
        v_half = v + 0.5 * a * dt
        x_new = x + v_half * dt
        v_new = v_half + 0.5 * a * dt

    return x_new, v_new

def tajima_implicit(x, v, E, B, q, m, dt):
    
    v_minus_half = v - 0.5*q/m*E*dt;
    B_mag = np.linalg.norm(B)
    eps = q*B_mag/m*dt/2 # = omega*dt/2
    R = 1/B_mag * np.array([[0, B[2], -B[1]], [-B[2], 0, B[0]], [B[1], -B[0], 0]])
    M_minus = np.eye(3) - R * eps
    M_plus = np.eye(3) + R * eps
    M_inv = np.linalg.inv(M_minus) # matrix inversion
    v = M_inv @ (M_plus @ v_minus_half) + M_inv @ E * q/m * dt
    x = x + v*dt
    
    return x, v

def boris(x, v, E, B, q, m, dt):
    """
    Implements the Boris algorithm for particle pushing.

    Args:
        x (numpy.ndarray): Initial position of the particle (3D vector).
        v (numpy.ndarray): Initial velocity of the particle (3D vector).
        E (numpy.ndarray): Electric field (3D vector).
        B (numpy.ndarray): Magnetic field (3D vector).
        q (float): Charge of the particle.
        m (float): Mass of the particle.
        dt (float): Time step.

    Returns:
        tuple: Updated position and velocity of the particle after the push.
    """
    # Step 1: Half-update of velocity
    v_minus = v + (q * E / m) * (dt / 2)

    # Step 2: Rotation of velocity
    t = (q * B / m) * (dt / 2)
    s = 2 * t / (1 + np.dot(t, t))
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    # Step 3: Half-update of velocity
    v_new = v_plus + (q * E / m) * (dt / 2)

    # Step 4: Update of position
    x_new = x + v_new * dt

    return x_new, v_new

# Example usage
if __name__ == "__main__":
    # Initial conditions
    x = np.array([0.0, 0.0, 0.0])  # Initial position
    v = np.array([1.0, 0.0, 0.0])  # Initial velocity
    a = np.array([0.0, 0.0, -9.8])  # Acceleration (gravity) or a = def... function
    E = np.array([0.0, 0.0, 0.0])  # Electric field
    B = np.array([0.0, 0.0, 1.0])  # Magnetic field
    q = 1.0  # Charge of the particle
    m = 1.0  # Mass of the particle
    dt = 0.1  # Time step