import numpy as np


def legendre_p2(x: np.ndarray) -> np.ndarray:
    return 0.5 * (3.0 * x**2 - 1.0)


def weight_p2(theta: np.ndarray, beta: float) -> np.ndarray:
    """
    Axisymmetric quadrupole anisotropy:
        W(theta) = 1 + beta * P2(cos theta)
    """
    return 1.0 + beta * legendre_p2(np.cos(theta))


def weight_p2_plus_dipole(theta: np.ndarray, phi: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """
    Non-axisymmetric extension:
        W(theta, phi) = 1 + beta P2(cos theta) + gamma sin(theta) cos(phi)
    """
    return 1.0 + beta * legendre_p2(np.cos(theta)) + gamma * np.sin(theta) * np.cos(phi)


def validate_nonnegative(weight: np.ndarray, tol: float = 1e-12) -> bool:
    return np.min(weight) >= -tol