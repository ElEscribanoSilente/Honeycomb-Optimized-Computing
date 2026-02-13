"""
Cargas de trabajo pesadas variadas para benchmarks y tests de estrés.

Tipos:
- render_3d: raycasting (NumPy, memoria + CPU)
- matrix: multiplicación de matrices grandes (NumPy, CPU/ cache)
- simulation: muchos pasos de simulación (bucles + NumPy)
- hash_work: trabajo estilo hash iterado (CPU puro, poco memoria)
- monte_carlo: muestreo aleatorio y reducción (NumPy)
- math_*: tareas matemáticas complejas (autovalores, FFT, integración, etc.)
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Any, Dict, List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ─── Matrix (CPU + memoria) ─────────────────────────────────────────────────

def workload_matrix_mult(
    size: int = 512,
    iterations: int = 4,
) -> float:
    """Multiplicación de matrices NxN repetida. Carga CPU y uso de cache."""
    np.random.seed(42)
    a = np.random.standard_normal((size, size)).astype(np.float64)
    b = np.random.standard_normal((size, size)).astype(np.float64)
    acc = np.eye(size, dtype=np.float64)
    for _ in range(iterations):
        acc = acc @ a @ b
    return float(np.sum(acc))


def workload_matrix_svd(size: int = 256, n_components: int = 32) -> float:
    """SVD reducida sobre matriz aleatoria. Coste numérico alto."""
    np.random.seed(43)
    m = np.random.standard_normal((size, size)).astype(np.float64)
    u, s, vh = np.linalg.svd(m, full_matrices=False)
    return float(np.sum(s[:n_components]))


# ─── Simulación (bucles + NumPy) ─────────────────────────────────────────────

def workload_simulation_steps(
    steps: int = 2000,
    state_size: int = 128,
) -> float:
    """Muchos pasos de una simulación simple (estilo integrador)."""
    np.random.seed(44)
    state = np.random.standard_normal(state_size).astype(np.float64)
    A = np.random.standard_normal((state_size, state_size)) * 0.01
    for _ in range(steps):
        state = state + A @ state
        state = np.tanh(state)
    return float(np.sum(state))


# ─── Trabajo estilo hash (CPU, poco memoria) ───────────────────────────────────

def workload_hash_like(
    data_size: int = 64 * 1024,
    rounds: int = 5000,
) -> int:
    """Iteraciones tipo hash sobre un buffer (solo CPU, sin NumPy pesado)."""
    data = bytearray(b"x" * data_size)
    h = 0x811C9DC5
    for _ in range(rounds):
        for i in range(0, len(data) - 4, 4):
            w = (
                data[i]
                | (data[i + 1] << 8)
                | (data[i + 2] << 16)
                | (data[i + 3] << 24)
            )
            h = ((h ^ w) * 0x01000193) & 0xFFFFFFFF
        data[0] = (h & 0xFF)
        data[1] = ((h >> 8) & 0xFF)
    return h


# ─── Monte Carlo ────────────────────────────────────────────────────────────

def workload_monte_carlo(
    num_samples: int = 500_000,
    ndim: int = 6,
) -> float:
    """Muestreo aleatorio y reducción (estilo estimación de integral)."""
    np.random.seed(45)
    x = np.random.standard_normal((num_samples, ndim))
    r = np.sqrt(np.sum(x * x, axis=1))
    inside = np.sum(r < 1.0)
    return float(inside) / num_samples


# ─── Tareas matemáticas complejas ─────────────────────────────────────────────

def workload_math_eigen(size: int = 180, n_eig: int = 20) -> float:
    """Autovalores de matriz simétrica definida positiva (problema de valores propios)."""
    np.random.seed(46)
    A = np.random.standard_normal((size, size)).astype(np.float64)
    A = A.T @ A + size * np.eye(size)
    w, _ = np.linalg.eigh(A)
    return float(np.sum(w[-n_eig:]))


def workload_math_fft(signal_size: int = 2**18, n_ffts: int = 50) -> float:
    """Muchas FFT sobre señales largas (análisis espectral)."""
    np.random.seed(47)
    x = np.random.standard_normal(signal_size).astype(np.float64)
    acc = 0.0
    for _ in range(n_ffts):
        X = np.fft.rfft(x)
        acc += float(np.sum(np.abs(X)))
        x = np.roll(x, 1)
    return acc


def workload_math_integrate(n_samples: int = 200_000, ndim: int = 4) -> float:
    """Integración numérica por Monte Carlo de función costosa (estilo alta dimensión)."""
    np.random.seed(48)
    x = np.random.uniform(0, 1, (n_samples, ndim))
    # Función tipo Rastrigin/oscillatoria (muchas evaluciones)
    y = np.sum(np.cos(2 * np.pi * x) * (x ** 2), axis=1) + np.prod(x, axis=1)
    return float(np.mean(y))


def workload_math_solve(size: int = 320, n_rhs: int = 8) -> float:
    """Resolver sistemas lineales Ax=b (Cholesky + sustituciones) para matriz SPD."""
    np.random.seed(49)
    A = np.random.standard_normal((size, size)).astype(np.float64)
    A = A.T @ A + size * np.eye(size)
    B = np.random.standard_normal((size, n_rhs)).astype(np.float64)
    L = np.linalg.cholesky(A)
    # L y = B, luego L.T x = y
    Y = np.linalg.solve(L, B)
    X = np.linalg.solve(L.T, Y)
    return float(np.sum(X))


def workload_math_poly_roots(degree: int = 120, n_polys: int = 3) -> float:
    """Raíces de polinomios de grado alto (compañero + autovalores)."""
    np.random.seed(50)
    acc = 0.0
    for _ in range(n_polys):
        coef = np.random.standard_normal(degree + 1).astype(np.float64)
        coef[0] = 1.0
        roots = np.roots(coef)
        acc += float(np.sum(np.real(roots)))
    return acc


# ─── Render 3D (delegado) ────────────────────────────────────────────────────

def workload_render_3d(
    width: int = 96,
    height: int = 96,
    num_spheres: int = 6,
    samples_per_pixel: int = 3,
) -> float:
    """Mini render 3D; retorna suma del buffer para validar."""
    from benchmarks.workload_render3d import mini_render_3d
    img = mini_render_3d(width, height, num_spheres, samples_per_pixel)
    return float(np.sum(img))


# ─── Registro de cargas (para benchmark mixto) ───────────────────────────────

WORKLOADS: Dict[str, Callable[[], Any]] = {
    "render_3d": lambda: workload_render_3d(96, 96, 6, 3),
    "matrix_mult": lambda: workload_matrix_mult(384, 3),
    "matrix_svd": lambda: workload_matrix_svd(200, 24),
    "simulation": lambda: workload_simulation_steps(1500, 96),
    "hash_work": lambda: workload_hash_like(32 * 1024, 3000),
    "monte_carlo": lambda: workload_monte_carlo(300_000, 5),
    # Tareas matemáticas complejas
    "math_eigen": lambda: workload_math_eigen(160, 16),
    "math_fft": lambda: workload_math_fft(2**17, 30),
    "math_integrate": lambda: workload_math_integrate(150_000, 4),
    "math_solve": lambda: workload_math_solve(280, 6),
    "math_poly_roots": lambda: workload_math_poly_roots(100, 2),
}


def run_workload(name: str) -> Any:
    """Ejecuta una carga por nombre. Lanza KeyError si no existe."""
    return WORKLOADS[name]()
