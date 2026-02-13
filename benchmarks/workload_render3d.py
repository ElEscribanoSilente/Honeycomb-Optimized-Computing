"""
Carga de trabajo pesada: mini render 3D por raycasting (NumPy).
Usado para probar el SwarmScheduler con tareas CPU-intensivas.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


def _ray_sphere_intersect(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intersección rayo-esfera vectorizada.
    ray_origin: (N,3) ray_dir: (N,3) center: (3,) -> t0 (N,) t1 (N,)
    """
    oc = ray_origin - center
    a = (ray_dir * ray_dir).sum(axis=1)
    b = 2.0 * (oc * ray_dir).sum(axis=1)
    c = (oc * oc).sum(axis=1) - radius * radius
    disc = b * b - 4 * a * c
    mask = disc >= 0
    sqrt_d = np.sqrt(np.maximum(disc, 0))
    denom = 2.0 * np.maximum(a, 1e-8)
    t0 = (-b - sqrt_d) / denom
    t1 = (-b + sqrt_d) / denom
    return np.where(mask, t0, np.inf), np.where(mask, t1, np.inf)


def _render_tile(
    width: int,
    height: int,
    tile_y_start: int,
    tile_y_end: int,
    tile_x_start: int,
    tile_x_end: int,
    cam_pos: np.ndarray,
    spheres: List[Tuple[np.ndarray, float, np.ndarray]],
    samples_per_pixel: int = 2,
) -> np.ndarray:
    """
    Renderiza un tile (porción) de la imagen por raycasting.
    Devuelve un array 2D (tile_h, tile_w) con valores 0..1.
    """
    tile_h = tile_y_end - tile_y_start
    tile_w = tile_x_end - tile_x_start
    out = np.zeros((tile_h, tile_w), dtype=np.float32)
    # Coordenadas de píxel del tile en espacio imagen [-1,1]
    y = np.linspace(1.0, -1.0, tile_h)
    x = np.linspace(-1.0, 1.0, tile_w)
    xx, yy = np.meshgrid(x, y)
    # Dirección base del rayo (plano z=-1)
    aspect = width / max(height, 1)
    rx = xx * aspect
    ry = yy
    rz = np.full_like(rx, -1.0)
    ray_dir = np.stack([rx, ry, rz], axis=-1)
    n = ray_dir.shape[0] * ray_dir.shape[1]
    ray_dir_flat = ray_dir.reshape(n, 3)
    norm = np.linalg.norm(ray_dir_flat, axis=1, keepdims=True)
    ray_dir_flat = ray_dir_flat / np.maximum(norm, 1e-8)
    ray_origin_flat = np.tile(cam_pos, (n, 1))
    # Múltiples muestras por píxel para hacer el trabajo más pesado
    for _ in range(samples_per_pixel):
        t_min = np.full(n, np.inf)
        for center, radius, color in spheres:
            t0, t1 = _ray_sphere_intersect(
                ray_origin_flat, ray_dir_flat, center, radius
            )
            hit = (t0 > 0.01) & (t0 < t_min)
            t_min = np.where(hit, t0, t_min)
        # Shading simple: distancia como atenuación
        intensity = np.where(np.isfinite(t_min), 1.0 / (1.0 + t_min * 0.1), 0.0)
        out += intensity.reshape(tile_h, tile_w)
    out /= samples_per_pixel
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def mini_render_3d(
    width: int = 128,
    height: int = 128,
    num_spheres: int = 8,
    samples_per_pixel: int = 4,
) -> np.ndarray:
    """
    Mini render 3D: raycasting a varias esferas.
    Carga CPU moderada-alta (NumPy vectorizado).
    """
    cam_pos = np.array([0.0, 0.0, 2.0], dtype=np.float64)
    np.random.seed(42)
    spheres: List[Tuple[np.ndarray, float, np.ndarray]] = []
    for _ in range(num_spheres):
        center = np.random.uniform(-0.8, 0.8, 3)
        center[2] = np.random.uniform(-1.5, 0.0)
        radius = np.random.uniform(0.1, 0.35)
        color = np.random.uniform(0.2, 1.0, 3)
        spheres.append((center, radius, color))
    return _render_tile(0, height, 0, height, 0, width, cam_pos, spheres, samples_per_pixel)


def mini_render_3d_tile(
    width: int,
    height: int,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    num_spheres: int = 6,
    samples_per_pixel: int = 3,
) -> np.ndarray:
    """
    Renderiza un solo tile. Para repartir el frame entre tareas del swarm.
    """
    cam_pos = np.array([0.0, 0.0, 2.0], dtype=np.float64)
    np.random.seed(42)
    spheres: List[Tuple[np.ndarray, float, np.ndarray]] = []
    for _ in range(num_spheres):
        center = np.random.uniform(-0.8, 0.8, 3)
        center[2] = np.random.uniform(-1.5, 0.0)
        radius = np.random.uniform(0.1, 0.35)
        color = np.random.uniform(0.2, 1.0, 3)
        spheres.append((center, radius, color))
    return _render_tile(
        width, height, y_start, y_end, x_start, x_end,
        cam_pos, spheres, samples_per_pixel,
    )


def save_render(image: np.ndarray, path: str | Path) -> Path:
    """
    Guarda el buffer de render (float 0..1) como imagen.
    Usa PGM (sin dependencias) o PNG si hay matplotlib.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    try:
        import matplotlib.pyplot as plt
        plt.imsave(str(path), img_u8, cmap="gray")
        return path
    except ImportError:
        pass

    # PGM (Portable Graymap): cualquier visor de imágenes lo abre
    with open(path.with_suffix(".pgm"), "wb") as f:
        h, w = img_u8.shape
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(img_u8.tobytes())
    return path.with_suffix(".pgm")


def render_and_save(
    width: int = 256,
    height: int = 256,
    num_spheres: int = 8,
    samples_per_pixel: int = 4,
    out_path: str | Path = "render_esferas.png",
) -> Path:
    """Genera un mini render 3D y lo guarda en disco. Devuelve la ruta del archivo."""
    img = mini_render_3d(width, height, num_spheres, samples_per_pixel)
    return save_render(img, out_path)


if __name__ == "__main__":
    import sys
    out = Path(__file__).resolve().parent.parent / "render_esferas.png"
    path = render_and_save(
        width=320,
        height=240,
        num_spheres=8,
        samples_per_pixel=4,
        out_path=out,
    )
    print(f"Render guardado: {path}")
    print("Ábrelo con tu visor de imágenes para ver las esferas.")
    sys.exit(0)
