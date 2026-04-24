"""
HOC Core · Grid geometry (private)
==================================

Geometría hexagonal pura y helpers de coordenadas.

Provee ``HexDirection``, ``HexCoord``, ``HexRegion``, ``HexPathfinder`` y
el alias ``HexRing`` (== ``HexRegion``). Todo lo aquí contenido es
*self-contained*: no importa celdas, ni config, ni grid — lo que permite
romper la dependencia circular ``cells ↔ grid`` al split.

Este módulo es *interno*. Los nombres públicos se re-exportan desde
:mod:`hoc.core.grid`.

Extraído de ``core.py`` en Fase 3.3.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing import TypeAlias

__all__ = [
    "HexDirection",
    "HexCoord",
    "HexRegion",
    "HexPathfinder",
    "HexRing",
    "CoordTuple",
    "CubeTuple",
    "PixelTuple",
]

# Type aliases
CoordTuple: TypeAlias = tuple[int, int]
CubeTuple: TypeAlias = tuple[int, int, int]
PixelTuple: TypeAlias = tuple[float, float]


# ═══════════════════════════════════════════════════════════════════════════════
# COORDENADAS HEXAGONALES OPTIMIZADAS
# ═══════════════════════════════════════════════════════════════════════════════


class HexDirection(IntEnum):
    """
    6 direcciones en el grid hexagonal.
    Ordenadas en sentido horario desde arriba-derecha.
    """

    NE = 0  # Noreste   (+1, -1)
    E = 1  # Este      (+1,  0)
    SE = 2  # Sureste   ( 0, +1)
    SW = 3  # Suroeste  (-1, +1)
    W = 4  # Oeste     (-1,  0)
    NW = 5  # Noroeste  ( 0, -1)

    def opposite(self) -> HexDirection:
        """Retorna la dirección opuesta."""
        return HexDirection((self + 3) % 6)

    def rotate_cw(self, steps: int = 1) -> HexDirection:
        """Rotar en sentido horario."""
        return HexDirection((self + steps) % 6)

    def rotate_ccw(self, steps: int = 1) -> HexDirection:
        """Rotar en sentido antihorario."""
        return HexDirection((self - steps) % 6)

    @property
    def vector(self) -> CoordTuple:
        """Vector de dirección (dq, dr)."""
        return _DIRECTION_VECTORS[self]

    @classmethod
    def from_angle(cls, angle_deg: float) -> HexDirection:
        """Obtiene la dirección más cercana a un ángulo."""
        normalized = angle_deg % 360
        index = round(normalized / 60) % 6
        return cls(index)


# Vectores de dirección precalculados (inmutables)
_DIRECTION_VECTORS: Final[tuple[CoordTuple, ...]] = (
    (1, -1),  # NE
    (1, 0),  # E
    (0, 1),  # SE
    (-1, 1),  # SW
    (-1, 0),  # W
    (0, -1),  # NW
)

# Arrays NumPy para operaciones vectoriales
_DIRECTION_ARRAY: Final = np.array(_DIRECTION_VECTORS, dtype=np.int32)


@dataclass(frozen=True, slots=True, order=True)
class HexCoord:
    """
    Coordenada hexagonal axial (q, r) optimizada.

    Inmutable y hasheable para usar como clave de diccionario.
    La tercera coordenada s es implícita: s = -q - r
    """

    q: int
    r: int

    def __post_init__(self):
        """Validación y coerción de tipos."""
        if not isinstance(self.q, int):
            object.__setattr__(self, "q", int(self.q))
        if not isinstance(self.r, int):
            object.__setattr__(self, "r", int(self.r))

    @property
    def s(self) -> int:
        """Tercera coordenada cúbica (implícita)."""
        return -self.q - self.r

    # B10: @cached_property es incompatible con slots=True en dataclass frozen.
    # Las computaciones aquí son O(1) sobre dos enteros, así que @property basta.
    @property
    def cube(self) -> CubeTuple:
        """Retorna coordenadas cúbicas (q, r, s)."""
        return (self.q, self.r, self.s)

    @property
    def array(self) -> NDArray[np.int32]:
        """Representación NumPy para operaciones vectoriales."""
        return np.array([self.q, self.r], dtype=np.int32)

    @property
    def magnitude(self) -> int:
        """Distancia desde el origen."""
        return (abs(self.q) + abs(self.r) + abs(self.s)) // 2

    def __add__(self, other: HexCoord) -> HexCoord:
        if isinstance(other, HexCoord):
            return HexCoord(self.q + other.q, self.r + other.r)
        return NotImplemented

    def __sub__(self, other: HexCoord) -> HexCoord:
        if isinstance(other, HexCoord):
            return HexCoord(self.q - other.q, self.r - other.r)
        return NotImplemented

    def __mul__(self, scalar: int) -> HexCoord:
        if isinstance(scalar, (int, np.integer)):
            return HexCoord(self.q * scalar, self.r * scalar)
        return NotImplemented

    def __rmul__(self, scalar: int) -> HexCoord:
        return self.__mul__(scalar)

    def __neg__(self) -> HexCoord:
        return HexCoord(-self.q, -self.r)

    def __abs__(self) -> int:
        return self.magnitude

    def distance_to(self, other: HexCoord) -> int:
        """Distancia de Manhattan hexagonal."""
        dq = abs(self.q - other.q)
        dr = abs(self.r - other.r)
        ds = abs(self.s - other.s)
        return (dq + dr + ds) // 2

    def neighbor(self, direction: HexDirection) -> HexCoord:
        """Obtiene el vecino en la dirección dada."""
        dq, dr = direction.vector
        return HexCoord(self.q + dq, self.r + dr)

    def neighbors(self) -> tuple[HexCoord, ...]:
        """Retorna los 6 vecinos en orden horario desde NE."""
        return tuple(self.neighbor(d) for d in HexDirection)

    def direction_to(self, other: HexCoord) -> HexDirection | None:
        """Obtiene la dirección hacia otra coordenada adyacente."""
        diff = (other.q - self.q, other.r - self.r)
        for d in HexDirection:
            if d.vector == diff:
                return d
        return None

    def ring(self, radius: int) -> tuple[HexCoord, ...]:
        """Retorna todas las celdas en el anillo a distancia `radius`."""
        return _cached_ring(self.q, self.r, radius)

    def spiral(self, radius: int) -> Iterator[HexCoord]:
        """Genera celdas en espiral desde el centro hasta radio `radius`."""
        yield self
        for ring_r in range(1, radius + 1):
            yield from self.ring(ring_r)

    def filled_hexagon(self, radius: int) -> tuple[HexCoord, ...]:
        """Retorna todas las celdas dentro del radio (inclusive)."""
        return _cached_filled_hex(self.q, self.r, radius)

    def line_to(self, other: HexCoord) -> list[HexCoord]:
        """Genera línea recta desde self hasta other usando interpolación."""
        n = self.distance_to(other)
        if n == 0:
            return [self]

        results = []
        for i in range(n + 1):
            t = i / n
            q = self.q + (other.q - self.q) * t
            r = self.r + (other.r - self.r) * t
            s = self.s + (other.s - self.s) * t
            results.append(_cube_round(q, r, s))

        return results

    def lerp(self, other: HexCoord, t: float) -> HexCoord:
        """Interpolación lineal entre dos coordenadas."""
        q = self.q + (other.q - self.q) * t
        r = self.r + (other.r - self.r) * t
        s = self.s + (other.s - self.s) * t
        return _cube_round(q, r, s)

    def rotate_around(self, center: HexCoord, steps: int = 1) -> HexCoord:
        """Rota esta coordenada alrededor de un centro (steps * 60°)."""
        vec = self - center
        q, r, s = vec.q, vec.r, vec.s

        for _ in range(steps % 6):
            q, r, s = -r, -s, -q

        return center + HexCoord(q, r)

    def reflect_across(self, axis: HexDirection) -> HexCoord:
        """Refleja la coordenada a través de un eje."""
        q, r, s = self.q, self.r, self.s

        if axis in (HexDirection.E, HexDirection.W):
            return HexCoord(q, s)
        elif axis in (HexDirection.NE, HexDirection.SW):
            return HexCoord(s, r)
        else:  # NW, SE
            return HexCoord(r, q)

    def to_pixel(self, size: float = 1.0, orientation: str = "flat") -> PixelTuple:
        """Convierte a coordenadas de pixel."""
        if orientation == "flat":
            x = size * (3 / 2 * self.q)
            y = size * (math.sqrt(3) / 2 * self.q + math.sqrt(3) * self.r)
        else:
            x = size * (math.sqrt(3) * self.q + math.sqrt(3) / 2 * self.r)
            y = size * (3 / 2 * self.r)
        return (x, y)

    @classmethod
    def from_pixel(
        cls, x: float, y: float, size: float = 1.0, orientation: str = "flat"
    ) -> HexCoord:
        """Convierte coordenadas de pixel a hexagonal."""
        if orientation == "flat":
            q = (2 / 3 * x) / size
            r = (-1 / 3 * x + math.sqrt(3) / 3 * y) / size
        else:
            q = (math.sqrt(3) / 3 * x - 1 / 3 * y) / size
            r = (2 / 3 * y) / size

        return _cube_round(q, r, -q - r)

    @classmethod
    def origin(cls) -> HexCoord:
        """Retorna el origen (0, 0)."""
        return _ORIGIN

    def to_dict(self) -> dict[str, int]:
        """Serializa a diccionario."""
        return {"q": self.q, "r": self.r}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> HexCoord:
        """Deserializa desde diccionario."""
        return cls(data["q"], data["r"])


# Constante para origen
_ORIGIN = HexCoord(0, 0)


@lru_cache(maxsize=1024)
def _cached_ring(q: int, r: int, radius: int) -> tuple[HexCoord, ...]:
    """Genera anillo con cache."""
    if radius == 0:
        return (HexCoord(q, r),)

    center = HexCoord(q, r)
    results = []

    current = center + HexCoord(-radius, 0)

    for direction in HexDirection:
        for _ in range(radius):
            results.append(current)
            current = current.neighbor(direction)

    return tuple(results)


@lru_cache(maxsize=256)
def _cached_filled_hex(q: int, r: int, radius: int) -> tuple[HexCoord, ...]:
    """
    Genera hexágono relleno con cache.

    v3.0 FIX: Renombrado variable de loop a `ring_r` para evitar
    shadowing del parámetro `r`.
    """
    center = HexCoord(q, r)
    results = [center]
    for ring_r in range(1, radius + 1):
        results.extend(center.ring(ring_r))
    return tuple(results)


def _cube_round(q: float, r: float, s: float) -> HexCoord:
    """Redondea coordenadas cúbicas flotantes al hexágono más cercano."""
    rq = round(q)
    rr = round(r)
    rs = round(s)

    q_diff = abs(rq - q)
    r_diff = abs(rr - r)
    s_diff = abs(rs - s)

    if q_diff > r_diff and q_diff > s_diff:
        rq = -rr - rs
    elif r_diff > s_diff:
        rr = -rq - rs

    return HexCoord(int(rq), int(rr))


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE COORDENADAS
# ═══════════════════════════════════════════════════════════════════════════════


class HexRegion:
    """
    Región de coordenadas hexagonales para operaciones en lote.
    """

    __slots__ = ("_bounds", "_coord_set", "_coords")

    def __init__(self, coords: Sequence[HexCoord]):
        self._coords = tuple(coords)
        self._coord_set = frozenset(coords)
        self._bounds: tuple[int, int, int, int] | None = None

    @classmethod
    def from_ring(cls, center: HexCoord, radius: int) -> HexRegion:
        return cls(center.ring(radius))

    @classmethod
    def from_area(cls, center: HexCoord, radius: int) -> HexRegion:
        return cls(center.filled_hexagon(radius))

    @classmethod
    def from_line(cls, start: HexCoord, end: HexCoord) -> HexRegion:
        return cls(start.line_to(end))

    def __contains__(self, coord: HexCoord) -> bool:
        return coord in self._coord_set

    def __iter__(self) -> Iterator[HexCoord]:
        return iter(self._coords)

    def __len__(self) -> int:
        return len(self._coords)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Retorna (min_q, max_q, min_r, max_r)."""
        if self._bounds is None:
            if not self._coords:
                self._bounds = (0, 0, 0, 0)
            else:
                qs = [c.q for c in self._coords]
                rs = [c.r for c in self._coords]
                self._bounds = (min(qs), max(qs), min(rs), max(rs))
        return self._bounds

    def union(self, other: HexRegion) -> HexRegion:
        return HexRegion(list(self._coord_set | other._coord_set))

    def intersection(self, other: HexRegion) -> HexRegion:
        return HexRegion(list(self._coord_set & other._coord_set))

    def difference(self, other: HexRegion) -> HexRegion:
        return HexRegion(list(self._coord_set - other._coord_set))

    @property
    def centroid(self) -> HexCoord:
        """v3.0: Centro geométrico aproximado de la región."""
        if not self._coords:
            return _ORIGIN
        avg_q = sum(c.q for c in self._coords) / len(self._coords)
        avg_r = sum(c.r for c in self._coords) / len(self._coords)
        return _cube_round(avg_q, avg_r, -avg_q - avg_r)


# Alias para compatibilidad con HOC.__init__ y métricas (anillo = región de un anillo)
HexRing = HexRegion


class HexPathfinder:
    """
    Pathfinding A* optimizado para grids hexagonales.

    v3.0: Soporte para costos variables por celda.
    """

    def __init__(
        self,
        walkable_check: Callable[[HexCoord], bool],
        cost_fn: Callable[[HexCoord], float] | None = None,
    ):
        self._walkable = walkable_check
        self._cost_fn = cost_fn or (lambda _: 1.0)

    def find_path(
        self, start: HexCoord, goal: HexCoord, max_iterations: int = 10000
    ) -> list[HexCoord] | None:
        """
        Encuentra el camino más corto usando A*.

        v3.0: Soporta costos variables por celda via cost_fn.
        """
        if start == goal:
            return [start]

        if not self._walkable(goal):
            return None

        open_set: list[tuple[float, int, HexCoord]] = [(0.0, 0, start)]
        came_from: dict[HexCoord, HexCoord] = {}
        g_score: dict[HexCoord, float] = {start: 0.0}

        counter = 0
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            current_g = g_score.get(current, float("inf"))

            for neighbor in current.neighbors():
                if not self._walkable(neighbor):
                    continue

                move_cost = self._cost_fn(neighbor)
                tentative_g = current_g + move_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + neighbor.distance_to(goal)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

        return None

    def _reconstruct_path(
        self, came_from: dict[HexCoord, HexCoord], current: HexCoord
    ) -> list[HexCoord]:
        """Reconstruye el camino desde came_from."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
