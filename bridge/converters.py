"""
HOC Bridge — coordinate converters between hex and cartesian space.

Phase 6 split from the legacy ``bridge.py`` monolith. Public API preserved
through ``hoc.bridge.__init__`` (Phase 3 split pattern: ``core.py`` →
``core/``, ``metrics.py`` → ``metrics/``).
"""

from __future__ import annotations

import math
from enum import Enum, auto

from ..core import HexCoord


class HexToCartesian:
    """
    Conversor de coordenadas hexagonales a cartesianas.

    Soporta dos layouts:
    - FLAT_TOP: Hexágonos con lado plano arriba
    - POINTY_TOP: Hexágonos con vértice arriba

    Uso:
        converter = HexToCartesian(size=1.0, layout=HexLayout.FLAT_TOP)
        x, y = converter.convert(HexCoord(2, 3))
        center = converter.center(HexCoord(0, 0))
    """

    class Layout(Enum):
        FLAT_TOP = auto()
        POINTY_TOP = auto()

    def __init__(
        self,
        size: float = 1.0,
        layout: HexToCartesian.Layout = None,
        origin: tuple[float, float] = (0.0, 0.0),
    ):
        self.size = size
        self.layout = layout or self.Layout.FLAT_TOP
        self.origin = origin

    def convert(self, coord: HexCoord) -> tuple[float, float]:
        """
        Convierte coordenada hexagonal a cartesiana.

        Args:
            coord: Coordenada hexagonal (q, r)

        Returns:
            Tupla (x, y) en coordenadas cartesianas
        """
        if self.layout == self.Layout.FLAT_TOP:
            x = self.size * (3 / 2 * coord.q)
            y = self.size * (math.sqrt(3) / 2 * coord.q + math.sqrt(3) * coord.r)
        else:  # POINTY_TOP
            x = self.size * (math.sqrt(3) * coord.q + math.sqrt(3) / 2 * coord.r)
            y = self.size * (3 / 2 * coord.r)

        return (x + self.origin[0], y + self.origin[1])

    def center(self, coord: HexCoord) -> tuple[float, float]:
        """Obtiene el centro de un hexágono en coordenadas cartesianas."""
        return self.convert(coord)

    def corners(self, coord: HexCoord) -> list[tuple[float, float]]:
        """
        Obtiene las 6 esquinas de un hexágono.

        Returns:
            Lista de 6 tuplas (x, y) en orden horario
        """
        cx, cy = self.center(coord)
        corners = []

        for i in range(6):
            if self.layout == self.Layout.FLAT_TOP:
                angle = math.pi / 3 * i
            else:
                angle = math.pi / 3 * i + math.pi / 6

            x = cx + self.size * math.cos(angle)
            y = cy + self.size * math.sin(angle)
            corners.append((x, y))

        return corners

    def bounding_box(self, coord: HexCoord) -> tuple[float, float, float, float]:
        """
        Obtiene el bounding box de un hexágono.

        Returns:
            Tupla (min_x, min_y, max_x, max_y)
        """
        corners = self.corners(coord)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return (min(xs), min(ys), max(xs), max(ys))


class CartesianToHex:
    """
    Conversor de coordenadas cartesianas a hexagonales.

    Uso:
        converter = CartesianToHex(size=1.0)
        coord = converter.convert(3.5, 2.1)
        nearest = converter.nearest(3.5, 2.1)
    """

    def __init__(
        self,
        size: float = 1.0,
        layout: HexToCartesian.Layout = None,
        origin: tuple[float, float] = (0.0, 0.0),
    ):
        self.size = size
        self.layout = layout or HexToCartesian.Layout.FLAT_TOP
        self.origin = origin

    def convert(self, x: float, y: float) -> HexCoord:
        """
        Convierte coordenadas cartesianas a hexagonales.

        Args:
            x: Coordenada X
            y: Coordenada Y

        Returns:
            Coordenada hexagonal más cercana
        """
        # Ajustar por origen
        x = x - self.origin[0]
        y = y - self.origin[1]

        if self.layout == HexToCartesian.Layout.FLAT_TOP:
            q = (2 / 3 * x) / self.size
            r = (-1 / 3 * x + math.sqrt(3) / 3 * y) / self.size
        else:  # POINTY_TOP
            q = (math.sqrt(3) / 3 * x - 1 / 3 * y) / self.size
            r = (2 / 3 * y) / self.size

        return self._axial_round(q, r)

    def nearest(self, x: float, y: float) -> HexCoord:
        """Alias para convert()."""
        return self.convert(x, y)

    def _axial_round(self, q: float, r: float) -> HexCoord:
        """Redondea coordenadas axiales flotantes."""
        s = -q - r

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

    def in_hexagon(self, x: float, y: float, coord: HexCoord) -> bool:
        """
        Verifica si un punto está dentro de un hexágono.

        Args:
            x, y: Punto a verificar
            coord: Hexágono a verificar

        Returns:
            True si el punto está dentro del hexágono
        """
        nearest = self.convert(x, y)
        return nearest == coord


__all__ = [
    "HexToCartesian",
    "CartesianToHex",
]
