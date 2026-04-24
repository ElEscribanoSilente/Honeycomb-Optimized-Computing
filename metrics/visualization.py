"""
HOC Metrics · Visualization — ASCII + SVG + HTML del grid hexagonal.

Contiene :class:`ColorScheme` y :class:`HoneycombVisualizer`. Extraído de
``metrics.py`` en Fase 3.3 (continuación).
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar

from ..core.cells_base import CellRole, CellState
from ..core.grid_geometry import HexCoord

if TYPE_CHECKING:
    from ..core.cells_base import HoneycombCell
    from ..core.grid import HoneycombGrid

__all__ = ["ColorScheme", "HoneycombVisualizer"]


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════


class ColorScheme(Enum):
    """Esquemas de color para visualización."""

    LOAD = auto()  # Por carga (verde → rojo)
    STATE = auto()  # Por estado
    ROLE = auto()  # Por rol
    PHEROMONE = auto()  # Por nivel de feromona
    ACTIVITY = auto()  # Por actividad reciente


class HoneycombVisualizer:
    """
    Visualizador del grid hexagonal.

    Renderiza el panal en diferentes formatos:
    - ASCII art
    - SVG
    - HTML interactivo

    Uso:
        viz = HoneycombVisualizer(grid)
        print(viz.render_ascii())
        svg = viz.render_svg()
    """

    # Caracteres para renderizado ASCII
    ASCII_CHARS: ClassVar[dict[CellRole, str]] = {
        CellRole.QUEEN: "👑",
        CellRole.WORKER: "⬡",
        CellRole.DRONE: "🐝",
        CellRole.NURSERY: "🥚",
        CellRole.STORAGE: "📦",
        CellRole.GUARD: "🛡",
        CellRole.SCOUT: "🔍",
    }

    LOAD_CHARS: ClassVar[list[str]] = ["⬡", "🟢", "🟡", "🟠", "🔴"]

    STATE_CHARS: ClassVar[dict[CellState, str]] = {
        CellState.EMPTY: "○",
        CellState.ACTIVE: "●",
        CellState.IDLE: "◐",
        CellState.SPAWNING: "◉",
        CellState.MIGRATING: "↔",
        CellState.FAILED: "✗",
        CellState.RECOVERING: "↻",
        CellState.SEALED: "▣",
    }

    def __init__(self, grid: HoneycombGrid):
        self.grid = grid
        self._color_scheme = ColorScheme.LOAD

    def set_color_scheme(self, scheme: ColorScheme) -> None:
        """Establece el esquema de color."""
        self._color_scheme = scheme

    def render_ascii(self, scheme: ColorScheme | None = None, show_coords: bool = False) -> str:
        """
        Renderiza el grid como ASCII art.

        Args:
            scheme: Esquema de color a usar
            show_coords: Mostrar coordenadas

        Returns:
            String con el grid renderizado
        """
        scheme = scheme or self._color_scheme
        lines = []

        radius = self.grid.config.radius

        for r in range(-radius, radius + 1):
            # Offset para alineación hexagonal
            indent = " " * abs(r)
            row = []

            for q in range(-radius, radius + 1):
                coord = HexCoord(q, r)

                if coord in self.grid._cells:
                    cell = self.grid._cells[coord]
                    char = self._get_cell_char(cell, scheme)

                    if show_coords:
                        char = f"{char}({q},{r})"

                    row.append(char)
                else:
                    row.append("  ")

            lines.append(indent + " ".join(row))

        return "\n".join(lines)

    def _get_cell_char(self, cell: HoneycombCell, scheme: ColorScheme) -> str:
        """Obtiene el carácter para una celda según el esquema."""
        if scheme == ColorScheme.ROLE:
            return self.ASCII_CHARS.get(cell.role, "⬡")

        elif scheme == ColorScheme.STATE:
            return self.STATE_CHARS.get(cell.state, "?")

        elif scheme == ColorScheme.LOAD:
            load_idx = min(int(cell.load * len(self.LOAD_CHARS)), len(self.LOAD_CHARS) - 1)
            return self.LOAD_CHARS[load_idx]

        elif scheme == ColorScheme.PHEROMONE:
            if cell.pheromone_level > 0.7:
                return "🔥"
            elif cell.pheromone_level > 0.3:
                return "🌡"
            else:
                return "❄"

        elif scheme == ColorScheme.ACTIVITY:
            if cell._last_activity > 0.5:
                return "⚡"
            else:
                return "💤"

        return "⬡"

    def render_svg(
        self, width: int = 800, height: int = 600, scheme: ColorScheme | None = None
    ) -> str:
        """
        Renderiza el grid como SVG.

        Args:
            width: Ancho del SVG
            height: Alto del SVG
            scheme: Esquema de color

        Returns:
            String con SVG
        """
        scheme = scheme or self._color_scheme

        # Calcular escala
        radius = self.grid.config.radius
        hex_size = min(width, height) / (radius * 4 + 2)
        center_x = width / 2
        center_y = height / 2

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="#1a1a2e"/>',
        ]

        for coord, cell in self.grid._cells.items():
            # Convertir a coordenadas pixel
            x = center_x + hex_size * (3 / 2 * coord.q)
            y = center_y + hex_size * (math.sqrt(3) / 2 * coord.q + math.sqrt(3) * coord.r)

            # Color según esquema
            color = self._get_cell_color(cell, scheme)

            # Generar hexágono
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                px = x + hex_size * 0.9 * math.cos(angle)
                py = y + hex_size * 0.9 * math.sin(angle)
                points.append(f"{px:.1f},{py:.1f}")

            svg_parts.append(
                f'<polygon points="{" ".join(points)}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            )

            # Etiqueta opcional
            if cell.role == CellRole.QUEEN:
                svg_parts.append(
                    f'<text x="{x}" y="{y}" text-anchor="middle" '
                    f'dominant-baseline="central" fill="white" font-size="12">👑</text>'
                )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def _get_cell_color(self, cell: HoneycombCell, scheme: ColorScheme) -> str:
        """Obtiene el color para una celda según el esquema."""
        if scheme == ColorScheme.LOAD:
            # Verde a rojo según carga
            r = int(255 * cell.load)
            g = int(255 * (1 - cell.load))
            return f"rgb({r},{g},100)"

        elif scheme == ColorScheme.STATE:
            colors = {
                CellState.EMPTY: "#333333",
                CellState.ACTIVE: "#00ff00",
                CellState.IDLE: "#888888",
                CellState.SPAWNING: "#ffff00",
                CellState.MIGRATING: "#00ffff",
                CellState.FAILED: "#ff0000",
                CellState.RECOVERING: "#ff8800",
                CellState.SEALED: "#0000ff",
            }
            return colors.get(cell.state, "#ffffff")

        elif scheme == ColorScheme.ROLE:
            colors = {
                CellRole.QUEEN: "#ffd700",
                CellRole.WORKER: "#4a90d9",
                CellRole.DRONE: "#ff6600",
                CellRole.NURSERY: "#ff69b4",
                CellRole.STORAGE: "#808080",
                CellRole.GUARD: "#8b0000",
                CellRole.SCOUT: "#00ced1",
            }
            return colors.get(cell.role, "#ffffff")

        elif scheme == ColorScheme.PHEROMONE:
            intensity = min(cell.pheromone_level, 1.0)
            return f"rgb({int(255*intensity)},100,{int(255*(1-intensity))})"

        return "#ffffff"

    def render_html(self, scheme: ColorScheme | None = None) -> str:
        """Renderiza como HTML interactivo."""
        svg = self.render_svg(scheme=scheme)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>HOC Honeycomb Visualizer</title>
    <style>
        body {{
            background: #1a1a2e;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }}
        svg polygon:hover {{
            stroke-width: 3;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    {svg}
</body>
</html>
"""
        return html
