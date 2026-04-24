"""
HOC Metrics · Rendering — mapas de calor y flujos de comunicación.

Contiene :class:`HeatmapRenderer` y :class:`FlowVisualizer` (SVG + HTML).
Extraído de ``metrics.py`` en Fase 3.3 (continuación).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from .visualization import HoneycombVisualizer

if TYPE_CHECKING:
    from ..core.grid import HoneycombGrid
    from ..core.grid_geometry import HexCoord

__all__ = ["HeatmapRenderer", "FlowVisualizer"]


class HeatmapRenderer:
    """
    Renderiza mapas de calor del panal.

    Visualiza distribución de:
    - Carga
    - Feromonas
    - Errores
    - Actividad

    Uso:
        heatmap = HeatmapRenderer(grid)
        svg = heatmap.render("load")
    """

    def __init__(self, grid: HoneycombGrid):
        self.grid = grid

    def render(self, metric: str = "load", width: int = 600, height: int = 600) -> str:
        """
        Renderiza mapa de calor.

        Args:
            metric: Métrica a visualizar (load, pheromone, errors)
            width: Ancho
            height: Alto

        Returns:
            SVG string
        """
        # Obtener valores
        values = {}
        for coord, cell in self.grid._cells.items():
            if metric == "load":
                values[coord] = cell.load
            elif metric == "pheromone":
                values[coord] = cell.pheromone_level
            elif metric == "errors":
                values[coord] = min(cell._error_count / 10, 1.0)
            else:
                values[coord] = 0.0

        # Normalizar
        max_val = max(values.values()) if values else 1.0
        if max_val > 0:
            values = {k: v / max_val for k, v in values.items()}

        # Crear visualizador temporal con colores personalizados
        HoneycombVisualizer(self.grid)

        # Renderizar con color personalizado basado en valores
        return self._render_heatmap_svg(values, width, height)

    def _render_heatmap_svg(self, values: dict[HexCoord, float], width: int, height: int) -> str:
        """Renderiza SVG de mapa de calor."""
        radius = self.grid.config.radius
        hex_size = min(width, height) / (radius * 4 + 2)
        center_x = width / 2
        center_y = height / 2

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="#000000"/>',
        ]

        for coord, value in values.items():
            # Convertir a coordenadas pixel
            x = center_x + hex_size * (3 / 2 * coord.q)
            y = center_y + hex_size * (math.sqrt(3) / 2 * coord.q + math.sqrt(3) * coord.r)

            # Color de mapa de calor (azul → rojo)
            r = int(255 * value)
            b = int(255 * (1 - value))
            color = f"rgb({r},0,{b})"

            # Generar hexágono
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                px = x + hex_size * 0.9 * math.cos(angle)
                py = y + hex_size * 0.9 * math.sin(angle)
                points.append(f"{px:.1f},{py:.1f}")

            svg_parts.append(
                f'<polygon points="{" ".join(points)}" '
                f'fill="{color}" stroke="#333333" stroke-width="0.5"/>'
            )

        # Leyenda
        svg_parts.append(self._render_legend(width, height))

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def _render_legend(self, width: int, height: int) -> str:
        """Renderiza leyenda del mapa de calor."""
        legend_width = 20
        legend_height = 100
        x = width - legend_width - 20
        y = (height - legend_height) / 2

        parts = []

        # Gradiente
        steps = 20
        for i in range(steps):
            value = i / steps
            r = int(255 * value)
            b = int(255 * (1 - value))
            step_y = y + legend_height * (1 - i / steps)
            step_height = legend_height / steps + 1
            parts.append(
                f'<rect x="{x}" y="{step_y}" width="{legend_width}" '
                f'height="{step_height}" fill="rgb({r},0,{b})"/>'
            )

        # Labels
        parts.append(
            f'<text x="{x + legend_width + 5}" y="{y + 10}" '
            f'fill="white" font-size="10">High</text>'
        )
        parts.append(
            f'<text x="{x + legend_width + 5}" y="{y + legend_height}" '
            f'fill="white" font-size="10">Low</text>'
        )

        return "\n".join(parts)


class FlowVisualizer:
    """
    Visualiza flujos de comunicación en el panal.

    Muestra:
    - Rastros de feromonas
    - Patrones de danza
    - Comandos reales

    Uso:
        flow = FlowVisualizer(grid, nectar_flow)
        svg = flow.render_pheromone_trails()
    """

    def __init__(self, grid: HoneycombGrid, nectar_flow: Any | None = None):
        self.grid = grid
        self.nectar_flow = nectar_flow

    def render_pheromone_trails(self, width: int = 600, height: int = 600) -> str:
        """Renderiza rastros de feromonas."""
        radius = self.grid.config.radius
        hex_size = min(width, height) / (radius * 4 + 2)
        center_x = width / 2
        center_y = height / 2

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="#1a1a2e"/>',
        ]

        # Dibujar celdas base
        for coord, _cell in self.grid._cells.items():
            x = center_x + hex_size * (3 / 2 * coord.q)
            y = center_y + hex_size * (math.sqrt(3) / 2 * coord.q + math.sqrt(3) * coord.r)

            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="{hex_size*0.3}" ' f'fill="#333366" opacity="0.5"/>'
            )

        # Dibujar conexiones de feromonas
        for coord, cell in self.grid._cells.items():
            if cell.pheromone_level < 0.1:
                continue

            x1 = center_x + hex_size * (3 / 2 * coord.q)
            y1 = center_y + hex_size * (math.sqrt(3) / 2 * coord.q + math.sqrt(3) * coord.r)

            for neighbor in cell.get_all_neighbors():
                if neighbor and neighbor.pheromone_level > 0.1:
                    x2 = center_x + hex_size * (3 / 2 * neighbor.coord.q)
                    y2 = center_y + hex_size * (
                        math.sqrt(3) / 2 * neighbor.coord.q + math.sqrt(3) * neighbor.coord.r
                    )

                    intensity = (cell.pheromone_level + neighbor.pheromone_level) / 2
                    opacity = min(intensity, 0.8)
                    width = 1 + intensity * 3

                    svg_parts.append(
                        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                        f'stroke="#ff6600" stroke-width="{width}" opacity="{opacity}"/>'
                    )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def render_activity_flow(self, width: int = 600, height: int = 600) -> str:
        """Renderiza flujo de actividad."""
        # Similar a pheromone trails pero basado en actividad
        radius = self.grid.config.radius
        hex_size = min(width, height) / (radius * 4 + 2)
        center_x = width / 2
        center_y = height / 2

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="#0a0a1a"/>',
        ]

        # Partículas de actividad
        for coord, cell in self.grid._cells.items():
            if cell._last_activity < 0.1:
                continue

            x = center_x + hex_size * (3 / 2 * coord.q)
            y = center_y + hex_size * (math.sqrt(3) / 2 * coord.q + math.sqrt(3) * coord.r)

            # Círculo pulsante
            radius_val = hex_size * 0.3 * (1 + cell._last_activity * 0.5)
            opacity = 0.3 + cell._last_activity * 0.5

            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="{radius_val}" '
                f'fill="#00ff00" opacity="{opacity}"/>'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def get_flow_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas de flujo."""
        total_pheromone = 0.0
        active_cells = 0
        connections = 0

        for _coord, cell in self.grid._cells.items():
            total_pheromone += cell.pheromone_level

            if cell._last_activity > 0.1:
                active_cells += 1

            for neighbor in cell.get_all_neighbors():
                if neighbor and cell.pheromone_level > 0.1 and neighbor.pheromone_level > 0.1:
                    connections += 1

        return {
            "total_pheromone": total_pheromone,
            "active_cells": active_cells,
            "pheromone_connections": connections // 2,  # Dividir por 2 (bidireccional)
            "average_pheromone": total_pheromone / len(self.grid._cells) if self.grid._cells else 0,
        }
