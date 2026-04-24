"""
HOC Core · Constants
====================

Constantes de uso general del subpaquete ``hoc.core``.

Fase 3.3: este módulo es un scaffolding creado durante el split de
``core.py``. Contiene una primera tanda de magic numbers extraídos de los
defaults de :class:`hoc.core.grid.HoneycombConfig`. La extracción es
*ongoing work* — fases posteriores migrarán más constantes aquí y luego
harán que los defaults de ``HoneycombConfig`` referencien este módulo.

Por ahora estas constantes están duplicadas con los defaults de
``HoneycombConfig`` (el dataclass sigue siendo la fuente de la verdad
runtime). Se exportan para código cliente que necesita los valores sin
instanciar un config completo.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "DEFAULT_RADIUS",
    "DEFAULT_VCORES_PER_CELL",
    "DEFAULT_MAX_ENTITIES_PER_CELL",
    "DEFAULT_PHEROMONE_DECAY_RATE",
    "DEFAULT_PHEROMONE_DIFFUSION_RATE",
    "PHEROMONE_ACTIVE_THRESHOLD",
]

# ─── Grid geometry ─────────────────────────────────────────────────────────────
#: Radio por defecto del grid hexagonal (anillos alrededor de la reina).
DEFAULT_RADIUS: Final[int] = 10

# ─── Cell capacity ─────────────────────────────────────────────────────────────
#: vCores simultáneos que admite una celda antes de considerarse llena.
DEFAULT_VCORES_PER_CELL: Final[int] = 8

#: Entidades máximas que pueden residir en una única celda.
DEFAULT_MAX_ENTITIES_PER_CELL: Final[int] = 100

# ─── Pheromone dynamics ────────────────────────────────────────────────────────
#: Fracción de intensidad que decae por tick.
DEFAULT_PHEROMONE_DECAY_RATE: Final[float] = 0.1

#: Fracción de intensidad que difunde a vecinos por tick.
DEFAULT_PHEROMONE_DIFFUSION_RATE: Final[float] = 0.05

#: Umbral bajo el cual una feromona se considera inactiva y se limpia.
PHEROMONE_ACTIVE_THRESHOLD: Final[float] = 0.001
