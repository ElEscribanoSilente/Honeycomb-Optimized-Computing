"""
HOC Core · Cells (facade)
=========================

Celdas del panal hexagonal. Fachada pública que re-exporta desde
los submódulos internos:

- :mod:`hoc.core.cells_base`: ``HoneycombCell`` + enums ``CellState`` y
  ``CellRole``.
- :mod:`hoc.core.cells_specialized`: las 7 subclases especializadas
  (``QueenCell``, ``WorkerCell``, ``DroneCell``, ``NurseryCell``,
  ``StorageCell``, ``GuardCell``, ``ScoutCell``).

La identidad de clase se conserva — ``hoc.core.cells.HoneycombCell`` es
literalmente el mismo objeto que ``hoc.core.cells_base.HoneycombCell``.

El split de ``cells_base`` vs. ``cells_specialized`` se hizo durante Fase
3.3 para respetar el límite DoD de 800 LOC por archivo.
"""

from __future__ import annotations

from .cells_base import CellRole, CellState, HoneycombCell
from .cells_specialized import (
    DroneCell,
    GuardCell,
    NurseryCell,
    QueenCell,
    ScoutCell,
    StorageCell,
    WorkerCell,
)

__all__ = [
    "CellState",
    "CellRole",
    "HoneycombCell",
    "QueenCell",
    "WorkerCell",
    "DroneCell",
    "NurseryCell",
    "StorageCell",
    "GuardCell",
    "ScoutCell",
]
