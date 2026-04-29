"""
HOC Core · Pheromones (internal)
================================

Sistema de feromonas interno al grid hexagonal para comunicación estigmérgica.

Define ``PheromoneType``, ``PheromoneDeposit`` y ``PheromoneField``, usados
internamente por cada celda para mantener niveles de feromona locales.

NOTA: existe un ``PheromoneType`` adicional en ``hoc.nectar`` — son enums
distintos con propósitos diferentes. Este módulo es el copy *interno* del
core; el de nectar forma parte de la API de alto nivel de comunicación.

Extraído de ``core.py`` en Fase 3.3.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from .grid import HexCoord

__all__ = [
    "PheromoneType",
    "PheromoneDeposit",
    "PheromoneField",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SISTEMA DE FEROMONAS AVANZADO
# ═══════════════════════════════════════════════════════════════════════════════


class PheromoneType(Enum):
    """Tipos de feromonas para comunicación estigmérgica."""

    FOOD = auto()
    DANGER = auto()
    PATH = auto()
    RECRUIT = auto()
    HOME = auto()
    WORK = auto()
    EXPLORATION = auto()


@dataclass(slots=True)
class PheromoneDeposit:
    """Depósito de feromona con metadatos."""

    ptype: PheromoneType
    intensity: float
    timestamp: float = field(default_factory=time.time)
    source_coord: HexCoord | None = None
    decay_rate: float = 0.1

    def decay(self, elapsed: float = 1.0) -> float:
        """Aplica decaimiento y retorna nueva intensidad."""
        self.intensity *= (1.0 - self.decay_rate) ** elapsed
        return self.intensity

    # Minimum intensity to be considered active (below this, pheromone is cleaned up)
    ACTIVE_THRESHOLD: ClassVar[float] = 0.001

    @property
    def is_active(self) -> bool:
        return self.intensity > self.ACTIVE_THRESHOLD


class PheromoneField:
    """
    Campo de feromonas para una celda.

    v3.0: batch_decay con NumPy para mejor rendimiento.
    """

    __slots__ = ("_deposits", "_lock", "_total_intensity")

    def __init__(self):
        self._deposits: dict[PheromoneType, PheromoneDeposit] = {}
        self._total_intensity: float = 0.0
        self._lock = threading.Lock()

    def deposit(
        self,
        ptype: PheromoneType,
        amount: float,
        source: HexCoord | None = None,
        decay_rate: float = 0.1,
    ) -> None:
        """Deposita feromona de un tipo."""
        with self._lock:
            if ptype in self._deposits:
                self._deposits[ptype].intensity = min(1.0, self._deposits[ptype].intensity + amount)
            else:
                self._deposits[ptype] = PheromoneDeposit(
                    ptype=ptype,
                    intensity=min(1.0, amount),
                    source_coord=source,
                    decay_rate=decay_rate,
                )
            self._update_total()

    def get_intensity(self, ptype: PheromoneType) -> float:
        deposit = self._deposits.get(ptype)
        return deposit.intensity if deposit else 0.0

    def decay_all(self, elapsed: float = 1.0) -> None:
        """Aplica decaimiento a todas las feromonas.

        Phase 7.6: when the field carries 4+ deposits, batch the
        decay through numpy. ``intensity *= (1 - decay_rate)**elapsed``
        vectorises cleanly into a single ``np.power`` + multiply, with
        a tombstone pass to drop sub-threshold deposits. Below the
        threshold the per-deposit Python loop is faster (numpy adds
        constant overhead that swamps the win on n<=3).
        """
        with self._lock:
            n = len(self._deposits)
            if n >= 4:
                # SIMD path. Snapshot current state into parallel
                # arrays, apply the decay, write back, drop dead rows.
                ptypes = list(self._deposits)
                intensities = np.fromiter(
                    (self._deposits[p].intensity for p in ptypes),
                    dtype=np.float64,
                    count=n,
                )
                rates = np.fromiter(
                    (self._deposits[p].decay_rate for p in ptypes),
                    dtype=np.float64,
                    count=n,
                )
                factors = np.power(1.0 - rates, elapsed)
                new_intensities = intensities * factors
                threshold = PheromoneDeposit.ACTIVE_THRESHOLD
                for ptype, intensity in zip(ptypes, new_intensities, strict=True):
                    if intensity <= threshold:
                        del self._deposits[ptype]
                    else:
                        self._deposits[ptype].intensity = float(intensity)
            else:
                to_remove: list[PheromoneType] = []
                for ptype, deposit in self._deposits.items():
                    deposit.decay(elapsed)
                    if not deposit.is_active:
                        to_remove.append(ptype)
                for ptype in to_remove:
                    del self._deposits[ptype]

            self._update_total()

    def _update_total(self) -> None:
        self._total_intensity = sum(d.intensity for d in self._deposits.values())

    @property
    def total_intensity(self) -> float:
        return self._total_intensity

    @property
    def dominant_type(self) -> PheromoneType | None:
        if not self._deposits:
            return None
        return max(self._deposits.items(), key=lambda x: x[1].intensity)[0]

    def get_gradient_vector(self) -> dict[PheromoneType, float]:
        """v3.0: Retorna vector de intensidades por tipo."""
        return {ptype: dep.intensity for ptype, dep in self._deposits.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            ptype.name: {"intensity": dep.intensity, "decay_rate": dep.decay_rate}
            for ptype, dep in self._deposits.items()
        }
