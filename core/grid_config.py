"""
HOC Core · Grid configuration (private)
=======================================

Configuración del grid hexagonal y enumeración de topologías.

Provee ``HoneycombConfig`` (dataclass con validación exhaustiva) y
``GridTopology`` (enum FLAT/TORUS/SPHERE/INFINITE).

Este módulo es *interno* y no depende de celdas ni grid — permite que
``cells.py`` importe ``HoneycombConfig`` sin crear un ciclo.

Extraído de ``core.py`` en Fase 3.3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import Any

__all__ = ["HoneycombConfig", "GridTopology"]


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL PANAL (v3.0 - validación robusta)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HoneycombConfig:
    """Configuración del grid hexagonal con validación exhaustiva."""

    # Tamaño
    radius: int = 10
    initial_cells: int = 37

    # Celdas especializadas
    queens_per_grid: int = 1
    drones_per_ring: int = 2
    nurseries_per_grid: int = 3

    # Capacidad
    vcores_per_cell: int = 8
    max_entities_per_cell: int = 100

    # Comunicación
    pheromone_decay_rate: float = 0.1
    waggle_broadcast_range: int = 3
    pheromone_diffusion_rate: float = 0.05

    # Resiliencia
    replication_factor: int = 2
    failover_timeout_ms: int = 1000
    max_consecutive_errors: int = 3

    # Rendimiento
    parallel_ring_processing: bool = True
    max_parallel_rings: int = 4
    tick_batch_size: int = 50

    # Work-stealing
    steal_threshold_low: float = 0.3
    steal_threshold_high: float = 0.7
    max_steal_per_tick: int = 2

    # Topología
    topology: str = "flat"  # 'flat', 'torus', 'sphere'

    # Métricas
    metrics_history_size: int = 1000
    metrics_sample_rate: float = 1.0

    # v3.0: Circuit breaker
    circuit_breaker_threshold: int = 3
    circuit_breaker_recovery_s: float = 5.0

    # v3.0: Health monitoring
    health_check_interval_s: float = 10.0
    health_alert_load_threshold: float = 0.9

    # v3.1: Extracted magic numbers
    pheromone_active_threshold: float = 0.001
    pheromone_diffuse_threshold: float = 0.01
    load_change_event_threshold: float = 0.1
    health_critical_failed_ratio: float = 0.2
    health_critical_load: float = 0.95
    health_degraded_failed_ratio: float = 0.05
    cluster_health_load_weight: float = 0.3
    cluster_health_health_weight: float = 0.5
    cluster_health_balance_weight: float = 0.2
    scout_novelty_bonus: float = 0.5
    scout_explore_pheromone_weight: float = 0.3
    scout_path_deposit_intensity: float = 0.3
    scout_low_load_threshold: float = 0.1
    nursery_default_incubation_rate: float = 0.1
    # Visualization thresholds
    viz_load_high: float = 0.8
    viz_load_medium: float = 0.5
    viz_load_low: float = 0.3

    # Phase 6.4: auto-checkpointing during tick().
    # ``checkpoint_interval_ticks=None`` (default) disables auto-checkpoint.
    # Setting it to e.g. 100 + ``checkpoint_path="/var/hoc/snapshot.bin"``
    # makes the grid persist itself every 100 ticks. The write is atomic
    # (``HoneycombGrid.checkpoint`` writes to ``.tmp`` and renames) so a
    # crash mid-write does not corrupt the previous snapshot.
    checkpoint_interval_ticks: int | None = None
    checkpoint_path: str | None = None
    checkpoint_compress: bool = False

    def __post_init__(self):
        """
        v3.0 FIX: Usa ValueError en lugar de assert.
        assert se deshabilita con python -O, lo que bypasea toda validación.
        """
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.vcores_per_cell <= 0:
            raise ValueError(f"vcores_per_cell must be positive, got {self.vcores_per_cell}")
        if not (0.0 <= self.pheromone_decay_rate <= 1.0):
            raise ValueError(
                f"pheromone_decay_rate must be in [0, 1], got {self.pheromone_decay_rate}"
            )
        if self.topology not in ("flat", "torus", "sphere"):
            raise ValueError(f"invalid topology: {self.topology!r}")
        if self.steal_threshold_low >= self.steal_threshold_high:
            raise ValueError(
                f"steal_threshold_low ({self.steal_threshold_low}) must be < "
                f"steal_threshold_high ({self.steal_threshold_high})"
            )
        if self.max_parallel_rings <= 0:
            raise ValueError(f"max_parallel_rings must be positive, got {self.max_parallel_rings}")
        # Phase 6.4: validate checkpoint config consistency.
        if self.checkpoint_interval_ticks is not None:
            if self.checkpoint_interval_ticks <= 0:
                raise ValueError(
                    f"checkpoint_interval_ticks must be positive when set, "
                    f"got {self.checkpoint_interval_ticks}"
                )
            if not self.checkpoint_path:
                raise ValueError("checkpoint_interval_ticks requires checkpoint_path to be set")

    def cells_at_radius(self, r: int) -> int:
        return 1 if r == 0 else 6 * r

    def total_cells(self) -> int:
        return 1 + 3 * self.radius * (self.radius + 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HoneycombConfig:
        # Filtrar solo campos conocidos para forward-compatibility
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


class GridTopology(Enum):
    """Topología del grid hexagonal."""

    FLAT = auto()
    TORUS = auto()
    SPHERE = auto()
    INFINITE = auto()
