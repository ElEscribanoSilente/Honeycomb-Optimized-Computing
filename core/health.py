"""
HOC Core · Health & Circuit Breaker
===================================

Primitivas de salud del grid.

Provee:
- ``CircuitState`` / ``CircuitBreaker``: circuit breaker con backoff exponencial
  usado por cada ``HoneycombCell`` para evitar cascadas de fallos.
- ``HealthStatus`` / ``HealthMonitor``: monitor del grid que clasifica el
  estado global como ``HEALTHY`` / ``DEGRADED`` / ``CRITICAL``.

Extraído de ``core.py`` en Fase 3.3.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from .events import Event, EventBus, EventType, get_event_bus

if TYPE_CHECKING:
    from .grid import HoneycombGrid

__all__ = [
    "CircuitState",
    "CircuitBreaker",
    "HealthStatus",
    "HealthMonitor",
]


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER (v3.0 - nuevo)
# ═══════════════════════════════════════════════════════════════════════════════


class CircuitState(Enum):
    """Estado del circuit breaker."""

    CLOSED = auto()  # Funcionando normal
    OPEN = auto()  # Abierto, rechazando operaciones
    HALF_OPEN = auto()  # Probando recuperación


class CircuitBreaker:
    """
    Circuit breaker con backoff exponencial para protección de celdas.

    Previene cascadas de fallos cerrando el circuito cuando una celda
    falla repetidamente, con recovery automático.
    """

    __slots__ = (
        "_backoff_multiplier",
        "_failure_count",
        "_failure_threshold",
        "_last_failure_time",
        "_lock",
        "_max_recovery_timeout",
        "_recovery_timeout",
        "_state",
        "_success_count_in_half_open",
        "_success_threshold",
    )

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 5.0,
        success_threshold: int = 2,
        backoff_multiplier: float = 2.0,
        max_recovery_timeout: float = 300.0,
    ):
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._last_failure_time = 0.0
        self._success_count_in_half_open = 0
        self._success_threshold = success_threshold
        self._lock = threading.Lock()
        self._backoff_multiplier = backoff_multiplier
        self._max_recovery_timeout = max_recovery_timeout

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count_in_half_open = 0
            return self._state

    def record_success(self) -> None:
        """Registra operación exitosa."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count_in_half_open += 1
                if self._success_count_in_half_open >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._recovery_timeout = max(
                        5.0, self._recovery_timeout / self._backoff_multiplier
                    )
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Registra fallo."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._recovery_timeout = min(
                    self._max_recovery_timeout, self._recovery_timeout * self._backoff_multiplier
                )
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self._failure_threshold
            ):
                self._state = CircuitState.OPEN

    def allow_request(self) -> bool:
        """¿Se permite la operación?"""
        current = self.state
        return current in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def reset(self) -> None:
        """Reset manual."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count_in_half_open = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.name,
            "failure_count": self._failure_count,
            "recovery_timeout": self._recovery_timeout,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH MONITOR (v3.0 - nuevo)
# ═══════════════════════════════════════════════════════════════════════════════


class HealthStatus(Enum):
    """Estado de salud del sistema."""

    HEALTHY = auto()
    DEGRADED = auto()
    CRITICAL = auto()


class HealthMonitor:
    """
    Monitor de salud del grid con alertas automáticas.

    Evalúa: carga promedio, celdas fallidas, circuit breakers abiertos,
    y tendencia de carga.
    """

    __slots__ = (
        "_alert_threshold",
        "_check_interval",
        "_event_bus",
        "_grid",
        "_last_check",
        "_lock",
        "_status_history",
    )

    def __init__(
        self,
        grid: HoneycombGrid,
        event_bus: EventBus | None = None,
        check_interval: float = 10.0,
        alert_threshold: float = 0.9,
    ):
        self._grid = grid
        self._event_bus = event_bus or get_event_bus()
        self._check_interval = check_interval
        self._alert_threshold = alert_threshold
        self._last_check = 0.0
        self._status_history: deque = deque(maxlen=100)
        self._lock = threading.Lock()

    def check_health(self) -> dict[str, Any]:
        """Ejecuta health check completo."""
        now = time.time()

        with self._lock:
            stats = self._grid.get_stats()

            total = max(1, stats["total_cells"])
            failed_ratio = stats["failed_cells"] / total
            avg_load = stats["average_load"]

            grid_config = self._grid.config
            if (
                failed_ratio > grid_config.health_critical_failed_ratio
                or avg_load > grid_config.health_critical_load
            ):
                status = HealthStatus.CRITICAL
            elif (
                failed_ratio > grid_config.health_degraded_failed_ratio
                or avg_load > self._alert_threshold
            ):
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            result = {
                "status": status.name,
                "timestamp": now,
                "average_load": avg_load,
                "failed_cells": stats["failed_cells"],
                "failed_ratio": failed_ratio,
                "total_cells": total,
            }

            self._status_history.append(result)
            self._last_check = now

        # Emitir eventos
        self._event_bus.publish(Event(type=EventType.HEALTH_CHECK, source=self, data=result))

        if status != HealthStatus.HEALTHY:
            self._event_bus.publish(Event(type=EventType.HEALTH_ALERT, source=self, data=result))

        return result

    def should_check(self) -> bool:
        return time.time() - self._last_check >= self._check_interval

    def get_status_trend(self, window: int = 10) -> list[dict]:
        return list(self._status_history)[-window:]
