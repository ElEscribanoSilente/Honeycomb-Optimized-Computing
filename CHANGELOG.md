# Changelog

Todas las modificaciones notables del proyecto **HOC (Honeycomb Optimized
Computing)** se documentan en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

---

## [1.1.0-phase01] — 2026-04-22

**Cierre de Fase 1 — Estabilización crítica.** 378 tests pasando, cobertura
83–95% en los 4 módulos previamente sin tests, 0 vulnerabilidades de
dependencias, 0 hallazgos `bandit` HIGH.

Reporte completo: [snapshot/PHASE_01_CLOSURE.md](snapshot/PHASE_01_CLOSURE.md).

### Fixed

#### Bugs del roadmap (B1–B8)
- **B1** [`core.py`] `RWLock`: `try/finally` correcto, eliminado `bare except`.
- **B2** [`swarm.py`] `SwarmScheduler.tick()`: TOCTOU corregido extendiendo el lock.
- **B3** [`nectar.py`] Validación de `decay_rate` y `diffusion_rate` en `__init__`.
- **B4** [`resilience.py`] `_conduct_election` ahora exige quórum mayoritario y retorna `None` si no se alcanza (antes podía elegir reina con minoría).
- **B5** [`memory.py`] `PollenCache.put()` resta los bytes del valor antiguo **antes** del bucle de evicción (antes provocaba evicciones espurias al reemplazar una clave existente).
- **B7** [`metrics.py`] Buckets de `Histogram` ahora respetan la convención cumulativa de Prometheus (verificado por test).
- **B8** [`resilience.py`] `_repair_neighbor_link`: `try/except` alrededor de la búsqueda en `HexDirection` para tolerar nombres de dirección inválidos.

#### Bugs latentes descubiertos durante el testing (no estaban en el roadmap)
- **B2.5** [`swarm.py`] `SwarmScheduler.tick()` no limpiaba `_task_index` junto con `_task_queue`, provocando una fuga de memoria en runs largos.
- **B9** [`metrics.py`] 10 call sites accedían a `cell._pheromone_level` (atributo privado inexistente) en vez de la property pública `cell.pheromone_level`. Cualquier llamada a métricas habría lanzado `AttributeError` en runtime.
- **B10** [`core.py`] `HexCoord` usaba `@cached_property` con `@dataclass(frozen=True, slots=True)`, combinación que prohíbe `__dict__`. `HexCoord.cube`, `.array` y `.magnitude` lanzaban `TypeError` garantizado en cada acceso. Reemplazado por `@property` (la pérdida de cache es despreciable: cómputos O(1) sobre dos enteros).

> B9 y B10 demuestran el valor del esfuerzo de testing: ambos eran fallos de
> runtime garantizados que ningún test previo cubría.

### Added

- **`tests/test_memory.py`** — 71 tests, cobertura `memory.py` 94%. Incluye verificación de B5 (`test_replace_key_does_not_trigger_spurious_eviction`).
- **`tests/test_metrics.py`** — 76 tests, cobertura `metrics.py` 95%. Verifica convención cumulativa de Prometheus para `Histogram`.
- **`tests/test_resilience.py`** — 75 tests, cobertura `resilience.py` 83%. Incluye verificación de B4 y B8.
- **`tests/test_swarm.py`** — 65 tests, cobertura `swarm.py` 89%. Incluye verificación de B2.5 (`test_b2_5_no_leak_after_many_cycles`).
- **`tests/test_property.py`** — 53 tests Hypothesis cubriendo:
  - Álgebra de `HexCoord`: invariante cúbica `q+r+s=0`, simetría/triángulo de distancias, conmutatividad/asociatividad de suma, identidad/inverso, rotación 6×60° = identidad, anillos de tamaño `6r`, hexágono lleno de tamaño `1+3r(r+1)`, `lerp`, etc.
  - Semántica de `PheromoneField`/`PheromoneDeposit`: clamp `[0, 1]`, monotonía bajo cap, decay nunca incrementa intensidad, `total_intensity == sum`, `dominant_type` retorna ptype con máxima intensidad.
- **`snapshot/PHASE_01_CLOSURE.md`** — reporte completo de cierre de Fase 1.
- **`snapshot/bandit_phase01.json`** — escaneo de seguridad: 0 HIGH, 3 MEDIUM (todos `pickle`, planificados para Fase 2 con `mscs`), 4 LOW.
- **`snapshot/pip_audit_phase01.txt`** — auditoría de dependencias: 0 vulnerabilidades.
- **`ROADMAP.md`** — marcado FASE 1 como cerrada con resumen de resultados.
- **`.gitignore`** — añadidos `.hypothesis/` y `.claude/`.

### Deferred

- **B6** (TOCTOU en `swarm.py`/`resilience.py` cargando `load`) — clasificado como no bloqueante; se aborda en una fase posterior.
- Cobertura global 71% (objetivo era 75%). Los 4 módulos críticos superan el 80%; las brechas remanentes están en `core.py`, `bridge.py` y `nectar.py`, fuera del scope de Fase 1.

---

## [1.0.0] — 2026-03-XX (baseline)

**Snapshot inicial preservado** previo al inicio del roadmap de 10 fases.

- Tag preservado: `v1.0.0-baseline`
- Branch preservada: `baseline/v1.0.0`
- Documentación del estado: [snapshot/SNAPSHOT.md](snapshot/SNAPSHOT.md)

### Estado

- 8 módulos Python, ~10.557 LOC.
- Tests existentes: `test_core.py`, `test_nectar.py`, `test_bridge.py`, `test_heavy.py`.
- Sin tests directos: `memory.py`, `resilience.py`, `metrics.py`, `swarm.py`.
- Cobertura estimada: 30–40%.
- Auditoría inicial: 3 bugs críticos, 5 altos, 4 medios, 3 bajos (15 totales).

[1.1.0-phase01]: https://github.com/ElEscribanoSilente/Honeycomb-Optimized-Computing/releases/tag/v1.1.0-phase01
[1.0.0]: https://github.com/ElEscribanoSilente/Honeycomb-Optimized-Computing/releases/tag/v1.0.0-baseline
