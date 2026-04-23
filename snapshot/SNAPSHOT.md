# Baseline Snapshot — HOC v1.0.0

**Fecha**: 2026-04-22
**Propósito**: Preservar el estado actual del repositorio antes de iniciar el roadmap de 10 fases.

---

## Estado Git preservado

| Recurso | Valor | Cómo restaurar |
|---------|-------|----------------|
| **Tag** | `v1.0.0-baseline` | `git checkout v1.0.0-baseline` |
| **Branch** | `baseline/v1.0.0` | `git checkout baseline/v1.0.0` |
| **Commit HEAD** | `afb75d8` | `git reset --hard afb75d8` (destructivo) |
| **Patch uncommitted** | `snapshot/uncommitted-changes.patch` | `git apply snapshot/uncommitted-changes.patch` |

> **Nota**: Las "3 modificaciones sin commitear" reportadas por `git status` (README.md, benchmarks/workload_heavy.py, benchmarks/ANALISIS_BENCHMARK_PESADOS.md) son **solo diferencias de line-endings** (CRLF↔LF por Windows autocrlf), no cambios de contenido. El patch generado está vacío, lo cual es correcto.

---

## Inventario de archivos al snapshot

### Módulos Python (10.557 LOC)
| Archivo | LOC | Propósito |
|---------|-----|-----------|
| `core.py` | 3.624 | Grid hexagonal, células, EventBus, CircuitBreaker, HealthMonitor |
| `resilience.py` | 1.482 | Failover, sucesión de reina, recovery |
| `metrics.py` | 1.176 | Recolección + visualización |
| `nectar.py` | 1.106 | Feromonas, WaggleDance, RoyalJelly |
| `swarm.py` | 1.056 | Scheduler bio-inspirado |
| `bridge.py` | 915 | Integración CAMV (hex ↔ vCores) |
| `memory.py` | 862 | Persistencia 3 capas (Pollen/Comb/Honey) |
| `__init__.py` | 336 | Public API |

### Tests existentes
- `tests/test_core.py`, `test_nectar.py`, `test_bridge.py`, `test_heavy.py`, `conftest.py`
- **Sin tests**: `memory.py`, `resilience.py`, `metrics.py`, `swarm.py` (solo indirecto)

### Documentación
- `README.md`, `NECTAR_SPEC.md`
- `benchmarks/ANALISIS_BENCHMARK_PESADOS.md`, `benchmarks/ANALISIS_RENDER.md`

### Configuración
- `pyproject.toml` (v1.0.0, Python ≥3.10, dep única: `numpy>=1.21.0`)
- `requirements.txt`, `requirements-dev.txt`

---

## Métricas de baseline

### Auditoría previa (resumen)
- **Bugs**: 3 críticos, 5 altos, 4 medios, 3 bajos = **15 totales**
- **Seguridad**: 1 crítico (pickle), 5 altos, 4 medios, 2 bajos = **12 totales**
- **Calidad**: 2 críticos (memory/resilience sin tests), 7 altos = **17+ hallazgos**

### Cobertura de tests baseline
- **Estimada**: ~30-40% (sin medición formal aún)
- **Módulos sin cobertura directa**: `memory.py`, `resilience.py`, `metrics.py`, `swarm.py`

### Performance baseline (de [ANALISIS_BENCHMARK_PESADOS.md](../benchmarks/ANALISIS_BENCHMARK_PESADOS.md))
- 12/12 tareas completadas (100% éxito)
- ~25s tiempo total, 0.47 tareas/s throughput
- 3 ticks necesarios

---

## Restauración de emergencia

Si algo se rompe durante el roadmap, restaurar el baseline:

```bash
# Opción A: cambiar a branch baseline (no destructivo)
git checkout baseline/v1.0.0

# Opción B: restaurar main al estado baseline (DESTRUCTIVO — pierde cambios)
git checkout main
git reset --hard v1.0.0-baseline

# Opción C: cherry-pick lo que sirve y descartar resto
git checkout -b recovery v1.0.0-baseline
git cherry-pick <commits-buenos>
```

---

## Política durante el roadmap

1. **Nunca borrar** `baseline/v1.0.0` ni el tag `v1.0.0-baseline`.
2. **Cada fase** se desarrolla en branch propia: `phase/01-stabilization`, `phase/02-security`, etc.
3. **Merge a `main`** solo tras pasar: tests + audit de seguridad + audit general.
4. **Tag al cerrar fase**: `v1.1.0-phase01`, `v1.2.0-phase02`, etc.
5. **CHANGELOG.md** se actualiza al cerrar cada fase (creado en Fase 3).
