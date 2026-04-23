# Changelog

Todas las modificaciones notables del proyecto **HOC (Honeycomb Optimized
Computing)** se documentan en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

---

## [1.2.0-phase02] — 2026-04-23

**Cierre de Fase 2 — Seguridad & Hardening.** 421 tests pasando (43 nuevos
dedicados a seguridad), `pickle` erradicado del código de producción y
reemplazado por `mscs` con HMAC-SHA256, mensajes de `NectarFlow`/`RoyalJelly`
firmados, protocolo Raft-like con votos firmados en `QueenSuccession`,
Bandit limpio en todas las severidades, overhead end-to-end +3.5% (<5%).

Reporte completo: [snapshot/PHASE_02_CLOSURE.md](snapshot/PHASE_02_CLOSURE.md).

### Added

- **`security.py`** (nuevo módulo, 83% cobertura) — primitivas centralizadas:
  `serialize`/`deserialize` con HMAC, `sign_payload`/`verify_signature`,
  `secure_random`/`secure_choice`/`secure_shuffle` sobre `secrets.SystemRandom`,
  `safe_join` con `PathTraversalError`, `RateLimiter`/`rate_limit` token
  bucket, `sanitize_error` (respeta `HOC_DEBUG`).
- **`HOC_HMAC_KEY`** (env var) — permite fijar clave HMAC compartida entre
  procesos. En ausencia, cada proceso genera una clave efímera de 32 bytes.
- **`tests/test_security.py`** — 43 tests cubriendo las 5 áreas obligatorias:
  - `TestMscsRejectsMalicious` (5): payloads pickle-RCE rechazados, HMAC
    tamper detection, foreign key rejection, registry strict, CombStorage
    tamper-detection.
  - `TestRoyalCommandQueenOnly` (6): DroneCell bloqueado en `priority=10`,
    Queen aceptada, threshold exacto, `update_queen_coord`, forge detection.
  - `TestQuorumSignedVotes` (7): voter duplicado, voto sin firma, firma
    manipulada, wrong term, candidato desconocido, mayoría, term monotónico.
  - `TestPheromoneBoundedDoS` (4): 10K deposits/misma coord, 10K distintas,
    metadata flood, auto-sign.
  - `TestHoneyArchivePathTraversal` (5): `../`, absoluto, null byte, key
    válido, primitive `safe_join`.
  - Clases transversales: `TestHmacPrimitives`, `TestCsprng`,
    `TestRateLimiter`, `TestDanceSigning`, `TestHiveMemoryIntegration`.
- **`Vote`** dataclass (`resilience.py`) — voto firmado con
  `voter`/`candidate`/`term`/`timestamp`/`signature` para el protocolo
  Raft-like de sucesión.
- **`signature`** campo opcional en `PheromoneDeposit`, `DanceMessage`,
  `RoyalMessage` con métodos `_canonical_payload()`, `sign()`, `verify()`.
- **`issuer`** en `RoyalMessage` + `RoyalJelly.HIGH_PRIORITY_THRESHOLD=8`.
- **`RoyalJelly.update_queen_coord()`** — propaga la sucesión a la capa de
  comunicación.
- **`QueenSuccession.current_term`** property; `_tally_votes` público-ish
  para testing (incluye contadores de rechazo por razón).
- **`SwarmScheduler.execute_on_cell()`** — API pública para ejecución
  directa sobre celda (rate-limited).
- **`SwarmConfig.submit_rate_per_second`/`submit_rate_burst`/
  `execute_rate_per_second`/`execute_rate_burst`** — tunables del rate
  limiter.
- **`PheromoneTrail` params**: `max_coords` (default 10_000), `max_metadata_keys`
  (default 100). LRU evicción sobre coordenadas.
- **`HoneyArchive._validate_key()`** — rechaza claves con traversal, null
  bytes, o paths absolutos.
- **`snapshot/bandit_phase02.json`** — 0 HIGH, 0 MEDIUM, 0 LOW.
- **`snapshot/pip_audit_phase02.txt`** — "No known vulnerabilities found".
- **`snapshot/PHASE_02_CLOSURE.md`** — reporte completo de cierre.

### Changed

- **`memory.py`**: `pickle.dumps/loads` reemplazados por
  `security.serialize/deserialize` con HMAC-SHA256 en `CombStorage.put/get`
  y `HoneyArchive.archive/retrieve`. `PollenCache.put` usa `mscs.dumps`
  sin HMAC para estimación de tamaño. `PollenCache._evict_one` política
  RANDOM usa `secrets.SystemRandom`.
- **`memory.HoneyArchive.__init__`**: `base_path` default ahora
  `tempfile.gettempdir()/hoc-honey` (fix Bandit B108, antes `/tmp/honey`).
  `base_path` se normaliza a absoluto con `Path.resolve()`.
- **`resilience.py`**: `import pickle` + `pickle.dumps` en
  `CombRepair._check_data_integrity` reemplazados por `security.serialize`.
- **`resilience.QueenSuccession`**: `_conduct_election` refactorizado a
  protocolo Raft-like. Usa `_tally_votes` que rechaza voters duplicados,
  firmas inválidas, terms incorrectos, candidatos desconocidos, y exige
  mayoría estricta (>50%).
- **`nectar.PheromoneTrail`**: `_deposits` migrado de `defaultdict(dict)`
  a `OrderedDict` con cap LRU (`max_coords`). Metadata merge acotada por
  `max_metadata_keys`. Cada deposit nuevo se auto-firma.
- **`nectar.WaggleDance`**: `start_dance` auto-firma; `propagate` preserva
  firma original (los campos mutables quality/ttl están fuera del payload
  canónico).
- **`nectar.RoyalJelly.issue_command`**: acepta kwarg `issuer`. Lanza
  `PermissionError` si `priority >= HIGH_PRIORITY_THRESHOLD` y `issuer` no
  es la Queen actual. Todos los comandos firmados.
- **`swarm.BeeBehavior.should_respond`**: `random.random()` →
  `secure_random()`.
- **`swarm.SwarmScheduler._initialize_behaviors`**: `random.shuffle` →
  `secure_shuffle`.
- **`swarm.SwarmScheduler.submit_task`**: rate-limited vía `RateLimiter`.
- **Logs sanitizados** (via `security.sanitize_error`) en 6 sitios:
  `memory.CombStorage.get`, `memory.HoneyArchive.archive/retrieve`,
  `resilience.CellFailover._migrate_work`, `resilience.SwarmRecovery.execute_recovery_plan`,
  `nectar.WaggleDance.start_dance`, `swarm.SwarmScheduler.tick`,
  `bridge` (2 sitios).

### Fixed

- **Bandit B108** (`memory.HoneyArchive`): default `base_path="/tmp/honey"`
  reemplazado por `tempfile.gettempdir()/hoc-honey` para evitar race/symlink
  attacks en POSIX multi-usuario.
- **Defaultdict silencioso** (`nectar.PheromoneTrail._deposits`): migración
  a `OrderedDict` elimina el crecimiento sin cota que permitía DoS por
  flood de coordenadas.

### Removed

- **`import pickle`** eliminado de `memory.py` y `resilience.py` en path
  de producción. `pickle` solo aparece ahora en `tests/test_security.py`
  como input adversarial para verificar que `mscs` lo rechaza.
- **`import random`** eliminado de `memory.py` (era un import inline en
  `_evict_one`) y `swarm.py` / `resilience.py` (no usados tras el
  reemplazo por `secrets.SystemRandom`).

### Security

- **`pickle` → `mscs`** con HMAC-SHA256 y registry strict cierra el vector
  RCE clásico por `__reduce__` malicioso. Un atacante que plantee bytes
  en CombStorage/HoneyArchive/RPC hit MSCSecurityError antes de reconstruir.
- **HMAC-SHA256 sobre mensajes** de `NectarFlow`/`RoyalJelly` atestigua
  origen (un atacante sin la clave no puede forjar mensajes aceptables).
- **Queen-only enforcement** en `RoyalCommand` priority ≥ 8 cierra el
  vector "DroneCell forja EMERGENCY" aun con clave compartida.
- **Votos firmados + `term_number` monotónico** en `QueenSuccession`
  blindan la elección contra replay, votos duplicados y candidatos
  forjados.
- **CSPRNG** (`secrets.SystemRandom`) en decisiones que afectan el
  scheduling/capacity (respuesta a estímulos, shuffle de roles, política
  RANDOM de evicción de cache) impide manipulación predictiva vía seed
  del RNG global.
- **Rate limiting** en APIs públicas (`submit_task`, `execute_on_cell`)
  con default 1000/s burst 2000 y 10_000/s burst 20_000 respectivamente.
- **Path validation** (`safe_join`) en `HoneyArchive` rechaza traversal,
  rutas absolutas y null bytes — defense in depth antes de que el
  checkpoint a disco se active.
- **Bounded growth** en `PheromoneTrail` con LRU cap (default 10K coords,
  100 metadata keys por deposit) mitiga DoS por flood.
- **Log sanitization**: producción oculta detalles de excepción
  (`HOC_DEBUG=0` default); activable con `HOC_DEBUG=1`.

### Deferred

- Logs sanitizados en `core.py` (6 sitios de callbacks internos) — no
  security-sensitive pero convendría pasarlos por `sanitize_error` en una
  fase futura de consistencia.

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

[1.2.0-phase02]: https://github.com/ElEscribanoSilente/Honeycomb-Optimized-Computing/releases/tag/v1.2.0-phase02
[1.1.0-phase01]: https://github.com/ElEscribanoSilente/Honeycomb-Optimized-Computing/releases/tag/v1.1.0-phase01
[1.0.0]: https://github.com/ElEscribanoSilente/Honeycomb-Optimized-Computing/releases/tag/v1.0.0-baseline
