# Análisis del benchmark de tareas pesadas (mixtas)

## Objetivo

Evaluar el **SwarmScheduler** con cargas de trabajo **variadas y pesadas** (CPU, memoria, mezcla) para medir:

- Throughput y tiempo total
- Comportamiento por tipo de tarea
- Estabilidad bajo estrés (muchas tareas, varios tipos)

---

## Tipos de carga

| Tipo              | Descripción breve                    | Perfil aproximado   |
|-------------------|--------------------------------------|----------------------|
| `render_3d`       | Mini raycasting 96×96, 6 esferas     | CPU + memoria        |
| `matrix_mult`     | Multiplicación de matrices 384×384×3 | CPU + cache          |
| `matrix_svd`      | SVD sobre matriz 200×200             | CPU numérico         |
| `simulation`      | 1500 pasos de integrador (estado 96) | Bucles + NumPy       |
| `hash_work`       | ~3000 rondas de hash sobre 32 KB     | CPU, poco memoria    |
| `monte_carlo`     | 300k muestras, 5D                    | NumPy aleatorio      |
| `math_eigen`      | Autovalores matriz simétrica 160×160 | Álgebra lineal       |
| `math_fft`        | 30 FFT sobre señal 2^17              | FFT / señal          |
| `math_integrate`  | Integración MC 150k puntos, 4D        | Integración numérica |
| `math_solve`      | Sistemas lineales Cholesky 280×280   | Resolver Ax=b        |
| `math_poly_roots` | Raíces de 2 polinomios grado 100     | Polinomios / eig     |

---

## Cómo ejecutar

```bash
# Benchmark mixto (2 tareas de cada tipo por defecto) — ~25–60 s
python -m benchmarks.bench_heavy_mixed

# Tests pesados (pueden tardar 1–3 min en total)
pytest tests/test_heavy.py -v

# Solo tests de estrés
pytest tests/test_heavy.py::TestHeavyStress -v

# Sin pytest (si fallan plugins del entorno)
python -m tests.test_heavy
```

---

## Métricas a revisar

1. **elapsed_seconds**: Tiempo total hasta cola vacía.
2. **total_completed / total_tasks**: Que todas las tareas completen (o anotar fallos).
3. **tasks_per_second**: Throughput global.
4. **completed_by_type**: Completadas por tipo (detectar tipos que fallen más).
5. **ticks_run**: Cuántos ticks hasta vaciar la cola (relación con foragers y duplicación de ejecuciones).
6. **scheduler_stats.tasks_completed**: Contador de tareas únicas completadas (desde la corrección en tick(): cada tarea se asigna solo a un behavior por tick).

---

## Interpretación

- **Throughput bajo**: Tareas muy pesadas o pocos Foragers; subir `grid_radius` o aligerar carga.
- **failed_by_type > 0**: Revisar timeouts o excepciones en ese tipo de carga.
- **ticks_run muy alto** para pocas tareas: Posible que muchos Foragers no “respondan” (umbral aleatorio) o que el mismo trabajo se ejecute varias veces (mismo task elegido por varios behaviors).
- **Comparar por tipo**: Si un tipo tarda mucho más, domina el tiempo total; considerar dividirlo en subtareas o reducir coste por tarea.

---

## Ejemplo de resultado real

Tras ejecutar `python -m benchmarks.bench_heavy_mixed` (2 tareas × 6 tipos = 12 tareas):

```
Tiempo total:     25.320 s
Tareas totales:   12 (12 completadas, 0 fallidas)
Ticks:            3
Throughput:       0.47 tareas/s
Celdas grid:      37

Por tipo (completadas):
  render_3d: 2 ok, 0 failed
  matrix_mult: 2 ok, 0 failed
  matrix_svd: 2 ok, 0 failed
  simulation: 2 ok, 0 failed
  hash_work: 2 ok, 0 failed
  monte_carlo: 2 ok, 0 failed

Scheduler tasks_completed: 12 (tras corrección en tick())
```

**Análisis:**

- Todas las tareas completaron (12/12).
- Throughput ~0,45–0,47 tareas/s: el tiempo está dominado por el coste CPU de cada tarea (matrix_svd y matrix_mult son los más pesados).
- Tras la corrección en `tick()`: cada tarea se asigna solo a un behavior por tick; `tasks_completed` coincide con las tareas únicas. Con suficientes Foragers puede bastar 1 tick para vaciar la cola.
- Ningún tipo falló; los timeouts por defecto (120 s) son suficientes para estas cargas.

---

## Referencias

- **ANALISIS_RENDER.md**: Análisis del benchmark solo-render; documenta la corrección en `tick()` (claimed_this_tick).
- **bench_heavy_mixed.py**: Código del benchmark mixto.
- **workload_heavy.py**: Definición de cada carga.
- **tests/test_heavy.py**: Tests pesados y de estrés.
