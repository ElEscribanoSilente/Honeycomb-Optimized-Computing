# Análisis del resultado: benchmark mini render 3D en SwarmScheduler

## Resultado observado

```
Tiempo total:     0.784 s
Tareas:           8 enviadas, 8 completadas
Ticks scheduler: 3
Throughput:      10.20 tareas/s
Celdas grid:     37

Stats scheduler:
  tick_count: 3
  queue_size: 0
  pending_tasks: 0
  tasks_completed: 48   ← discrepancia
  tasks_failed: 0
  behaviors: ForagerBehavior: 16, NurseBehavior: 4, ScoutBehavior: 4, GuardBehavior: 3
```

---

## 1. Qué está bien

- **Todas las tareas se completan:** las 8 tareas únicas llegan a estado `COMPLETED`; la cola queda vacía en 3 ticks.
- **Rendimiento:** ~0,78 s para 8 renders 96×96 con 6 esferas y 4 muestras/píxel es razonable para NumPy en un solo hilo.
- **Throughput reportado:** 8/0,784 ≈ 10,2 tareas/s es coherente con “8 tareas en 0,78 s”.
- **Composición del enjambre:** 16 Foragers (los que ejecutan `compute`), 4 Nurses, 4 Scouts, 3 Guards; solo los Foragers toman tareas de tipo `compute`.

---

## 2. Discrepancia: 8 tareas vs 48 “tasks_completed”

- **Hecho:** En el benchmark se envían **8 tareas** y se comprueba que **8 tareas** tienen `state == TaskState.COMPLETED`.
- **Hecho:** El scheduler reporta **tasks_completed = 48**.

Por tanto, **el contador del scheduler no cuenta “tareas únicas completadas”, sino “veces que se ha ejecutado una tarea”**.

### Causa en el código (`swarm.py` → `tick()`)

En cada tick:

1. Se construye una vez la lista de pendientes:
   ```python
   pending_tasks = [t for t in self._task_queue if t.state == TaskState.PENDING]
   ```
2. Se itera sobre **todos** los comportamientos (p. ej. 16 Foragers).
3. Para **cada** comportamiento, `available` se arma desde la **misma** lista `pending_tasks` (no se actualiza cuando una tarea se asigna o se ejecuta).
4. No se “reserva” la tarea al seleccionarla: varios Foragers pueden recibir la misma lista y elegir la **misma** tarea (p. ej. la de mayor estímulo).
5. Cada vez que un comportamiento hace `execute_task(task)` y tiene éxito, se hace `self._tasks_completed += 1`.

Consecuencia: **varios Foragers pueden ejecutar la misma tarea en el mismo tick**. Cada ejecución incrementa `_tasks_completed`, aunque sea la misma tarea. Por eso 8 tareas únicas pueden generar 48 incrementos (por ejemplo, ~6 ejecuciones por tarea en promedio en 3 ticks).

---

## 3. Resumen del análisis

| Métrica              | Valor   | Interpretación |
|----------------------|--------|----------------|
| Tareas únicas        | 8      | Correcto: 8 trabajos de render. |
| Tiempo total         | 0,78 s | Tiempo real de ejecución. |
| Throughput (único)   | 8/0,78 | ~10,2 tareas/s, métrica fiable. |
| Ticks                | 3      | El scheduler vacía la cola en 3 pasadas. |
| tasks_completed (API)| 48     | Cuenta ejecuciones, no tareas; no usar como “tareas únicas completadas”. |

Conclusión: el **resultado del benchmark es válido** (8 tareas pesadas completadas en ~0,78 s), pero el **contador interno del scheduler es engañoso** porque no evita que la misma tarea sea ejecutada por varios workers en el mismo tick.

---

## 4. Corrección aplicada

En `swarm.py` → `tick()` se mantiene un set `claimed_this_tick` con los `task_id` ya asignados en el tick. Al construir `available` para cada comportamiento se excluyen las tareas cuyo `task_id` está en ese set; al seleccionar una tarea se añade su `task_id` a `claimed_this_tick` antes de ejecutarla. Así cada tarea se ejecuta como máximo una vez por tick y `tasks_completed` refleja tareas únicas.
