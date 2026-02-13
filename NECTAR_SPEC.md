# NectarFlow / Feromona Digital – Especificación

| Campo | Valor |
|-------|--------|
| **Módulo** | HOC (Honeycomb Optimized Computing) |
| **Componentes** | Feromona digital, Waggle Dance, difusión hexagonal |
| **Referencia** | `vent_engine/HOC/nectar.py`, `vent_engine/HOC/core.py` |

---

## 1. Estructura de la feromona digital

### 1.1 Campos del depósito

Cada depósito de feromona se modela como:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| **tipo** | `PheromoneType` | Semántica del mensaje (ver tabla de tipos). |
| **intensidad** | `float` ∈ [0, 1] (por celda) o [0, max_intensity] (trail global) | Cantidad de señal; puede sumarse hasta un tope por celda. |
| **timestamp** | `float` | Momento del depósito (para decaimiento temporal). |
| **source** | `HexCoord \| None` | Celda origen (opcional). |
| **decay_rate** | `float` ∈ [0, 1] | Constante de decaimiento por tipo (por defecto según tipo). |
| **metadata** | `Dict[str, Any]` | Datos opcionales (ej. entity_id, task_id). |

### 1.2 Tipos de feromona

| Tipo | Uso | Decaimiento (default) | Persistencia |
|------|-----|------------------------|--------------|
| `TRAIL` | Camino general / rastro | 0.05 | Alta |
| `FOOD` | Recurso encontrado | 0.03 | Muy alta |
| `DANGER` | Peligro / error | 0.15 | Baja |
| `BUSY` | Celda ocupada | 0.20 | Muy baja |
| `AVAILABLE` | Celda disponible | 0.10 | Media |
| `RECRUITMENT` | Reclutamiento de ayuda | 0.08 | Alta |
| `ALARM` | Alerta general | 0.25 | Muy baja |
| `SUCCESS` | Tarea completada bien | 0.02 | Muy alta |
| `FAILURE` | Tarea fallida | 0.10 | Media |

En **core** (por celda) se usan además: `PATH`, `RECRUIT`, `HOME`, `WORK`, `EXPLORATION`. La intensidad por celda se mantiene en [0, 1]; en `PheromoneTrail` (global) el techo es configurable (`max_intensity`, ej. 10.0).

### 1.3 Decaimiento

- **Fórmula estándar (exponencial):**  
  `I_new = I * exp(-decay_rate * elapsed)`  
  donde `elapsed` es el tiempo transcurrido desde `timestamp` (en unidades de tick o segundos según configuración).

- **Estrategias alternativas** (`PheromoneDecay`):  
  `LINEAR`, `STEP`, `NONE` (ver `nectar.PheromoneTrail`).

- **Umbral de limpieza:** depósitos con `intensity < 0.001` se eliminan.

---

## 2. Protocolo Waggle Dance

### 2.1 Información transmitida

Cada mensaje de danza (`DanceMessage`) codifica:

| Campo | Tipo | Significado |
|-------|------|-------------|
| **source** | `HexCoord` | Celda que ejecuta la danza (origen del mensaje). |
| **direction** | `DanceDirection` | Dirección al recurso (ángulo 0–360°, mapeado a hex). |
| **distance** | `int` | Distancia al recurso en número de celdas hexagonales. |
| **quality** | `float` ∈ [0, 1] | Calidad/valor del recurso; mayor = más atractivo. |
| **resource_type** | `str` | Tipo de recurso (ej. `"food"`, `"work"`, `"generic"`). |
| **timestamp** | `float` | Momento de creación. |
| **ttl** | `int` | Time-to-live (número de saltos de propagación restantes). |
| **metadata** | `Dict[str, Any]` | Extensión (ej. prioridad, entity_id). |

### 2.2 Semántica (inspiración abeja)

- **Dirección:** equivalente al ángulo de la danza respecto a la “referencia solar” (en el grid, eje de referencia).
- **Distancia:** proporcional a la “duración” del tramo de danza (más celdas = más lejos).
- **Calidad:** vigor de la danza; influye en atenuación y competencia entre mensajes.

### 2.3 Codificación para transmisión

- **Formato compacto (bytes):**  
  `resource_type | direction.value | distance | quality`  
  (ej. `"food|60|5|0.90"`).

- **Propagación:** broadcast direccional; la señal se propaga más fuerte en `direction` y se atenúa en la dirección opuesta (`attenuation` y `competition_threshold` en `WaggleDance`).

### 2.4 Uso típico

- Indicar ubicación de **trabajo disponible** o **recurso**.
- **Reclutamiento:** otras celdas pueden seguir `direction` y `distance` (vía `target_coord()`) para acercarse al recurso.
- **Competencia:** en cada celda se mantienen solo un número limitado de danzas (ej. las 3 de mayor calidad).

---

## 3. Actualización y difusión en topología hexagonal

### 3.1 Topología

- Cada celda tiene **exactamente 6 vecinos** (direcciones `HexDirection`: NW, NE, E, SE, SW, W).
- Vecino de `coord` en dirección `d`: `coord.neighbor(d)`.
- Solo se considera difusión entre celdas que existen en el grid (si se usa `valid_coords` o acceso por grid).

### 3.2 Orden de actualización por tick

1. **Decaimiento:** en cada celda (o en cada depósito del trail), aplicar decaimiento a todas las feromonas según `elapsed` y `decay_rate`.
2. **Difusión:** repartir una fracción de la intensidad de cada tipo a los 6 vecinos.

Así se evita que la difusión use intensidades ya decaídas del mismo tick.

### 3.3 Fórmula de difusión

- **Por celda:**  
  Para cada tipo con `intensity > umbral` (ej. 0.01):  
  `diffuse_amount = intensity * diffusion_rate / 6`  
  y se deposita `diffuse_amount` en cada uno de los 6 vecinos (mismo tipo, mismo `decay_rate`).

- **Parámetros en configuración (core):**
  - `pheromone_decay_rate`: decaimiento por tick (default 0.1).
  - `pheromone_diffusion_rate`: fracción que se reparte a vecinos (default 0.05).

### 3.4 Dónde se aplica

- **En `HoneycombGrid` (core):**  
  `_update_pheromones()` recorre las celdas, llama `cell.decay_pheromones(elapsed)` y luego `cell.diffuse_pheromones()` (cada celda deposita en sus 6 vecinos).

- **En NectarFlow (nectar.py):**  
  `PheromoneTrail` puede usar `evaporate()` (decaimiento) y `diffuse_to_neighbors(rate, valid_coords)` para difusión sobre coordenadas hexagonales (vecinos vía `HexCoord.neighbor(d)`).  
  `NectarFlow.tick()` ejecuta evaporación, difusión (si está habilitada) y propagación de danzas.

### 3.5 Resumen de flujo por tick

```
Para cada tick del panal:
  1. Decaimiento: todas las celdas / todos los depósitos del trail.
  2. Difusión: por cada celda/depósito, repartir intensity * rate / 6 a los 6 vecinos.
  3. Waggle: propagar DanceMessages con atenuación direccional y TTL.
  4. Royal Jelly: procesar comandos pendientes (sin cambio en feromonas).
```

---

## 4. Referencia rápida de código

| Concepto | Archivo | Clase / función |
|----------|---------|------------------|
| Feromona (trail global) | `nectar.py` | `PheromoneTrail`, `PheromoneDeposit`, `PheromoneType` |
| Feromona (por celda) | `core.py` | `PheromoneField`, `PheromoneDeposit`, `PheromoneType` |
| Decaimiento por tipo | `nectar.py` | `PheromoneType.decay_rate()` |
| Waggle Dance | `nectar.py` | `WaggleDance`, `DanceMessage`, `DanceDirection` |
| Decaimiento + difusión grid | `core.py` | `HoneycombGrid._update_pheromones()`, `HoneycombCell.diffuse_pheromones()` |
| Difusión en NectarFlow | `nectar.py` | `PheromoneTrail.diffuse_to_neighbors()`, `NectarFlow.tick()` |
