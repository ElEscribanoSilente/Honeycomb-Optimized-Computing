# HOC State Machines

> **Auto-generated** by `scripts/generate_state_machines_md.py`. Do not edit by hand.
> Regenerate after touching any FSM in `state_machines/`. CI fails if this file drifts from the FSM specs.

Each diagram below is exported from a `HocStateMachine` instance via `to_mermaid()`. Triggers shown in the labels are the synthetic names auto-generated from `<source>__to__<dest>`; explicit triggers appear when an FSM module declares them.

## Index

- [CellState](#cellstate)
- [PheromoneDeposit](#pheromonedeposit)
- [TaskLifecycle](#tasklifecycle)
- [QueenSuccession](#queensuccession)
- [FailoverFlow](#failoverflow)

## CellState

- **States** (7): `ACTIVE`, `EMPTY`, `FAILED`, `IDLE`, `MIGRATING`, `RECOVERING`, `SEALED`
- **Initial state**: `EMPTY`

```mermaid
stateDiagram-v2
    EMPTY --> IDLE : vcore_added
    IDLE --> EMPTY : vcore_drained_idle
    ACTIVE --> EMPTY : vcore_drained_active
    IDLE --> ACTIVE : tick_started
    ACTIVE --> IDLE : tick_completed
    ACTIVE --> FAILED : tick_failed
    FAILED --> RECOVERING : recovery_started
    RECOVERING --> EMPTY : recovery_completed
    RECOVERING --> IDLE : recovery_restored
    ACTIVE --> FAILED : admin_mark_failed
    EMPTY --> FAILED : admin_mark_failed
    FAILED --> FAILED : admin_mark_failed
    IDLE --> FAILED : admin_mark_failed
    MIGRATING --> FAILED : admin_mark_failed
    RECOVERING --> FAILED : admin_mark_failed
    SEALED --> FAILED : admin_mark_failed
    ACTIVE --> IDLE : admin_set_idle
    EMPTY --> IDLE : admin_set_idle
    FAILED --> IDLE : admin_set_idle
    IDLE --> IDLE : admin_set_idle
    MIGRATING --> IDLE : admin_set_idle
    RECOVERING --> IDLE : admin_set_idle
    SEALED --> IDLE : admin_set_idle
    ACTIVE --> RECOVERING : admin_recover
    EMPTY --> RECOVERING : admin_recover
    FAILED --> RECOVERING : admin_recover
    IDLE --> RECOVERING : admin_recover
    MIGRATING --> RECOVERING : admin_recover
    RECOVERING --> RECOVERING : admin_recover
    SEALED --> RECOVERING : admin_recover
    ACTIVE --> EMPTY : admin_reset
    EMPTY --> EMPTY : admin_reset
    FAILED --> EMPTY : admin_reset
    IDLE --> EMPTY : admin_reset
    MIGRATING --> EMPTY : admin_reset
    RECOVERING --> EMPTY : admin_reset
    SEALED --> EMPTY : admin_reset
    ACTIVE --> ACTIVE : admin_force_active
    EMPTY --> ACTIVE : admin_force_active
    FAILED --> ACTIVE : admin_force_active
    IDLE --> ACTIVE : admin_force_active
    MIGRATING --> ACTIVE : admin_force_active
    RECOVERING --> ACTIVE : admin_force_active
    SEALED --> ACTIVE : admin_force_active
    ACTIVE --> MIGRATING : admin_start_migration
    EMPTY --> MIGRATING : admin_start_migration
    FAILED --> MIGRATING : admin_start_migration
    IDLE --> MIGRATING : admin_start_migration
    MIGRATING --> MIGRATING : admin_start_migration
    RECOVERING --> MIGRATING : admin_start_migration
    SEALED --> MIGRATING : admin_start_migration
    ACTIVE --> SEALED : admin_seal
    EMPTY --> SEALED : admin_seal
    FAILED --> SEALED : admin_seal
    IDLE --> SEALED : admin_seal
    MIGRATING --> SEALED : admin_seal
    RECOVERING --> SEALED : admin_seal
    SEALED --> SEALED : admin_seal
```

## PheromoneDeposit

- **States** (4): `DECAYING`, `DIFFUSING`, `EVAPORATED`, `FRESH`
- **Initial state**: `FRESH`

```mermaid
stateDiagram-v2
    FRESH --> DECAYING : aged_out [guarded]
    DECAYING --> DIFFUSING : diffusion_started [guarded]
    DIFFUSING --> DECAYING : diffusion_completed
    DECAYING --> EVAPORATED : intensity_below_cleanup [guarded]
    DIFFUSING --> EVAPORATED : intensity_below_cleanup_during_diffuse [guarded]
```

## TaskLifecycle

- **States** (5): `CANCELLED`, `COMPLETED`, `FAILED`, `PENDING`, `RUNNING`
- **Initial state**: `PENDING`

```mermaid
stateDiagram-v2
    PENDING --> RUNNING : claimed
    RUNNING --> COMPLETED : completed
    RUNNING --> FAILED : failed
    FAILED --> PENDING : retry
    PENDING --> CANCELLED : cancelled_pending
    RUNNING --> CANCELLED : cancelled_running
    PENDING --> COMPLETED : force_completed_from_pending
    PENDING --> FAILED : force_failed_from_pending
```

## QueenSuccession

- **States** (6): `DETECTING`, `ELECTED`, `FAILED`, `NOMINATING`, `STABLE`, `VOTING`
- **Initial state**: `STABLE`

```mermaid
stateDiagram-v2
    STABLE --> DETECTING : heartbeat_lost [guarded]
    DETECTING --> NOMINATING : failure_confirmed [guarded]
    DETECTING --> STABLE : false_alarm
    NOMINATING --> VOTING : candidates_identified [guarded]
    NOMINATING --> FAILED : no_candidates
    VOTING --> ELECTED : quorum_reached [guarded]
    VOTING --> FAILED : tally_rejected
    ELECTED --> STABLE : queen_promoted
    FAILED --> STABLE : cooldown_expired
```

## FailoverFlow

- **States** (5): `DEGRADED`, `HEALTHY`, `LOST`, `MIGRATING`, `RECOVERED`
- **Initial state**: `HEALTHY`

```mermaid
stateDiagram-v2
    HEALTHY --> DEGRADED : circuit_opened [guarded]
    DEGRADED --> MIGRATING : strategy_picked [guarded]
    MIGRATING --> RECOVERED : migration_succeeded [guarded]
    MIGRATING --> LOST : migration_failed
    LOST --> HEALTHY : cells_provisioned [guarded]
    RECOVERED --> HEALTHY : stabilized [guarded]
```

