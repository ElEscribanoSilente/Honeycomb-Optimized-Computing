"""
Reified transitions — methods that *are* state transitions.

The :func:`transition` decorator marks a method as a state-changing
operation. When the method is called, the decorator:

1. Validates that the current state matches ``from_`` (if specified).
2. Executes the method body.
3. On clean return, mutates ``self.state`` to ``to``. The mutation
   passes through the host's ``__setattr__`` (and therefore through
   any wired FSM, e.g. :class:`hoc.swarm.HiveTask`'s wire-up).
4. On exception from the method, **state does not mutate** — the
   exception propagates and ``self.state`` remains at ``from_``.

The decorator stores the ``(from_, to)`` pair on the method as the
``__choreo_transition__`` attribute. This makes the transition
introspectable for documentation, tooling, and future static analysis
extensions to ``choreo``.

Why
---

Phase 4.1 wires ``HiveTask.__setattr__`` so that ``task.state = X``
validates against the FSM. That works, but it leaves the *meaning* of
each state mutation distributed across ``swarm.py`` (a worker reads
your code top-to-bottom and has to infer that ``task.state = RUNNING``
means "claim"). With ``@transition``, the same mutation lives inside
``HiveTask.claim(worker)`` — self-documenting, and the call-site reads
``task.claim(worker)`` instead of ``task.state = TaskState.RUNNING``.

Two APIs coexist:

- **Direct mutation** (``task.state = TaskState.RUNNING``) — still
  valid, still routed through the FSM via ``__setattr__``.
- **Reified call** (``task.claim(worker)``) — additive, no breaking
  change. Existing call-sites in ``swarm.py`` are unchanged.

Usage
-----

::

    from state_machines.reified import transition

    @dataclass
    class HiveTask:
        state: TaskState = TaskState.PENDING

        @transition(from_=TaskState.PENDING, to=TaskState.RUNNING)
        def claim(self, worker: WorkerCell) -> None:
            self._assigned_to = worker.coord

    # caller:
    task.claim(worker)
    assert task.state is TaskState.RUNNING

The decorator also works with classmethods; instance methods are the
common case in HOC.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from .base import IllegalStateTransition

F = TypeVar("F", bound=Callable[..., Any])


def transition(*, from_: Any = None, to: Any) -> Callable[[F], F]:
    """Decorator factory: declare that the wrapped method transitions
    ``self.state`` from ``from_`` to ``to``.

    Parameters
    ----------
    from_:
        The expected current state. ``None`` means "any" — the decorator
        skips the pre-condition check and only mutates state on return.
    to:
        The state to set after the method returns successfully.
    """

    def decorator(method: F) -> F:
        @wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            current = getattr(self, "state", None)
            if from_ is not None and current is not from_:
                raise IllegalStateTransition(
                    fsm_name=type(self).__name__,
                    source=getattr(current, "name", str(current)),
                    target=getattr(to, "name", str(to)),
                    reason="reified_from_mismatch",
                )

            result = method(self, *args, **kwargs)

            # Mutate state via __setattr__ so any wired FSM
            # validates the transition. If the wire-up rejects, the
            # IllegalStateTransition propagates from there — meaning
            # the user has decorated a method whose ``to`` is not
            # reachable from ``from_`` in the underlying FSM.
            self.state = to
            return result

        wrapper.__choreo_transition__ = (from_, to)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
