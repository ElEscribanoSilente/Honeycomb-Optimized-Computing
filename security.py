"""
HOC Security - Secure serialization, HMAC auth, cryptographic RNG, path safety
==============================================================================

Este módulo centraliza todas las primitivas de seguridad utilizadas en Fase 2:

1. **Serialización segura** via `mscs` (autor: @Esraderey, MIT, zero deps):
   - Registry de clases (no ejecuta código arbitrario como `pickle`)
   - HMAC-SHA256 nativo para autenticación criptográfica
   - Verificación en deserialización

2. **RNG criptográfico** (`secrets.SystemRandom`) para decisiones sensibles:
   - Sustituye `random.random()` en caminos de elección de trabajo,
     shuffle de roles y desempates en scheduling.

3. **Autenticación de mensajes** (HMAC-SHA256) para DanceMessage,
   RoyalMessage y PheromoneDeposit (Fase 2.2).

4. **Rate limiting** de APIs públicas (submit_task, execute_on_cell).

5. **Path validation** contra traversal en HoneyArchive / checkpoints.

6. **Log sanitization**: filtrado de detalles de excepción en producción.

Config:
- Variable de entorno ``HOC_HMAC_KEY`` (hex o texto utf-8, ≥16 bytes) para
  compartir clave entre procesos. En ausencia, se genera una clave efímera
  por proceso (apta para dev/test, NO para clústeres multi-proceso).

Notas sobre ``mscs`` (v2.4.0):
El roadmap menciona ``mscs.Registry()`` y ``mscs.Serializer()``; la API real
en la versión instalada es funcional (``mscs.register(cls)`` a nivel módulo
y ``mscs.dumps/loads(..., hmac_key=...)``). Este módulo envuelve esa API
para aislar al resto del proyecto de cambios futuros.
"""

from __future__ import annotations

import functools
import hashlib
import hmac as _hmac
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Set, TypeVar, Union

import mscs

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# ═══════════════════════════════════════════════════════════════════════════════
# HMAC KEY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

_MIN_KEY_BYTES: int = 16
_HMAC_KEY: Optional[bytes] = None
_HMAC_KEY_LOCK = threading.Lock()


def _load_env_key() -> Optional[bytes]:
    """Lee HOC_HMAC_KEY de entorno. Intenta hex, si falla trata como utf-8."""
    env = os.environ.get("HOC_HMAC_KEY")
    if not env:
        return None
    try:
        raw = bytes.fromhex(env)
    except ValueError:
        raw = env.encode("utf-8")
    if len(raw) < _MIN_KEY_BYTES:
        raise ValueError(
            f"HOC_HMAC_KEY demasiado corta: {len(raw)} bytes, "
            f"mínimo {_MIN_KEY_BYTES}"
        )
    return raw


def get_hmac_key() -> bytes:
    """
    Devuelve la clave HMAC activa.

    Orden de resolución:
    1. Clave configurada explícitamente vía ``set_hmac_key()``.
    2. Variable de entorno ``HOC_HMAC_KEY`` (hex preferido).
    3. Clave efímera aleatoria (32 bytes) generada una vez por proceso.

    Producción debería establecer ``HOC_HMAC_KEY`` explícitamente para que
    todos los workers compartan la misma clave.
    """
    global _HMAC_KEY
    with _HMAC_KEY_LOCK:
        if _HMAC_KEY is None:
            env_key = _load_env_key()
            if env_key is not None:
                _HMAC_KEY = env_key
            else:
                _HMAC_KEY = secrets.token_bytes(32)
                logger.info(
                    "HOC HMAC key generada efímeramente (%d bytes). "
                    "Para persistencia multi-proceso define HOC_HMAC_KEY.",
                    len(_HMAC_KEY),
                )
        return _HMAC_KEY


def set_hmac_key(key: bytes) -> None:
    """Fija la clave HMAC (útil para tests y para configuración explícita)."""
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("hmac key debe ser bytes")
    if len(key) < _MIN_KEY_BYTES:
        raise ValueError(
            f"hmac key demasiado corta: {len(key)} bytes, mínimo {_MIN_KEY_BYTES}"
        )
    global _HMAC_KEY
    with _HMAC_KEY_LOCK:
        _HMAC_KEY = bytes(key)


def reset_hmac_key() -> None:
    """Reinicia la clave para que la próxima llamada la regenere. Solo tests."""
    global _HMAC_KEY
    with _HMAC_KEY_LOCK:
        _HMAC_KEY = None


# ═══════════════════════════════════════════════════════════════════════════════
# MSCS SERIALIZATION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

MSCSecurityError = mscs.MSCSecurityError

_REGISTERED: Set[type] = set()
_REGISTRY_LOCK = threading.Lock()


def register(cls: type) -> type:
    """
    Registra una clase como segura para deserialización con ``mscs``.

    Idempotente: registrar dos veces no falla. Thread-safe.
    Uso como decorador:

        @register
        @dataclass
        class MyObject:
            x: int
    """
    with _REGISTRY_LOCK:
        if cls not in _REGISTERED:
            mscs.register(cls)
            _REGISTERED.add(cls)
    return cls


def is_registered(cls: type) -> bool:
    return cls in _REGISTERED


def serialize(obj: Any, *, sign: bool = True) -> bytes:
    """
    Serializa con ``mscs`` y firma HMAC-SHA256 por defecto.

    Args:
        obj: objeto a serializar. Su tipo debe estar registrado si es custom.
        sign: si True (default), firma con HMAC. False solo para datos
              no-confidenciales donde la verificación no aplica (p.e. caches
              locales cuyo contenido nunca se acepta como input).
    """
    if sign:
        return mscs.dumps(obj, hmac_key=get_hmac_key())
    return mscs.dumps(obj)


def deserialize(data: bytes, *, verify: bool = True, strict: bool = True) -> Any:
    """
    Deserializa con verificación HMAC y registro estricto.

    Args:
        data: bytes producidos por ``serialize``.
        verify: si True (default), verifica HMAC-SHA256. Rechaza con
                ``MSCSecurityError`` si falla.
        strict: si True (default), rechaza clases no registradas.
    """
    if verify:
        return mscs.loads(data, strict=strict, hmac_key=get_hmac_key())
    return mscs.loads(data, strict=strict)


# ═══════════════════════════════════════════════════════════════════════════════
# HMAC MESSAGE SIGNING (para DanceMessage, RoyalMessage, PheromoneDeposit)
# ═══════════════════════════════════════════════════════════════════════════════

def sign_payload(payload: bytes, key: Optional[bytes] = None) -> bytes:
    """Genera tag HMAC-SHA256 sobre ``payload`` (32 bytes)."""
    return _hmac.new(key or get_hmac_key(), payload, hashlib.sha256).digest()


def verify_signature(payload: bytes, tag: bytes, key: Optional[bytes] = None) -> bool:
    """Verifica tag HMAC en tiempo constante (resistente a timing attacks)."""
    if not isinstance(tag, (bytes, bytearray)) or len(tag) != 32:
        return False
    expected = sign_payload(payload, key=key)
    return _hmac.compare_digest(expected, bytes(tag))


# ═══════════════════════════════════════════════════════════════════════════════
# RNG CRIPTOGRÁFICO
# ═══════════════════════════════════════════════════════════════════════════════

_system_rng = secrets.SystemRandom()


def secure_random() -> float:
    """Float aleatorio en [0, 1) usando ``secrets.SystemRandom`` (CSPRNG)."""
    return _system_rng.random()


def secure_choice(seq):
    """Elemento aleatorio de ``seq`` usando CSPRNG. Lanza IndexError si vacío."""
    return _system_rng.choice(seq)


def secure_shuffle(lst: list) -> list:
    """Shuffle in-place de ``lst`` usando CSPRNG. Retorna la misma lista."""
    _system_rng.shuffle(lst)
    return lst


# ═══════════════════════════════════════════════════════════════════════════════
# PATH VALIDATION (contra path traversal)
# ═══════════════════════════════════════════════════════════════════════════════

class PathTraversalError(ValueError):
    """Intento de escapar de un directorio base detectado."""


def safe_join(base: Union[str, Path], relative: str) -> Path:
    """
    Resuelve ``base / relative`` asegurando que el resultado esté contenido
    en ``base``. Rechaza `..`, symlinks fuera, rutas absolutas, null bytes.

    Returns:
        Path absoluto resuelto.

    Raises:
        PathTraversalError: si ``relative`` intenta escapar de ``base``.
    """
    if not isinstance(relative, str):
        raise PathTraversalError(f"relative debe ser str, recibido: {type(relative).__name__}")
    if "\x00" in relative:
        raise PathTraversalError("null byte en path")
    # Rechaza rutas absolutas (e.g. "/etc/passwd" o "C:\\Windows")
    if Path(relative).is_absolute():
        raise PathTraversalError(f"path absoluto no permitido: {relative!r}")

    base_resolved = Path(base).resolve()
    # Usar / sobre Path acepta separadores ambiguos; resolve elimina '..' y symlinks.
    candidate = (base_resolved / relative).resolve()

    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        raise PathTraversalError(
            f"path traversal detectado: {relative!r} escapa de {str(base_resolved)!r}"
        )
    return candidate


# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITING (token bucket)
# ═══════════════════════════════════════════════════════════════════════════════

class RateLimitExceeded(RuntimeError):
    """Excepción lanzada cuando se supera el ritmo permitido."""


class RateLimiter:
    """
    Token bucket rate limiter thread-safe.

    - ``per_second``: tokens añadidos por segundo.
    - ``burst``: capacidad máxima del bucket (permite ráfagas cortas).
    """

    def __init__(self, per_second: float, burst: Optional[int] = None):
        if per_second <= 0:
            raise ValueError(f"per_second debe ser > 0, recibido: {per_second!r}")
        self.rate: float = float(per_second)
        self.burst: int = int(burst) if burst is not None else max(1, int(per_second))
        if self.burst <= 0:
            raise ValueError(f"burst debe ser > 0, recibido: {burst!r}")
        self._tokens: float = float(self.burst)
        self._last: float = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self, n: int = 1) -> bool:
        """Consume ``n`` tokens si hay disponibles. Retorna True si OK."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def __call__(self, fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not self.try_acquire():
                raise RateLimitExceeded(
                    f"{fn.__name__} rate limit exceeded "
                    f"({self.rate}/s, burst={self.burst})"
                )
            return fn(*args, **kwargs)
        return wrapper  # type: ignore[return-value]


def rate_limit(per_second: float, burst: Optional[int] = None) -> Callable[[F], F]:
    """Decorator factory. Cada decorador tiene su propio bucket."""
    limiter = RateLimiter(per_second, burst)

    def decorator(fn: F) -> F:
        return limiter(fn)

    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# LOG SANITIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _is_debug_mode() -> bool:
    """Lee ``HOC_DEBUG`` de entorno. Default: False (producción)."""
    return os.environ.get("HOC_DEBUG", "").lower() in ("1", "true", "yes", "on")


def sanitize_error(exc: BaseException) -> str:
    """
    Retorna una representación de excepción segura para logs.

    En producción (default): solo el tipo (``ValueError``).
    En debug (``HOC_DEBUG=1``): tipo + mensaje.

    Nunca incluye traceback ni ``repr`` del objeto; eso es solo para el
    handler de debug.
    """
    if _is_debug_mode():
        return f"{type(exc).__name__}: {exc}"
    return type(exc).__name__


__all__ = [
    # HMAC key management
    "get_hmac_key",
    "set_hmac_key",
    "reset_hmac_key",
    # Serialization
    "MSCSecurityError",
    "register",
    "is_registered",
    "serialize",
    "deserialize",
    # Message signing
    "sign_payload",
    "verify_signature",
    # RNG
    "secure_random",
    "secure_choice",
    "secure_shuffle",
    # Path safety
    "PathTraversalError",
    "safe_join",
    # Rate limiting
    "RateLimitExceeded",
    "RateLimiter",
    "rate_limit",
    # Logging
    "sanitize_error",
]
