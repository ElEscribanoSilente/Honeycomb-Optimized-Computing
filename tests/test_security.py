"""
Tests de seguridad de Fase 2 — HOC Honeycomb Optimized Computing
=================================================================

Verifican que los hardenings de seguridad implementados en Fase 2 cierran
efectivamente los vectores de ataque correspondientes. Cada clase de test
cubre una de las cinco áreas obligatorias del roadmap:

1. ``TestMscsRejectsMalicious``  — payload estilo pickle RCE no ejecuta.
2. ``TestRoyalCommandQueenOnly`` — alta prioridad requiere firma de Queen.
3. ``TestQuorumSignedVotes``     — votos duplicados/inválidos rechazados.
4. ``TestPheromoneBoundedDoS``   — flood no agota memoria.
5. ``TestHoneyArchivePathTraversal`` — traversal rechazado.

Además cubre el HMAC round-trip global, el CSPRNG replacement y rate
limiting para completar la cobertura del módulo ``security``.
"""

from __future__ import annotations

import os
import pickle

import pytest

from hoc.core import HexCoord, HoneycombConfig, HoneycombGrid
from hoc.memory import CombStorage, HiveMemory, HoneyArchive, MemoryConfig
from hoc.nectar import (
    DanceDirection,
    DanceMessage,
    NectarFlow,
    PheromoneTrail,
    PheromoneType,
    RoyalCommand,
    RoyalJelly,
)
from hoc.resilience import QueenSuccession, ResilienceConfig, Vote
from hoc.security import (
    MSCSecurityError,
    PathTraversalError,
    RateLimiter,
    RateLimitExceeded,
    deserialize,
    rate_limit,
    safe_join,
    secure_choice,
    secure_random,
    serialize,
    set_hmac_key,
    sign_payload,
    verify_signature,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures comunes
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def fixed_hmac_key():
    """Fija una clave HMAC determinista por test para reproducibilidad."""
    set_hmac_key(b"phase02-security-test-key-32bytes!!")
    yield
    set_hmac_key(b"phase02-security-test-key-32bytes!!")


@pytest.fixture
def tmp_grid():
    """Grid pequeño para tests que requieren topología."""
    cfg = HoneycombConfig(radius=1, vcores_per_cell=1)
    return HoneycombGrid(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. mscs rechaza payloads maliciosos
# ═══════════════════════════════════════════════════════════════════════════════


class EvilReduce:
    """
    Clase que si fuera deserializada por ``pickle.loads`` ejecutaría
    ``os.system`` — el clásico payload RCE de pickle. ``mscs`` rechaza
    bytes que no son de su propio formato por construcción (no
    interpreta el opcode pickle en absoluto).
    """

    def __reduce__(self):
        return (os.system, ("echo pwned > /tmp/hoc_pwned",))


class TestMscsRejectsMalicious:
    """
    Fase 2.1 — Reemplazo de pickle por mscs. Payloads construidos con
    ``pickle.dumps`` de clases con ``__reduce__`` malicioso NO deben
    producir ejecución cuando se alimentan a ``mscs.loads``.
    """

    def test_pickle_rce_bytes_are_rejected_by_mscs(self):
        """Payload pickle clásico RCE → mscs lo rechaza sin ejecutar."""
        malicious_bytes = pickle.dumps(EvilReduce())
        # El "marcador mágico" de mscs no coincide con el header pickle,
        # así que mscs falla en parse antes de reconstruir nada.
        with pytest.raises((MSCSecurityError, Exception)):
            deserialize(malicious_bytes, verify=True)

    def test_hmac_tampered_bytes_rejected(self):
        """Bytes válidos + firma HMAC → flip de 1 byte rompe la verificación."""
        valid = serialize({"x": 1, "y": 2})
        tampered = valid[:-1] + bytes([valid[-1] ^ 0xFF])
        with pytest.raises(MSCSecurityError):
            deserialize(tampered, verify=True)

    def test_foreign_hmac_key_rejected(self):
        """Bytes firmados con otra clave → nuestra deserialización rechaza."""
        valid = serialize({"x": 1}, sign=True)
        # Cambiar clave y verificar que no valida.
        set_hmac_key(b"another-key-totally-different-xxxxxx")
        with pytest.raises(MSCSecurityError):
            deserialize(valid, verify=True)

    def test_strict_registry_rejects_unregistered_class(self):
        """Una clase no registrada no puede ser reconstruida vía mscs loads."""

        class LocalOnly:
            pass

        import mscs

        # dumps funciona, pero loads strict debe rechazar sin register.
        raw = mscs.dumps(LocalOnly())
        with pytest.raises(MSCSecurityError):
            mscs.loads(raw, strict=True)

    def test_combstorage_rejects_tampered_bytes(self, tmp_grid):
        """CombStorage + modificación de bytes almacenados → get retorna None."""
        comb = CombStorage(tmp_grid, MemoryConfig(comb_compression_enabled=False))
        comb.put("k", {"data": "ok"})
        # Localizar la celda donde está almacenada y manipular el byte.
        coord = comb._hash_to_coord("k")
        cell = comb._cells[coord]
        original = cell.data["k"]
        cell.data["k"] = original[:-1] + bytes([original[-1] ^ 0xFF])
        assert comb.get("k") is None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RoyalCommand — alta prioridad solo por Queen
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoyalCommandQueenOnly:
    """
    Fase 2.2 — Solo la :class:`QueenCell` puede emitir comandos con
    ``priority >= 8``. Un DroneCell/WorkerCell forjando un EMERGENCY
    debe ser rechazado aunque posea la clave HMAC.
    """

    def test_drone_cannot_issue_priority_10(self):
        """DroneCell → PermissionError en emisión con priority=10."""
        queen = HexCoord.origin()
        drone = HexCoord(2, -1)  # cualquier celda != queen
        jelly = RoyalJelly(queen)
        with pytest.raises(PermissionError):
            jelly.issue_command(RoyalCommand.EMERGENCY, priority=10, issuer=drone)

    def test_queen_can_issue_priority_10(self):
        """Queen legítima → comando aceptado y firmado."""
        queen = HexCoord.origin()
        jelly = RoyalJelly(queen)
        msg = jelly.issue_command(RoyalCommand.EMERGENCY, priority=10, issuer=queen)
        assert msg.priority == 10
        assert msg.issuer == queen
        assert msg.signature is not None
        assert msg.verify()

    def test_priority_low_no_issuer_required(self):
        """Priority < 8 no exige issuer (preserva compatibilidad)."""
        jelly = RoyalJelly(HexCoord.origin())
        msg = jelly.issue_command(RoyalCommand.BALANCE, priority=5)
        assert msg.priority == 5

    def test_priority_8_requires_explicit_issuer(self):
        """Priority exactamente 8 exige issuer explícito."""
        jelly = RoyalJelly(HexCoord.origin())
        # issuer=None es rechazado en el threshold exacto
        with pytest.raises(PermissionError):
            jelly.issue_command(RoyalCommand.REINFORCE, priority=8, issuer=None)

    def test_update_queen_coord_changes_enforcement(self):
        """Tras update_queen_coord, el nuevo issuer legítimo es la nueva reina."""
        old_queen = HexCoord.origin()
        new_queen = HexCoord(1, 0)
        jelly = RoyalJelly(old_queen)
        jelly.update_queen_coord(new_queen)
        # La vieja reina ya no puede emitir high priority.
        with pytest.raises(PermissionError):
            jelly.issue_command(RoyalCommand.EMERGENCY, priority=10, issuer=old_queen)
        # Pero la nueva sí.
        msg = jelly.issue_command(RoyalCommand.EMERGENCY, priority=10, issuer=new_queen)
        assert msg.issuer == new_queen

    def test_forged_royal_message_fails_verify(self):
        """Un mensaje con params manipulados post-firma falla verify()."""
        queen = HexCoord.origin()
        jelly = RoyalJelly(queen)
        msg = jelly.issue_command(
            RoyalCommand.EVACUATE, priority=9, issuer=queen, params={"radius": 3}
        )
        assert msg.verify()
        # Atacante modifica params para ampliar el radio de evacuación.
        msg.params["radius"] = 100
        assert not msg.verify()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Quorum con votos firmados y rechazo de duplicados
# ═══════════════════════════════════════════════════════════════════════════════


class TestQuorumSignedVotes:
    """
    Fase 2.3 — ``QueenSuccession._tally_votes`` rechaza:
    - votos sin firma o con firma inválida,
    - votos duplicados del mismo voter,
    - votos para candidatos no registrados,
    - votos de términos anteriores (anti-replay).
    """

    def test_duplicate_voter_rejected(self, tmp_grid):
        """Mismo voter vota dos veces → solo el primero cuenta."""
        succession = QueenSuccession(tmp_grid, ResilienceConfig())
        succession._term_number = 1
        candidates = {HexCoord(1, 0), HexCoord(-1, 0)}
        voter = HexCoord(0, 1)
        votes = [
            Vote(voter=voter, candidate=HexCoord(1, 0), term=1).sign(),
            Vote(voter=voter, candidate=HexCoord(-1, 0), term=1).sign(),
        ]
        # Solo un voto contado → no hay mayoría con 1 voto entre 2 candidatos.
        result = succession._tally_votes(votes, candidates, expected_term=1)
        # 1 voto total, majority threshold = 1 → pasa.
        assert result == HexCoord(1, 0)

    def test_unsigned_vote_rejected(self, tmp_grid):
        """Voto sin firma → rechazado."""
        succession = QueenSuccession(tmp_grid, ResilienceConfig())
        candidates = {HexCoord(1, 0)}
        votes = [
            Vote(voter=HexCoord(0, 1), candidate=HexCoord(1, 0), term=1),
            # signature=None → rejected
        ]
        assert succession._tally_votes(votes, candidates, expected_term=1) is None

    def test_tampered_signature_rejected(self, tmp_grid):
        """Firma manipulada → rechazada."""
        succession = QueenSuccession(tmp_grid, ResilienceConfig())
        candidates = {HexCoord(1, 0)}
        v = Vote(voter=HexCoord(0, 1), candidate=HexCoord(1, 0), term=1).sign()
        v.signature = v.signature[:-1] + bytes([v.signature[-1] ^ 0xFF])
        assert succession._tally_votes([v], candidates, expected_term=1) is None

    def test_wrong_term_rejected(self, tmp_grid):
        """Voto de otro term → rechazado (anti-replay)."""
        succession = QueenSuccession(tmp_grid, ResilienceConfig())
        candidates = {HexCoord(1, 0)}
        v = Vote(voter=HexCoord(0, 1), candidate=HexCoord(1, 0), term=5).sign()
        assert succession._tally_votes([v], candidates, expected_term=1) is None

    def test_unknown_candidate_rejected(self, tmp_grid):
        """Voto por candidato fuera del set oficial → rechazado."""
        succession = QueenSuccession(tmp_grid, ResilienceConfig())
        candidates = {HexCoord(1, 0)}
        v = Vote(voter=HexCoord(0, 1), candidate=HexCoord(5, 5), term=1).sign()
        assert succession._tally_votes([v], candidates, expected_term=1) is None

    def test_majority_enforced(self, tmp_grid):
        """Se requiere >50% — empate no elige ganador."""
        succession = QueenSuccession(tmp_grid, ResilienceConfig())
        candidates = {HexCoord(1, 0), HexCoord(-1, 0)}
        votes = [
            Vote(voter=HexCoord(0, 1), candidate=HexCoord(1, 0), term=1).sign(),
            Vote(voter=HexCoord(0, -1), candidate=HexCoord(-1, 0), term=1).sign(),
        ]
        # 1 vs 1 → no hay mayoría (threshold = 2).
        assert succession._tally_votes(votes, candidates, expected_term=1) is None

    def test_term_number_monotonic(self, tmp_grid):
        """Cada _conduct_election incrementa term en 1."""
        succession = QueenSuccession(tmp_grid, ResilienceConfig())
        t0 = succession.current_term
        succession._conduct_election([HexCoord.origin()])
        assert succession.current_term == t0 + 1
        succession._conduct_election([HexCoord.origin()])
        assert succession.current_term == t0 + 2


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PheromoneTrail bounded — DoS mitigado
# ═══════════════════════════════════════════════════════════════════════════════


class TestPheromoneBoundedDoS:
    """
    Fase 2.4 — ``PheromoneTrail`` acota memoria incluso bajo flood.
    """

    def test_10k_deposits_same_coord_bounded(self):
        """
        10K deposits en UNA coord → memoria acotada por len(PheromoneType).
        Cada ptype colapsa a una sola entrada; intensity se acumula pero
        el dict ``_deposits[coord]`` no crece sin cota.
        """
        trail = PheromoneTrail()
        coord = HexCoord(3, -1)
        for _ in range(10_000):
            trail.deposit(coord, PheromoneType.TRAIL, 0.0001)
        # Solo 1 entrada (un ptype) pese a 10K depósitos.
        assert len(trail._deposits) == 1
        assert len(trail._deposits[coord]) == 1

    def test_flood_distinct_coords_bounded_by_max_coords(self):
        """
        10K deposits en coords DISTINTAS → evicción LRU mantiene len <= max_coords.
        """
        trail = PheromoneTrail(max_coords=256)
        for i in range(10_000):
            trail.deposit(HexCoord(i, -i), PheromoneType.TRAIL, 0.1)
        assert len(trail._deposits) <= 256

    def test_metadata_flood_bounded(self):
        """
        Metadata con miles de keys → acotada a max_metadata_keys.
        """
        trail = PheromoneTrail(max_metadata_keys=10)
        for i in range(1000):
            trail.deposit(
                HexCoord.origin(),
                PheromoneType.TRAIL,
                0.01,
                metadata={f"k{i}": i},
            )
        deposit = trail._deposits[HexCoord.origin()][PheromoneType.TRAIL]
        assert len(deposit.metadata) <= 10

    def test_deposit_auto_signed(self):
        """Cada PheromoneDeposit creado nuevo está firmado."""
        trail = PheromoneTrail()
        trail.deposit(HexCoord(0, 1), PheromoneType.FOOD, 0.5)
        deposit = trail._deposits[HexCoord(0, 1)][PheromoneType.FOOD]
        assert deposit.signature is not None
        assert deposit.verify()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Path traversal en HoneyArchive
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoneyArchivePathTraversal:
    """
    Fase 2.4 — ``HoneyArchive.archive`` rechaza claves que apuntan fuera
    del base_path. Aunque el checkpoint actual es in-memory, validar aquí
    cierra el vector antes de que la persistencia a disco se active.
    """

    def test_traversal_dotdot_rejected(self):
        archive = HoneyArchive(MemoryConfig(), base_path="/tmp/hoc-test")
        assert not archive.archive("../../etc/passwd", {"secret": "pwned"})

    def test_traversal_absolute_rejected(self, tmp_path):
        archive = HoneyArchive(MemoryConfig(), base_path=str(tmp_path))
        # Ruta absoluta explícita → rechazada
        # (en Windows /etc se resuelve, en POSIX también; ambos son absolute)
        assert not archive.archive("/etc/passwd", {"x": 1})

    def test_traversal_null_byte_rejected(self, tmp_path):
        archive = HoneyArchive(MemoryConfig(), base_path=str(tmp_path))
        assert not archive.archive("ok\x00../../etc", {"x": 1})

    def test_valid_key_accepted(self, tmp_path):
        archive = HoneyArchive(MemoryConfig(), base_path=str(tmp_path))
        assert archive.archive("legit_key_001", {"x": 1})
        assert archive.retrieve("legit_key_001") == {"x": 1}

    def test_safe_join_primitive(self, tmp_path):
        """safe_join devuelve path confinado o lanza PathTraversalError."""
        root = tmp_path
        ok = safe_join(root, "sub/dir/file")
        assert root in ok.parents or ok.parent == root or ok == root / "sub" / "dir" / "file"

        with pytest.raises(PathTraversalError):
            safe_join(root, "../../escape")


# ═══════════════════════════════════════════════════════════════════════════════
# Primitivas transversales: HMAC, CSPRNG, rate limiting
# ═══════════════════════════════════════════════════════════════════════════════


class TestHmacPrimitives:
    def test_sign_verify_round_trip(self):
        payload = b"hello world"
        tag = sign_payload(payload)
        assert verify_signature(payload, tag)

    def test_verify_wrong_payload(self):
        tag = sign_payload(b"original")
        assert not verify_signature(b"modified", tag)

    def test_verify_wrong_tag_length(self):
        assert not verify_signature(b"x", b"short")

    def test_serialize_round_trip(self):
        for val in ({"a": 1}, [1, 2, 3], "string", 42, None, True):
            assert deserialize(serialize(val)) == val

    def test_unsigned_serialize_doesnt_verify(self):
        raw = serialize({"x": 1}, sign=False)
        # sin firma, verify=True debe rechazar
        with pytest.raises(MSCSecurityError):
            deserialize(raw, verify=True)
        # verify=False acepta
        assert deserialize(raw, verify=False) == {"x": 1}


class TestCsprng:
    def test_secure_random_in_unit_interval(self):
        for _ in range(100):
            v = secure_random()
            assert 0.0 <= v < 1.0

    def test_secure_choice_returns_element(self):
        seq = [1, 2, 3, 4, 5]
        for _ in range(20):
            assert secure_choice(seq) in seq


class TestRateLimiter:
    def test_basic_allow_and_deny(self):
        limiter = RateLimiter(per_second=10, burst=1)
        assert limiter.try_acquire()
        # El siguiente intento inmediato excede burst=1
        assert not limiter.try_acquire()

    def test_decorator_raises(self):
        @rate_limit(per_second=1, burst=1)
        def op():
            return 42

        assert op() == 42
        with pytest.raises(RateLimitExceeded):
            op()

    def test_refills_after_time(self):
        import time

        limiter = RateLimiter(per_second=100, burst=1)
        assert limiter.try_acquire()
        assert not limiter.try_acquire()
        time.sleep(0.05)  # ~5 tokens added at 100/s
        assert limiter.try_acquire()

    def test_submit_task_rate_limited(self, tmp_grid):
        """SwarmScheduler.submit_task respeta su limitador."""
        from hoc.nectar import NectarFlow
        from hoc.swarm import SwarmConfig, SwarmScheduler

        nectar = NectarFlow(tmp_grid)
        cfg = SwarmConfig(submit_rate_per_second=1, submit_rate_burst=1)
        sched = SwarmScheduler(tmp_grid, nectar, cfg)
        sched.submit_task("test", {})  # OK
        with pytest.raises(RateLimitExceeded):
            sched.submit_task("test", {})


# ═══════════════════════════════════════════════════════════════════════════════
# DanceMessage signing
# ═══════════════════════════════════════════════════════════════════════════════


class TestDanceSigning:
    def test_dance_auto_signed_by_nectar(self, tmp_grid):
        nectar = NectarFlow(tmp_grid)
        dance = nectar.start_dance(
            dancer=HexCoord.origin(),
            direction=DanceDirection.UP,
            distance=3,
            quality=0.8,
        )
        assert dance.signature is not None
        assert dance.verify()

    def test_dance_propagate_preserves_signature(self, tmp_grid):
        """Propagación cambia quality/ttl pero la firma original sigue válida."""
        nectar = NectarFlow(tmp_grid)
        nectar.start_dance(
            dancer=HexCoord.origin(),
            direction=DanceDirection.RIGHT,
            distance=2,
            quality=0.9,
        )
        nectar._dance.propagate(tmp_grid)
        # Hay danzas propagadas en vecinos del origen
        observed = nectar._dance.observe_dances(HexCoord.origin(), radius=2)
        # Al menos una propagada debería verify() OK
        assert any(d.verify() for d in observed)

    def test_tampered_dance_fields_break_signature(self):
        dance = DanceMessage(
            source=HexCoord.origin(),
            direction=DanceDirection.UP,
            distance=5,
            quality=1.0,
            resource_type="food",
        )
        dance.sign()
        assert dance.verify()
        # Modificar un campo de identidad rompe la firma
        dance.distance = 999
        assert not dance.verify()


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: HiveMemory end-to-end con HMAC
# ═══════════════════════════════════════════════════════════════════════════════


class TestHiveMemoryIntegration:
    def test_put_get_round_trip(self, tmp_grid):
        mem = HiveMemory(tmp_grid)
        mem.put("user:1", {"name": "alice", "age": 30})
        assert mem.get("user:1") == {"name": "alice", "age": 30}

    def test_archive_retrieve(self, tmp_grid, tmp_path):
        cfg = MemoryConfig()
        mem = HiveMemory(tmp_grid, cfg)
        mem.put("k", {"v": 1}, archive=True)
        # Override archive path to something under tmp for safety
        mem._honey.base_path = tmp_path.resolve()
        mem.archive("k")
        assert mem.get("k", include_archive=True) == {"v": 1}
