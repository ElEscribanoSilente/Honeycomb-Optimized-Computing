"""Tests para el módulo nectar de HOC (feromonas, Waggle Dance, Royal Jelly)."""
import pytest

from hoc.core import HexCoord, HexDirection, HoneycombGrid
from hoc.nectar import (
    PheromoneTrail,
    PheromoneType,
    PheromoneDecay,
    PheromoneDeposit,
    WaggleDance,
    DanceMessage,
    DanceDirection,
    NectarFlow,
    RoyalJelly,
    RoyalCommand,
)


class TestPheromoneTrail:
    """Tests para PheromoneTrail."""

    def test_deposit_and_sense(self):
        """Depositar y sensar feromona."""
        trail = PheromoneTrail()
        coord = HexCoord(0, 0)
        trail.deposit(coord, PheromoneType.FOOD, 1.0)
        level = trail.sense(coord, PheromoneType.FOOD)
        assert level == 1.0

    def test_sense_empty(self):
        """Sensar coord vacía retorna 0."""
        trail = PheromoneTrail()
        level = trail.sense(HexCoord(0, 0), PheromoneType.FOOD)
        assert level == 0.0

    def test_deposit_accumulates(self):
        """Depósitos múltiples acumulan intensidad."""
        trail = PheromoneTrail()
        coord = HexCoord(0, 0)
        trail.deposit(coord, PheromoneType.TRAIL, 0.5)
        trail.deposit(coord, PheromoneType.TRAIL, 0.3)
        level = trail.sense(coord, PheromoneType.TRAIL)
        assert level == 0.8

    def test_max_intensity_cap(self):
        """La intensidad no excede max_intensity."""
        trail = PheromoneTrail(max_intensity=2.0)
        coord = HexCoord(0, 0)
        trail.deposit(coord, PheromoneType.FOOD, 5.0)
        level = trail.sense(coord, PheromoneType.FOOD)
        assert level <= 2.0

    def test_clear(self):
        """clear elimina feromonas."""
        trail = PheromoneTrail()
        coord = HexCoord(0, 0)
        trail.deposit(coord, PheromoneType.FOOD, 1.0)
        trail.clear(coord, PheromoneType.FOOD)
        assert trail.sense(coord, PheromoneType.FOOD) == 0.0

    def test_get_stats(self):
        """get_stats retorna estadísticas válidas."""
        trail = PheromoneTrail()
        trail.deposit(HexCoord(0, 0), PheromoneType.FOOD, 1.0)
        stats = trail.get_stats()
        assert "locations" in stats
        assert stats["locations"] >= 0


class TestDanceMessage:
    """Tests para DanceMessage."""

    def test_encode_decode(self):
        """Codificación y decodificación round-trip."""
        source = HexCoord(1, 2)
        msg = DanceMessage(
            source=source,
            direction=DanceDirection.RIGHT,
            distance=5,
            quality=0.9,
            resource_type="food",
        )
        encoded = msg.encode()
        decoded = DanceMessage.decode(encoded, source)
        assert decoded.direction == msg.direction
        assert decoded.distance == msg.distance
        assert decoded.quality == msg.quality
        assert decoded.resource_type == msg.resource_type


class TestWaggleDance:
    """Tests para WaggleDance."""

    def test_start_dance(self):
        """Iniciar una danza crea mensaje."""
        dance = WaggleDance()
        coord = HexCoord(0, 0)
        msg = dance.start_dance(
            dancer=coord,
            direction=DanceDirection.UP,
            distance=3,
            quality=0.8,
            resource_type="work",
        )
        assert msg.source == coord
        assert msg.resource_type == "work"

    def test_get_stats(self):
        """get_stats retorna estadísticas."""
        dance = WaggleDance()
        dance.start_dance(
            HexCoord(0, 0),
            DanceDirection.UP,
            2,
            0.5,
            "generic",
        )
        stats = dance.get_stats()
        assert "total_dances" in stats or "active_locations" in stats


class TestNectarFlow:
    """Tests para NectarFlow."""

    @pytest.fixture
    def grid(self):
        """Grid pequeño para tests."""
        from hoc.core import HoneycombConfig
        config = HoneycombConfig(radius=1)
        return HoneycombGrid(config)

    def test_nectar_flow_creation(self, grid):
        """NectarFlow se crea con un grid."""
        nectar = NectarFlow(grid)
        assert nectar.grid is grid

    def test_deposit_and_sense(self, grid):
        """Depositar y sensar feromona a través de NectarFlow."""
        nectar = NectarFlow(grid)
        coord = grid.queen.coord if grid.queen else HexCoord.origin()
        nectar.deposit_pheromone(coord, PheromoneType.FOOD, 0.8)
        level = nectar.sense_pheromone(coord, PheromoneType.FOOD)
        assert level >= 0.8

    def test_tick(self, grid):
        """tick ejecuta sin fallar."""
        nectar = NectarFlow(grid)
        results = nectar.tick()
        assert "pheromones_evaporated" in results or "dances_propagated" in results


class TestRoyalJelly:
    """Tests para RoyalJelly."""

    def test_issue_command(self):
        """Emitir comando real."""
        jelly = RoyalJelly(HexCoord.origin())
        msg = jelly.issue_command(RoyalCommand.BALANCE, priority=5)
        assert msg.command == RoyalCommand.BALANCE
        assert msg.priority == 5

    def test_emergency_broadcast(self):
        """Emergency broadcast emite comando de máxima prioridad."""
        jelly = RoyalJelly(HexCoord.origin())
        jelly.emergency_broadcast("test emergency")
        assert jelly.get_pending_count() >= 1
