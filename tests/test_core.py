"""Tests para el módulo core de HOC (geometría hexagonal y grid)."""
import pytest

from hoc.core import (
    HexCoord,
    HexDirection,
    HoneycombGrid,
    HoneycombConfig,
    HoneycombCell,
    QueenCell,
    WorkerCell,
    CellState,
    CellRole,
    HexRing,
)


class TestHexCoord:
    """Tests para HexCoord."""

    def test_origin(self):
        """HexCoord.origin() retorna (0, 0)."""
        origin = HexCoord.origin()
        assert origin.q == 0
        assert origin.r == 0

    def test_neighbor_directions(self):
        """Cada dirección produce un vecino distinto."""
        coord = HexCoord(0, 0)
        neighbors = {coord.neighbor(d) for d in HexDirection}
        assert len(neighbors) == 6

    def test_neighbor_nw(self):
        """Vecino NW de (0,0) es (-1, 0) o similar según sistema axial."""
        coord = HexCoord(0, 0)
        nw = coord.neighbor(HexDirection.NW)
        assert nw != coord

    def test_distance_to_self(self):
        """Distancia de un coord a sí mismo es 0."""
        coord = HexCoord(3, -2)
        assert coord.distance_to(coord) == 0

    def test_distance_simetrica(self):
        """Distancia A->B == B->A."""
        a = HexCoord(1, 2)
        b = HexCoord(-3, 1)
        assert a.distance_to(b) == b.distance_to(a)

    def test_equality(self):
        """Coords con mismos q,r son iguales."""
        assert HexCoord(2, 3) == HexCoord(2, 3)
        assert HexCoord(2, 3) != HexCoord(2, 4)

    def test_hashable(self):
        """HexCoord es hashable (puede ir en sets/dicts)."""
        coords = {HexCoord(0, 0), HexCoord(1, 0), HexCoord(0, 0)}
        assert len(coords) == 2


class TestHexDirection:
    """Tests para HexDirection."""

    def test_opposite(self):
        """Cada dirección tiene una opuesta."""
        for d in HexDirection:
            opp = d.opposite()
            assert opp != d
            assert opp.opposite() == d

    def test_six_directions(self):
        """Hay exactamente 6 direcciones."""
        assert len(HexDirection) == 6


class TestHoneycombGrid:
    """Tests para HoneycombGrid."""

    def test_grid_creacion_default(self):
        """Grid se crea con configuración por defecto."""
        grid = HoneycombGrid()
        assert grid.cell_count > 0

    def test_grid_con_config(self):
        """Grid se crea con config personalizada."""
        config = HoneycombConfig(radius=2, vcores_per_cell=2)
        grid = HoneycombGrid(config)
        assert grid.config.radius == 2

    def test_queen_exists(self):
        """El grid tiene una celda reina."""
        grid = HoneycombGrid()
        assert grid.queen is not None
        assert isinstance(grid.queen, QueenCell)

    def test_get_cell(self):
        """get_cell retorna celda existente o None."""
        grid = HoneycombGrid()
        origin = HexCoord.origin()
        cell = grid.get_cell(origin)
        assert cell is not None
        assert cell.coord == origin

    def test_get_cell_nonexistent(self):
        """get_cell retorna None para coord fuera del grid."""
        grid = HoneycombGrid()
        far = HexCoord(1000, 1000)
        assert grid.get_cell(far) is None

    def test_tick(self):
        """El grid puede ejecutar un tick sin fallar."""
        grid = HoneycombGrid()
        grid.tick()
        assert grid.cell_count > 0


class TestHexRing:
    """Tests para HexRing (alias de HexRegion)."""

    def test_ring_radius_0(self):
        """Ring de radio 0 contiene solo el centro."""
        origin = HexCoord.origin()
        ring = list(HexRing.from_ring(origin, 0))
        assert len(ring) == 1
        assert origin in ring

    def test_ring_radius_1(self):
        """Ring de radio 1 tiene 6 celdas."""
        ring = list(HexRing.from_ring(HexCoord.origin(), 1))
        assert len(ring) == 6
