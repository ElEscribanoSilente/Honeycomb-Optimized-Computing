"""Benchmarks para el módulo core de HOC."""
import sys
from pathlib import Path

# Añadir raíz al path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))


def test_hexcoord_creation(benchmark):
    """Benchmark: creación de HexCoord."""
    from hoc.core import HexCoord
    benchmark(HexCoord, 5, -3)


def test_hexcoord_neighbor(benchmark):
    """Benchmark: obtener vecino."""
    from hoc.core import HexCoord, HexDirection
    coord = HexCoord(0, 0)
    benchmark(lambda: coord.neighbor(HexDirection.E))


def test_hexcoord_distance(benchmark):
    """Benchmark: calcular distancia entre coords."""
    from hoc.core import HexCoord
    a = HexCoord(10, -5)
    b = HexCoord(-3, 8)
    benchmark(a.distance_to, b)


def test_grid_creation(benchmark):
    """Benchmark: creación de grid pequeño."""
    from hoc.core import HoneycombGrid, HoneycombConfig
    config = HoneycombConfig(radius=2)
    benchmark(HoneycombGrid, config)


def test_grid_tick(benchmark):
    """Benchmark: tick del grid."""
    from hoc.core import HoneycombGrid, HoneycombConfig
    config = HoneycombConfig(radius=2)
    grid = HoneycombGrid(config)
    benchmark(grid.tick)


def test_ring_iteration(benchmark):
    """Benchmark: iterar sobre anillo de radio 5."""
    from hoc.core import HexCoord
    origin = HexCoord.origin()
    benchmark(lambda: list(origin.ring(5)))
