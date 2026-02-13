"""Benchmarks para el módulo nectar de HOC."""
import sys
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))


def test_pheromone_deposit(benchmark):
    """Benchmark: depositar feromona."""
    from hoc.core import HexCoord
    from hoc.nectar import PheromoneTrail, PheromoneType
    trail = PheromoneTrail()
    coord = HexCoord(0, 0)

    def do_deposit():
        trail.deposit(coord, PheromoneType.FOOD, 0.5)

    benchmark(do_deposit)


def test_pheromone_sense(benchmark):
    """Benchmark: sensar feromona."""
    from hoc.core import HexCoord
    from hoc.nectar import PheromoneTrail, PheromoneType
    trail = PheromoneTrail()
    trail.deposit(HexCoord(0, 0), PheromoneType.FOOD, 1.0)
    coord = HexCoord(0, 0)

    def do_sense():
        trail.sense(coord, PheromoneType.FOOD)

    benchmark(do_sense)


def test_nectar_flow_tick(benchmark):
    """Benchmark: tick completo de NectarFlow."""
    from hoc.core import HoneycombGrid, HoneycombConfig
    from hoc.nectar import NectarFlow
    config = HoneycombConfig(radius=2)
    grid = HoneycombGrid(config)
    nectar = NectarFlow(grid)

    benchmark(nectar.tick)


def test_dance_start(benchmark):
    """Benchmark: iniciar danza waggle."""
    from hoc.core import HexCoord
    from hoc.nectar import WaggleDance, DanceDirection
    dance = WaggleDance()
    coord = HexCoord(0, 0)

    def do_dance():
        dance.start_dance(coord, DanceDirection.UP, 3, 0.8, "work")

    benchmark(do_dance)
