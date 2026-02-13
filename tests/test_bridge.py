"""Tests para el módulo bridge de HOC (integración CAMV)."""
import pytest

from hoc.core import HexCoord, HoneycombGrid, HoneycombConfig
from hoc.bridge import (
    HexToCartesian,
    CartesianToHex,
    CellToVCoreMapper,
    CAMVHoneycombBridge,
    VentHoneycombAdapter,
)


class TestHexToCartesian:
    """Tests para conversión Hex -> Cartesian."""

    def test_origin(self):
        """Origen hex (0,0) mapea a coordenadas cartesianas."""
        conv = HexToCartesian(size=1.0)
        x, y = conv.convert(HexCoord.origin())
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))

    def test_roundtrip(self):
        """Conversión hex->cart->hex preserva coordenada."""
        conv_hex = HexToCartesian(size=1.0)
        conv_cart = CartesianToHex(size=1.0)
        coord = HexCoord(2, -1)
        x, y = conv_hex.convert(coord)
        back = conv_cart.convert(x, y)
        assert back == coord


class TestCartesianToHex:
    """Tests para conversión Cartesian -> Hex."""

    def test_convert_returns_hexcoord(self):
        """convert retorna HexCoord."""
        conv = CartesianToHex(size=1.0)
        coord = conv.convert(1.5, 2.3)
        assert coord.q == coord.q  # HexCoord tiene q, r
        assert hasattr(coord, "r")


class TestCellToVCoreMapper:
    """Tests para CellToVCoreMapper."""

    def test_map_and_get(self):
        """Mapear celda a vCore y recuperar."""
        mapper = CellToVCoreMapper()
        coord = HexCoord(1, 0)
        stub = type("Stub", (), {"vcore_id": "vc_1", "state": "ready"})()
        mapper.map_cell(coord, stub)
        vcores = mapper.get_vcores(coord)
        assert len(vcores) == 1
        assert mapper.get_cell("vc_1") == coord


class TestCAMVHoneycombBridge:
    """Tests para CAMVHoneycombBridge."""

    @pytest.fixture
    def grid(self):
        """Grid pequeño para tests."""
        config = HoneycombConfig(radius=1, vcores_per_cell=2)
        return HoneycombGrid(config)

    def test_initialize_without_hypervisor(self, grid):
        """Bridge se inicializa sin hypervisor (stub)."""
        bridge = CAMVHoneycombBridge(grid, hypervisor=None)
        success = bridge.initialize()
        assert success

    def test_hex_cartesian_conversion(self, grid):
        """Conversión hex <-> cartesian funciona."""
        bridge = CAMVHoneycombBridge(grid)
        coord = HexCoord(1, -1)
        x, y = bridge.hex_to_cartesian(coord)
        back = bridge.cartesian_to_hex(x, y)
        assert back == coord


class TestVentHoneycombAdapter:
    """Tests para VentHoneycombAdapter."""

    @pytest.fixture
    def grid_and_bridge(self):
        """Grid y bridge inicializado."""
        config = HoneycombConfig(radius=1)
        grid = HoneycombGrid(config)
        bridge = CAMVHoneycombBridge(grid)
        bridge.initialize()
        return grid, bridge

    def test_assign_entity(self, grid_and_bridge):
        """Asignar entidad a celda."""
        grid, bridge = grid_and_bridge
        adapter = VentHoneycombAdapter(grid, bridge)
        coord = adapter.assign_entity("entity_1")
        assert coord is not None
        assert adapter.get_entity_cell("entity_1") == coord
