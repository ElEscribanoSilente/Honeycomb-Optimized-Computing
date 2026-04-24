"""
Property-based tests con Hypothesis para HexCoord y PheromoneField.

Verifica invariantes matemáticas y propiedades de la geometría hexagonal y
del modelo estigmérgico de feromonas.
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, assume, given, settings, strategies as st

from core import (
    HexCoord,
    HexDirection,
    PheromoneDeposit,
    PheromoneField,
    PheromoneType,
)

# ─────────────────────────────────────────────────────────────────────────────
# Estrategias
# ─────────────────────────────────────────────────────────────────────────────

# Coordenadas razonables para evitar overflow
hex_int = st.integers(min_value=-1000, max_value=1000)
small_hex_int = st.integers(min_value=-50, max_value=50)


@st.composite
def hex_coords(draw, q_strategy=hex_int, r_strategy=hex_int):
    return HexCoord(draw(q_strategy), draw(r_strategy))


@st.composite
def small_hex_coords(draw):
    return HexCoord(draw(small_hex_int), draw(small_hex_int))


pheromone_types = st.sampled_from(list(PheromoneType))
intensities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
positive_intensities = st.floats(
    min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False
)
decay_rates = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)


# ─────────────────────────────────────────────────────────────────────────────
# HexCoord — invariantes geométricas
# ─────────────────────────────────────────────────────────────────────────────


class TestHexCoordCubeInvariant:
    """En coordenadas cúbicas hexagonales: q + r + s ≡ 0."""

    @given(q=hex_int, r=hex_int)
    def test_cube_sum_is_zero(self, q, r):
        c = HexCoord(q, r)
        assert c.q + c.r + c.s == 0

    @given(coord=hex_coords())
    def test_cube_property_consistent(self, coord):
        assert coord.cube == (coord.q, coord.r, coord.s)


class TestHexCoordDistance:
    """Propiedades métricas de la distancia hexagonal."""

    @given(a=hex_coords(), b=hex_coords())
    def test_distance_symmetric(self, a, b):
        assert a.distance_to(b) == b.distance_to(a)

    @given(a=hex_coords())
    def test_distance_to_self_is_zero(self, a):
        assert a.distance_to(a) == 0

    @given(a=hex_coords(), b=hex_coords())
    def test_distance_non_negative(self, a, b):
        assert a.distance_to(b) >= 0

    @given(a=small_hex_coords(), b=small_hex_coords(), c=small_hex_coords())
    def test_triangle_inequality(self, a, b, c):
        assert a.distance_to(c) <= a.distance_to(b) + b.distance_to(c)

    @given(coord=hex_coords())
    def test_magnitude_equals_distance_from_origin(self, coord):
        origin = HexCoord(0, 0)
        assert coord.magnitude == coord.distance_to(origin)

    @given(a=hex_coords(), b=hex_coords())
    def test_distance_zero_iff_equal(self, a, b):
        if a == b:
            assert a.distance_to(b) == 0
        else:
            assert a.distance_to(b) > 0


class TestHexCoordArithmetic:
    """Aritmética de coordenadas hexagonales."""

    @given(a=hex_coords(), b=hex_coords())
    def test_addition_commutative(self, a, b):
        assert a + b == b + a

    @given(a=small_hex_coords(), b=small_hex_coords(), c=small_hex_coords())
    def test_addition_associative(self, a, b, c):
        assert (a + b) + c == a + (b + c)

    @given(a=hex_coords())
    def test_addition_identity(self, a):
        zero = HexCoord(0, 0)
        assert a + zero == a
        assert zero + a == a

    @given(a=hex_coords(), b=hex_coords())
    def test_subtraction_inverse_of_addition(self, a, b):
        assert (a + b) - b == a

    @given(a=hex_coords())
    def test_negation_involution(self, a):
        assert -(-a) == a

    @given(a=hex_coords())
    def test_negation_addition_is_zero(self, a):
        zero = HexCoord(0, 0)
        assert a + (-a) == zero

    @given(a=hex_coords())
    def test_multiply_by_zero(self, a):
        zero = HexCoord(0, 0)
        assert a * 0 == zero

    @given(a=hex_coords())
    def test_multiply_by_one(self, a):
        assert a * 1 == a

    @given(a=small_hex_coords(), n=st.integers(min_value=-10, max_value=10))
    def test_scalar_distributes_over_addition(self, a, n):
        # n * (a + a) == n*a + n*a
        sum_first = (a + a) * n
        mul_first = (a * n) + (a * n)
        assert sum_first == mul_first

    @given(a=small_hex_coords(), n=st.integers(min_value=-10, max_value=10))
    def test_scalar_left_right_equal(self, a, n):
        assert n * a == a * n

    @given(a=hex_coords())
    def test_abs_equals_magnitude(self, a):
        assert abs(a) == a.magnitude


class TestHexCoordNeighbors:
    """Propiedades de la topología vecinal."""

    @given(coord=hex_coords())
    def test_all_neighbors_at_distance_one(self, coord):
        for n in coord.neighbors():
            assert coord.distance_to(n) == 1

    @given(coord=hex_coords())
    def test_six_distinct_neighbors(self, coord):
        ns = coord.neighbors()
        assert len(ns) == 6
        assert len(set(ns)) == 6

    @given(coord=hex_coords())
    def test_neighbors_are_symmetric(self, coord):
        # Si n es vecino de coord, coord es vecino de n
        for n in coord.neighbors():
            assert coord in n.neighbors()

    @given(coord=hex_coords(), direction=st.sampled_from(list(HexDirection)))
    def test_direction_to_neighbor_recovers_direction(self, coord, direction):
        n = coord.neighbor(direction)
        assert coord.direction_to(n) == direction

    @given(a=hex_coords(), b=hex_coords())
    def test_direction_to_non_neighbor_is_none(self, a, b):
        assume(a.distance_to(b) != 1)
        assert a.direction_to(b) is None


class TestHexCoordRing:
    """Propiedades de los anillos hexagonales."""

    @given(center=small_hex_coords(), radius=st.integers(min_value=1, max_value=10))
    def test_ring_size_is_six_times_radius(self, center, radius):
        ring = center.ring(radius)
        assert len(ring) == 6 * radius

    @given(center=small_hex_coords(), radius=st.integers(min_value=1, max_value=8))
    def test_all_ring_cells_at_correct_distance(self, center, radius):
        for cell in center.ring(radius):
            assert center.distance_to(cell) == radius

    @given(center=small_hex_coords())
    def test_ring_zero_is_center_only(self, center):
        ring = center.ring(0)
        assert ring == (center,)

    @given(center=small_hex_coords(), radius=st.integers(min_value=0, max_value=5))
    def test_filled_hexagon_size(self, center, radius):
        # 1 + 6 + 12 + ... + 6r = 1 + 3r(r+1)
        cells = center.filled_hexagon(radius)
        expected = 1 + 3 * radius * (radius + 1)
        assert len(cells) == expected

    @given(center=small_hex_coords(), radius=st.integers(min_value=0, max_value=5))
    def test_filled_hexagon_all_within_radius(self, center, radius):
        for cell in center.filled_hexagon(radius):
            assert center.distance_to(cell) <= radius


class TestHexCoordLine:
    """Propiedades de líneas e interpolación."""

    @given(a=small_hex_coords(), b=small_hex_coords())
    def test_line_starts_with_a_ends_with_b(self, a, b):
        line = a.line_to(b)
        assert line[0] == a
        assert line[-1] == b

    @given(a=small_hex_coords(), b=small_hex_coords())
    def test_line_length_equals_distance_plus_one(self, a, b):
        line = a.line_to(b)
        assert len(line) == a.distance_to(b) + 1

    @given(a=small_hex_coords(), b=small_hex_coords())
    def test_lerp_zero_returns_a(self, a, b):
        assert a.lerp(b, 0.0) == a

    @given(a=small_hex_coords(), b=small_hex_coords())
    def test_lerp_one_returns_b(self, a, b):
        assert a.lerp(b, 1.0) == b


class TestHexCoordHashEquality:
    """Propiedades de hash y igualdad para uso como clave de diccionario."""

    @given(q=hex_int, r=hex_int)
    def test_equal_coords_have_equal_hashes(self, q, r):
        a = HexCoord(q, r)
        b = HexCoord(q, r)
        assert a == b
        assert hash(a) == hash(b)

    @given(q=hex_int, r=hex_int)
    def test_can_use_as_dict_key(self, q, r):
        d = {HexCoord(q, r): "value"}
        assert d[HexCoord(q, r)] == "value"

    @given(coords=st.lists(hex_coords(), min_size=1, max_size=20))
    def test_set_dedupes_equal_coords(self, coords):
        unique = set(coords)
        # Convertir a tuplas para verificar
        unique_tuples = set((c.q, c.r) for c in coords)
        assert len(unique) == len(unique_tuples)


class TestHexCoordRotation:
    """Rotación alrededor de un centro (60° por paso)."""

    @given(coord=small_hex_coords(), center=small_hex_coords())
    def test_rotate_six_steps_returns_to_origin(self, coord, center):
        # 6 rotaciones de 60° = 360° = identidad
        rotated = coord.rotate_around(center, steps=6)
        assert rotated == coord

    @given(coord=small_hex_coords(), center=small_hex_coords())
    def test_rotate_preserves_distance_from_center(self, coord, center):
        original_d = center.distance_to(coord)
        for steps in range(1, 7):
            rotated = coord.rotate_around(center, steps=steps)
            assert center.distance_to(rotated) == original_d


# ─────────────────────────────────────────────────────────────────────────────
# PheromoneDeposit & PheromoneField
# ─────────────────────────────────────────────────────────────────────────────


class TestPheromoneDepositDecay:

    @given(
        intensity=positive_intensities,
        decay_rate=decay_rates,
        elapsed=st.floats(min_value=0.0, max_value=10.0),
    )
    def test_decay_never_increases_intensity(self, intensity, decay_rate, elapsed):
        d = PheromoneDeposit(
            ptype=PheromoneType.FOOD,
            intensity=intensity,
            decay_rate=decay_rate,
        )
        before = d.intensity
        d.decay(elapsed)
        assert d.intensity <= before + 1e-9

    @given(intensity=positive_intensities, decay_rate=decay_rates)
    def test_decay_zero_elapsed_is_identity(self, intensity, decay_rate):
        d = PheromoneDeposit(
            ptype=PheromoneType.FOOD,
            intensity=intensity,
            decay_rate=decay_rate,
        )
        before = d.intensity
        d.decay(0.0)
        assert math.isclose(d.intensity, before, rel_tol=1e-9)

    @given(intensity=positive_intensities, decay_rate=decay_rates)
    def test_decay_eventually_inactive(self, intensity, decay_rate):
        d = PheromoneDeposit(
            ptype=PheromoneType.FOOD,
            intensity=intensity,
            decay_rate=decay_rate,
        )
        # Aplicar suficientes ciclos para llegar bajo el umbral
        for _ in range(2000):
            d.decay(1.0)
            if not d.is_active:
                break
        assert not d.is_active

    @given(intensity=positive_intensities)
    def test_is_active_threshold(self, intensity):
        d = PheromoneDeposit(ptype=PheromoneType.FOOD, intensity=intensity)
        if intensity > PheromoneDeposit.ACTIVE_THRESHOLD:
            assert d.is_active
        else:
            assert not d.is_active


class TestPheromoneFieldDeposit:

    @given(ptype=pheromone_types, amount=intensities)
    def test_deposit_intensity_clamped_to_one(self, ptype, amount):
        f = PheromoneField()
        f.deposit(ptype, amount)
        assert 0.0 <= f.get_intensity(ptype) <= 1.0

    @given(ptype=pheromone_types, amounts=st.lists(intensities, min_size=1, max_size=10))
    def test_repeated_deposit_never_exceeds_one(self, ptype, amounts):
        f = PheromoneField()
        for a in amounts:
            f.deposit(ptype, a)
        assert f.get_intensity(ptype) <= 1.0 + 1e-9

    @given(ptype=pheromone_types, amounts=st.lists(positive_intensities, min_size=2, max_size=10))
    def test_deposit_monotonic_below_cap(self, ptype, amounts):
        f = PheromoneField()
        prev = 0.0
        for a in amounts:
            f.deposit(ptype, a)
            current = f.get_intensity(ptype)
            # Solo monotónico mientras no choque con el cap
            if prev < 1.0:
                assert current >= prev - 1e-9
            prev = current

    @given(
        deposits=st.lists(st.tuples(pheromone_types, positive_intensities), min_size=1, max_size=10)
    )
    def test_total_intensity_equals_sum(self, deposits):
        f = PheromoneField()
        for ptype, amount in deposits:
            f.deposit(ptype, amount)
        expected = sum(f.get_intensity(p) for p in PheromoneType)
        assert math.isclose(f.total_intensity, expected, rel_tol=1e-9, abs_tol=1e-9)

    @given(unused_ptype=pheromone_types)
    def test_get_intensity_zero_for_undeposited(self, unused_ptype):
        f = PheromoneField()
        assert f.get_intensity(unused_ptype) == 0.0

    @given(
        deposits=st.lists(st.tuples(pheromone_types, positive_intensities), min_size=1, max_size=10)
    )
    def test_dominant_type_has_max_intensity(self, deposits):
        f = PheromoneField()
        for ptype, amount in deposits:
            f.deposit(ptype, amount)
        dom = f.dominant_type
        if dom is not None:
            dom_intensity = f.get_intensity(dom)
            for p in PheromoneType:
                assert f.get_intensity(p) <= dom_intensity + 1e-9

    def test_dominant_type_is_none_when_empty(self):
        f = PheromoneField()
        assert f.dominant_type is None


class TestPheromoneFieldDecay:

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(
        deposits=st.lists(st.tuples(pheromone_types, positive_intensities), min_size=1, max_size=5)
    )
    def test_decay_all_never_increases_total(self, deposits):
        f = PheromoneField()
        for ptype, amount in deposits:
            f.deposit(ptype, amount)
        before = f.total_intensity
        f.decay_all(1.0)
        assert f.total_intensity <= before + 1e-9

    @given(
        deposits=st.lists(st.tuples(pheromone_types, positive_intensities), min_size=1, max_size=5)
    )
    def test_aggressive_decay_drains_field(self, deposits):
        f = PheromoneField()
        for ptype, amount in deposits:
            # decay rápido para que se limpien rápido
            f.deposit(ptype, amount, decay_rate=0.9)
        for _ in range(100):
            f.decay_all(1.0)
        # Tras mucho decay, total cerca de 0
        assert f.total_intensity < PheromoneDeposit.ACTIVE_THRESHOLD * len(PheromoneType)


class TestPheromoneFieldGradient:

    @given(
        deposits=st.lists(st.tuples(pheromone_types, positive_intensities), min_size=1, max_size=10)
    )
    def test_gradient_keys_match_active_deposits(self, deposits):
        f = PheromoneField()
        for ptype, amount in deposits:
            f.deposit(ptype, amount)
        gradient = f.get_gradient_vector()
        # Todos los ptypes con intensity > 0 deben aparecer
        for p in PheromoneType:
            if f.get_intensity(p) > 0:
                assert p in gradient
                assert math.isclose(gradient[p], f.get_intensity(p), rel_tol=1e-9)

    @given(
        deposits=st.lists(st.tuples(pheromone_types, positive_intensities), min_size=1, max_size=5)
    )
    def test_to_dict_keys_are_type_names(self, deposits):
        f = PheromoneField()
        for ptype, amount in deposits:
            f.deposit(ptype, amount)
        d = f.to_dict()
        for key in d:
            assert any(p.name == key for p in PheromoneType)
