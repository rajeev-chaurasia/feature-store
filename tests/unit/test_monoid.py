"""The monoid laws, and the one place they do not hold exactly.

Every served value is a fold over partial states, so if the algebra is wrong the whole
store is wrong in a way that looks like noise rather than like a bug.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from asofline.agg import AVG, COUNT, MAX, MIN, MONOIDS, SUM, Monoid, State, merge_all
from asofline.definitions import AggFunction

ALL = [SUM, COUNT, MIN, MAX, AVG]
FINITE = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


def states(monoid: Monoid) -> st.SearchStrategy[State]:
    """Reachable states only: an arbitrary tuple of floats is not necessarily one.

    An AVG state with a negative count, for example, cannot arise from any sequence of
    lifts and merges, and testing the laws on it would be testing nothing.
    """
    return st.lists(FINITE, min_size=0, max_size=8).map(
        lambda values: merge_all(monoid, [monoid.lift(v) for v in values])
    )


def approx(state: State) -> tuple[object, ...]:
    return tuple(pytest.approx(component, rel=1e-9, abs=1e-9) for component in state)


class TestLaws:
    @pytest.mark.parametrize("monoid", ALL, ids=lambda m: str(m.function))
    def test_identity_is_neutral_on_both_sides(self, monoid: Monoid) -> None:
        @given(state=states(monoid))
        def check(state: State) -> None:
            assert monoid.merge(monoid.identity, state) == approx(state)
            assert monoid.merge(state, monoid.identity) == approx(state)

        check()

    @pytest.mark.parametrize("monoid", ALL, ids=lambda m: str(m.function))
    def test_associativity(self, monoid: Monoid) -> None:
        @given(a=states(monoid), b=states(monoid), c=states(monoid))
        def check(a: State, b: State, c: State) -> None:
            left = monoid.merge(monoid.merge(a, b), c)
            right = monoid.merge(a, monoid.merge(b, c))
            assert left == approx(right)

        check()

    @pytest.mark.parametrize("monoid", ALL, ids=lambda m: str(m.function))
    def test_commutativity(self, monoid: Monoid) -> None:
        """Required because tiles arrive in whatever order the store hands them over."""

        @given(a=states(monoid), b=states(monoid))
        def check(a: State, b: State) -> None:
            assert monoid.merge(a, b) == approx(monoid.merge(b, a))

        check()

    @pytest.mark.parametrize("monoid", [SUM, COUNT, AVG], ids=lambda m: str(m.function))
    def test_inverse_undoes_a_merge(self, monoid: Monoid) -> None:
        """The property the batch prefix-sum path depends on.

        A window is computed as ``prefix(end) minus prefix(start)``, which is only the
        right answer if subtracting a partial state really does undo having merged it.
        """
        assert monoid.inverse is not None
        inverse = monoid.inverse

        @given(a=states(monoid), b=states(monoid))
        def check(a: State, b: State) -> None:
            assert inverse(monoid.merge(a, b), b) == approx(a)

        check()

    @pytest.mark.parametrize("monoid", [MIN, MAX], ids=lambda m: str(m.function))
    def test_extrema_have_no_inverse(self, monoid: Monoid) -> None:
        """Not an omission. There is no state that undoes having seen the minimum."""
        assert not monoid.has_inverse

    def test_arity_matches_the_identity(self) -> None:
        for monoid in ALL:
            assert len(monoid.identity) == monoid.arity
            assert len(monoid.lift(1.0)) == monoid.arity


class TestEmptyWindows:
    """``finalize(identity)`` must already be the right answer for an empty window.

    Getting this from the identity rather than from a special case downstream is what
    keeps the batch path and the online path from disagreeing about what "no events"
    means, which is a skew source that would look exactly like a data problem.
    """

    def test_sum_and_count_of_nothing_are_zero_not_null(self) -> None:
        assert SUM.finalize(SUM.identity) == 0.0
        assert COUNT.finalize(COUNT.identity) == 0.0

    def test_extrema_and_mean_of_nothing_are_null(self) -> None:
        assert MIN.finalize(MIN.identity) is None
        assert MAX.finalize(MAX.identity) is None
        assert AVG.finalize(AVG.identity) is None

    def test_a_single_event_finalizes_to_itself(self) -> None:
        for monoid in (SUM, MIN, MAX, AVG):
            assert monoid.finalize(monoid.lift(7.5)) == pytest.approx(7.5)
        assert COUNT.finalize(COUNT.lift(7.5)) == 1.0


class TestCount:
    def test_count_ignores_the_value_it_is_handed(self) -> None:
        assert COUNT.lift(0.0) == COUNT.lift(1e9) == (1.0,)

    @given(values=st.lists(FINITE, max_size=20))
    def test_count_is_the_number_of_lifts(self, values: list[float]) -> None:
        state = merge_all(COUNT, [COUNT.lift(v) for v in values])
        assert COUNT.finalize(state) == float(len(values))


class TestAverage:
    @given(values=st.lists(FINITE, min_size=1, max_size=20))
    def test_average_matches_the_arithmetic_mean(self, values: list[float]) -> None:
        assume(abs(sum(values)) > 1e-9 or len(values) == 1)
        state = merge_all(AVG, [AVG.lift(v) for v in values])
        result = AVG.finalize(state)
        assert result == pytest.approx(sum(values) / len(values), rel=1e-9, abs=1e-9)


class TestFloatAssociativityIsNotExact:
    """The reason ``rollup`` fixes a canonical merge order.

    A tolerance-based associativity test passes and hides this. The point is not that the
    error is large, it is that it is nonzero and order dependent, so two paths folding the
    same tiles in different orders disagree in the last bits forever. Canonicalising the
    order removes the disagreement instead of leaving the detector to report it every run.
    """

    def test_float_addition_is_not_associative(self) -> None:
        a, b, c = (0.1,), (0.2,), (0.3,)
        left = SUM.merge(SUM.merge(a, b), c)
        right = SUM.merge(a, SUM.merge(b, c))
        assert left == (0.6000000000000001,)
        assert right == (0.6,)
        assert left != right

    def test_the_same_tiles_in_two_orders_can_differ(self) -> None:
        """Four ordinary tile values, nothing contrived, and the totals differ."""
        tiles: list[State] = [(0.1,), (0.2,), (0.3,), (0.4,)]
        forward = merge_all(SUM, tiles)
        backward = merge_all(SUM, list(reversed(tiles)))
        assert forward == (1.0,)
        assert backward == (0.9999999999999999,)
        assert forward != backward


class TestRegistry:
    def test_every_aggregation_function_has_a_monoid(self) -> None:
        assert set(MONOIDS) == set(AggFunction)

    def test_the_definition_layer_and_the_algebra_agree_on_inverses(self) -> None:
        """Two independent statements of the same fact, kept in sync by this test.

        ``AggFunction.has_inverse`` is what the compiler branches on; ``Monoid.inverse``
        is what actually performs the subtraction. If they ever disagree, the compiler
        picks the prefix-sum path for a function that cannot subtract.
        """
        for function, monoid in MONOIDS.items():
            assert function.has_inverse == monoid.has_inverse, function

    def test_identities_are_the_extremes_for_extrema(self) -> None:
        assert math.isinf(MIN.identity[0]) and MIN.identity[0] > 0
        assert math.isinf(MAX.identity[0]) and MAX.identity[0] < 0
