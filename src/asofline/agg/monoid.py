"""Partial aggregate states, and the algebra over them.

Every value this store serves is assembled by merging partial states, so the states are
the real interface between the batch path and the streaming path. They are represented
uniformly as a fixed-length tuple of floats, chosen over a class per function for two
reasons: the online codec becomes one ``struct`` format string rather than five, and the
monoid laws become directly expressible as property tests over tuples.

Arities and meanings:

    SUM    (total,)          COUNT  (n,)
    MIN    (value,)          MAX    (value,)
    AVG    (total, n)

The identity of each monoid is chosen so that ``finalize(identity)`` is already the right
answer for an empty window, with no special casing anywhere downstream: the sum of no
events is 0, the count of no events is 0, and the minimum, maximum and mean of no events
do not exist and come back as ``None``.

**Float addition is not associative.** Merging the same tiles in a different order can
produce a different last bit. Both paths therefore merge in one canonical order, ascending
by tile index, which is enforced in ``rollup``. This removes an entire class of
training-serving skew by construction rather than leaving it for the detector to find.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from asofline.definitions.aggregation import AggFunction

State = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Monoid:
    """One aggregation's algebra.

    ``inverse`` is present only for the functions that form a group. The batch compiler
    branches on it: with an inverse a window is two prefix-sum lookups, without one it
    needs a range scan over the tiles the window covers.
    """

    function: AggFunction
    arity: int
    identity: State
    merge: Callable[[State, State], State]
    lift: Callable[[float], State]
    finalize: Callable[[State], float | None]
    inverse: Callable[[State, State], State] | None = None

    @property
    def has_inverse(self) -> bool:
        return self.inverse is not None


def _sum_merge(left: State, right: State) -> State:
    return (left[0] + right[0],)


def _sum_inverse(whole: State, part: State) -> State:
    return (whole[0] - part[0],)


def _avg_merge(left: State, right: State) -> State:
    return (left[0] + right[0], left[1] + right[1])


def _avg_inverse(whole: State, part: State) -> State:
    return (whole[0] - part[0], whole[1] - part[1])


def _avg_finalize(state: State) -> float | None:
    return state[0] / state[1] if state[1] else None


def _min_finalize(state: State) -> float | None:
    return None if math.isinf(state[0]) else state[0]


def _max_finalize(state: State) -> float | None:
    return None if math.isinf(state[0]) else state[0]


SUM = Monoid(
    function=AggFunction.SUM,
    arity=1,
    identity=(0.0,),
    merge=_sum_merge,
    lift=lambda value: (value,),
    finalize=lambda state: state[0],
    inverse=_sum_inverse,
)

COUNT = Monoid(
    function=AggFunction.COUNT,
    arity=1,
    identity=(0.0,),
    merge=_sum_merge,
    # COUNT counts rows, so the value it is handed is ignored. Taking an argument anyway
    # keeps one call signature across all five functions.
    lift=lambda _value: (1.0,),
    finalize=lambda state: state[0],
    inverse=_sum_inverse,
)

MIN = Monoid(
    function=AggFunction.MIN,
    arity=1,
    identity=(math.inf,),
    merge=lambda left, right: (min(left[0], right[0]),),
    lift=lambda value: (value,),
    finalize=_min_finalize,
)

MAX = Monoid(
    function=AggFunction.MAX,
    arity=1,
    identity=(-math.inf,),
    merge=lambda left, right: (max(left[0], right[0]),),
    lift=lambda value: (value,),
    finalize=_max_finalize,
)

AVG = Monoid(
    function=AggFunction.AVG,
    arity=2,
    identity=(0.0, 0.0),
    merge=_avg_merge,
    lift=lambda value: (value, 1.0),
    finalize=_avg_finalize,
    inverse=_avg_inverse,
)

MONOIDS: dict[AggFunction, Monoid] = {
    monoid.function: monoid for monoid in (SUM, COUNT, MIN, MAX, AVG)
}


def monoid_for(function: AggFunction) -> Monoid:
    try:
        return MONOIDS[function]
    except KeyError as error:  # pragma: no cover - unreachable while AggFunction is closed
        raise KeyError(f"no monoid registered for {function}") from error


def merge_all(monoid: Monoid, states: list[State]) -> State:
    """Fold ``states`` left to right, starting from the identity.

    Left to right rather than pairwise, because the order has to be reproducible and a
    tree fold's shape depends on the input length.
    """
    accumulator = monoid.identity
    for state in states:
        accumulator = monoid.merge(accumulator, state)
    return accumulator
