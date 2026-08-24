"""A seeded, deterministic engagement event generator.

Determinism is load-bearing rather than a convenience. Every number this project commits
is computed over a generated stream, so if the stream is not reproducible then neither is
the evidence, and the artifact validator is checking arithmetic on sand.

Three properties are deliberately built in, because without them the measurements would
be flattering rather than informative:

* **Zipf over both entities.** Real viewers and real videos are not uniform. A uniform
  generator hides hot-key behaviour in the online store and makes p99 look like p50.
* **A configurable late tail.** ``created_ts`` exceeds ``event_ts`` by a heavy-tailed
  delay for a fraction of events. This is the entire reason point-in-time correctness is
  hard, and a generator without it lets a leaky backfill pass every test.
* **Arrival ordering.** ``generate`` returns events sorted by ``created_ts``, because
  that is the order a consumer sees them. Sorting by ``event_ts`` instead would quietly
  remove the late tail it just built.
"""

from __future__ import annotations

import bisect
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from asofline.demo.events import EngagementEvent, EventType

# Rough mix for a feed product: most impressions never become a watch, and the
# engagement types thin out sharply after that.
_TYPE_WEIGHTS: tuple[tuple[EventType, float], ...] = (
    (EventType.IMPRESSION, 0.60),
    (EventType.WATCH, 0.30),
    (EventType.LIKE, 0.06),
    (EventType.SHARE, 0.03),
    (EventType.FOLLOW, 0.01),
)

_MAX_LATENESS_MS = 6 * 60 * 60 * 1000
"""Lateness is capped so the tail is heavy but bounded.

An uncapped Pareto draw occasionally produces an event that arrives days after it
happened, which makes any fixed-length backfill window nondeterministic in a way that has
nothing to do with the behaviour being measured.
"""


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Everything that varies between runs, in one hashable object.

    It is hashable on purpose: the run artifacts record it, so a committed measurement
    names the exact stream it was computed over.
    """

    seed: int = 20260823
    n_users: int = 5_000
    n_videos: int = 20_000
    n_events: int = 200_000
    start: datetime = field(default=datetime(2026, 8, 1, tzinfo=UTC))
    duration: timedelta = field(default=timedelta(days=10))
    user_zipf_exponent: float = 0.8
    video_zipf_exponent: float = 1.1
    late_fraction: float = 0.05
    prompt_delay_ms: int = 250
    late_delay_scale_ms: int = 60_000
    late_delay_alpha: float = 1.5

    def __post_init__(self) -> None:
        if self.start.tzinfo is None:
            raise ValueError("start must be timezone aware")
        if not 0.0 <= self.late_fraction <= 1.0:
            raise ValueError(f"late_fraction must be in [0, 1], got {self.late_fraction}")
        if self.late_delay_alpha <= 1.0:
            # A Pareto with alpha <= 1 has no finite mean, so the observed late fraction
            # would be dominated by whichever single draw happened to be largest.
            raise ValueError(f"late_delay_alpha must exceed 1, got {self.late_delay_alpha}")
        for name in ("n_users", "n_videos", "n_events"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("user_zipf_exponent", "video_zipf_exponent"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


class _ZipfSampler:
    """Sample ``0..n-1`` with weight proportional to ``1 / (rank + 1) ** exponent``."""

    __slots__ = ("_cumulative", "_total")

    def __init__(self, n: int, exponent: float) -> None:
        cumulative: list[float] = []
        running = 0.0
        for rank in range(n):
            running += 1.0 / (rank + 1) ** exponent
            cumulative.append(running)
        self._cumulative = cumulative
        self._total = running

    def sample(self, rng: random.Random) -> int:
        return bisect.bisect_left(self._cumulative, rng.random() * self._total)


class EngagementGenerator:
    """Builds a reproducible stream of engagement events."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        # Videos are more skewed than viewers. A single video going viral is normal;
        # a single viewer generating a sixth of all site traffic is not, and at
        # exponent 1.1 over 5000 users that is exactly what the sampler produced.
        self._users = _ZipfSampler(self.config.n_users, self.config.user_zipf_exponent)
        self._videos = _ZipfSampler(self.config.n_videos, self.config.video_zipf_exponent)

    def generate(self) -> list[EngagementEvent]:
        """Every event, ordered by ``created_ts``, which is arrival order."""
        events = self._generate_unordered()
        # Stable on event_id so two events sharing a created_ts still order identically
        # across runs. Without the tiebreak the sort is deterministic only by accident.
        events.sort(key=lambda event: (event.created_ts, event.event_id))
        return events

    def _generate_unordered(self) -> list[EngagementEvent]:
        config = self.config
        rng = random.Random(config.seed)
        start_ms = int(config.start.timestamp() * 1000)
        span_ms = int(config.duration.total_seconds() * 1000)

        types = [event_type for event_type, _ in _TYPE_WEIGHTS]
        weights = [weight for _, weight in _TYPE_WEIGHTS]

        events: list[EngagementEvent] = []
        for index in range(config.n_events):
            event_ts = start_ms + rng.randrange(span_ms)
            event_type = rng.choices(types, weights=weights, k=1)[0]
            events.append(
                EngagementEvent(
                    event_id=f"e{index:012d}",
                    event_type=event_type,
                    user_id=f"u{self._users.sample(rng):07d}",
                    video_id=f"v{self._videos.sample(rng):07d}",
                    event_ts=event_ts,
                    created_ts=event_ts + self._lateness_ms(rng),
                    watch_seconds=self._watch_seconds(rng, event_type),
                    liked=1 if event_type is EventType.LIKE else 0,
                    shared=1 if event_type is EventType.SHARE else 0,
                )
            )
        return events

    def _lateness_ms(self, rng: random.Random) -> int:
        config = self.config
        if rng.random() >= config.late_fraction:
            # The ordinary path still is not instant. A few hundred milliseconds of
            # transport delay is what separates event time from ingest time normally.
            return rng.randrange(config.prompt_delay_ms + 1)
        draw = rng.paretovariate(config.late_delay_alpha) * config.late_delay_scale_ms
        return min(int(draw), _MAX_LATENESS_MS)

    @staticmethod
    def _watch_seconds(rng: random.Random, event_type: EventType) -> float | None:
        if event_type is not EventType.WATCH:
            # Null, not zero. An impression has no watch duration, and calling it zero
            # would drag every average down by the impression rate.
            return None
        # Short-video watch time is right skewed: most watches are partial, a few loop.
        return round(min(rng.lognormvariate(2.0, 0.9), 600.0), 3)
