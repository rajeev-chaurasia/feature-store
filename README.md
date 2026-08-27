# asofline

[![ci](https://github.com/rajeev-chaurasia/feature-store/actions/workflows/ci.yml/badge.svg)](https://github.com/rajeev-chaurasia/feature-store/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A point-in-time correct feature store, where the interesting artifact is not the code but
the evidence chain: a shared window-semantics core that both the offline batch path and
the online streaming path compile against, a training-serving skew detector that found a
real production bug on its first live run rather than only the one deliberately injected
for its own test, and committed benchmark artifacts that a validator recomputes from raw
samples rather than trusts.

**Status:** all six planned phases (P0-P5) built, verified against the real stack, and
committed. See `results/` for committed evidence and the Verified numbers section below
for exact provenance.

## At a glance

| | |
|---|---|
| Test suite | 290 tests, all against the real stack, all passing together in one process |
| Online store | p50 5.34ms / p99 57.21ms at 50 QPS, 1000 requests, open loop |
| Streaming freshness | p50 34.6ms event-to-visible, 120 probes, zero failures |
| Point-in-time leak | 0.08% to 19.7% of rows change value depending on late-arrival width, measured, not assumed |
| Skew detection | clean pipeline under 2% mismatch; injected bug clears 14.3% and is named correctly |
| Found along the way | a real production bug and a real CI-blocking bug, neither of them planned |

Full detail with exact provenance for every one of these numbers is in
[Verified numbers](#verified-numbers).

## Problem

A feature store's only real job is one sentence: the value a model trains on should be
the value it would have seen in production, at that instant, given only what was known
then. Almost everything hard about building one follows from that sentence, and almost
every demo skips checking whether it actually holds. This project builds the store and
then spends comparable effort trying to catch it lying: an adversarial point-in-time test
that injects a future-dated feature change and asserts the join returns the old value, a
measured (not asserted) cost of getting point-in-time semantics wrong, and a skew
detector that is tested for false positives as rigorously as for true ones.

## Architecture

```
                         one shared definition
                  (asofline.definitions, asofline.agg)
                                  |
              +-------------------+-------------------+
              |                                        |
        BATCH PATH                              STREAMING PATH
   (Spark + Iceberg, event-time                 (Kafka -> Redis,
    tiles, point-in-time joins)                  live tile writes)
              |                                        |
    offline.tiles / compiler.batch          streaming.consumer
    offline.pit (as_of_known /                  online.store
      as_of_event_time)                       serving.app (FastAPI)
              |                                        |
              +-------------------+-------------------+
                                  |
                    skew.detector: do they still agree?
                     (feature_logs -> serving_log,
                      recompute, compare, classify)
```

Two independent consumers read `engagement_events` off Kafka, each with its own consumer
group so a stall in one is never mistaken for a stall in the other:
`streaming.to_iceberg` lands raw events for the batch path, `streaming.consumer`
maintains live tile state in Redis for the online path. The serving layer
(`serving.app`) reads Redis, answers a request, and fire-and-forgets a copy of what it
served onto `feature_logs`, which a third consumer (`streaming.feature_log_to_iceberg`)
lands in a long-form Iceberg table. `skew.detector` is what closes the loop: it
recomputes the same features the batch path's own tiled semantics would have produced and
compares.

## The unifying idea: tiles

A tile is a partial aggregate over one grid cell, storing the monoid state (a `sum` is
one float, an `avg` is `(sum, count)`, `min`/`max` the extremum) rather than the finished
value. One pure function, `asofline.agg.rollup`, merges tiles plus an exact head into a
served value, and both the batch compiler and the online consumer call it. That is what
makes "one definition compiled to two paths" a real claim rather than a decorative one:
the two paths cannot each reimplement window arithmetic slightly differently, because
neither of them implements it at all.

Two rules, enforced by property tests rather than only stated:

- **The leading edge is exact, the trailing edge snaps to the tile grid.** A window's
  effective length is in `[W, W + g)`, never exactly `W`. Snapping the leading edge too
  would trade freshness for no benefit; making the trailing edge exact would require
  keeping raw events for a full window, which is what tiles exist to avoid.
- **`as_of T` is exclusive.** An event at exactly `T` is not in the window ending at `T`.
  This is the smallest possible label leak, and the hardest to notice, so it is pinned by
  a test named for exactly that boundary.

Resolution is tiered, not a single global granularity: windows up to 12 hours resolve to
a 5-minute grid, up to 21 days to a 1-hour grid. The plan this project was built from
originally specified a single formula for this and the formula was wrong (see
`src/asofline/definitions/resolution.py`'s module docstring for the arithmetic); the
tiered ladder is what shipped, with the field cap enforced rather than documented.

## Tech stack (verified this session)

| Component | Pin | Note |
|---|---|---|
| Python | 3.12 | |
| PySpark | 4.0.4 | |
| Java | OpenJDK 17 | Spark 4.0 accepts 17 and 21 only; asserted before starting a JVM |
| Iceberg | 1.11.0, REST catalog | `S3FileIO` via `iceberg-aws-bundle`, not `hadoop-aws` |
| Object store | MinIO | |
| Kafka | 4.3.1, KRaft mode | no ZooKeeper |
| Redis | 8-alpine | persistence off; this store holds only derived state |
| Serving | FastAPI + uvicorn, `redis.asyncio`, `confluent-kafka` | |

No fallback was needed: the plan carried a documented Spark 3.5 fallback lane in case
Spark 4.0 plus Iceberg 1.11 misbehaved, decided in advance rather than at 11pm. The P0
smoke test passed on the first real run.

## Repository layout

```
src/asofline/
  definitions/    Entity, Aggregation, FeatureView, Registry -- validated at construction
  agg/             monoid states, window bounds, rollup -- the shared core, no I/O
  compiler/        a FeatureView compiles to a batch plan (spec.py, batch.py)
  offline/         Spark session, Iceberg tables, tile builder, point-in-time joins
  online/          Redis codec, key schema, head-event encoding, the async read path
  streaming/       three Kafka consumers: raw events, tile writes, feature-log ingestion
  serving/         the FastAPI app
  skew/            feature-log schema and the detector itself
  bench/           open-loop load generator, freshness probe
  experiments/     the P2 leakage measurement and lateness sweep
  demo/            the synthetic short-video engagement domain and its seeded generator
  artifacts.py     the evidence-artifact schema and validator every benchmark writes through
```

`agg/` and `definitions/` import nothing from any other package here, enforced by
`tests/unit/test_layering.py` rather than only documented: the window semantics have to
be testable with no JVM, no containers, and no network, for the same reason warpline's
correctness gate imports no `torch`.

## Verified numbers

Everything below was measured against the real stack in this repository's own test
suite or benchmark scripts, not asserted in prose.

- **P1, batch tiling:** 150 probe rows across both demo views and both compiler
  strategies (PREFIX and RANGE) match a pure-Python recomputation from raw events with
  zero mismatches. `tests/spark/test_backfill.py`
- **P2, the measured cost of getting this wrong:** row disagreement rate between
  `as_of_event_time` and `as_of_known` backfills scales from 0.08% at a 1-second late-tail
  scale to 19.7% at 30 minutes. The downstream AUC gap between a model scored honestly
  and the strict baseline stayed within +/-0.0001 at every point, including at 19.7%
  disagreement, reported because it is what came out: a logistic regression over these
  features rarely has its decision flipped by one late event. `src/asofline/experiments/`
- **P3, online-store latency:** 1000 requests at 50 QPS, open loop, real seeded Redis
  state, real FastAPI process. p50 5.34ms / p90 7.24ms / p99 57.21ms with fire-and-forget
  feature logging enabled, versus p50 4.79ms / p90 8.62ms / p99 40.93ms with it disabled:
  logging is not fully free of the request path under a single-worker process, and the
  comparison is committed as a pair rather than assumed away.
  `results/2026-08-24-online-latency/`
- **P4, streaming freshness:** 120 real event-to-visible-in-serving-response probes
  against a live Kafka-to-Redis consumer subprocess, zero failures. p50 34.6ms / p99
  39.7ms, dominated by the probe's own 30ms poll interval rather than by the pipeline.
  `results/p4-streaming-freshness/`
- **P5, skew detection:** a clean pipeline stays under 2% mismatch on every feature. A
  deliberately injected bug (drop the head merge for one feature, on half the served
  vectors) produces a 14.3% mismatch rate on that feature alone, landing entirely in the
  `partial_head_tile` bucket since `late_fraction=0` in that scenario rules out the other
  one. Sensitivity measured directly at 2% and 10% injection rates rather than assumed:
  observed mismatch rates of 0.67% and 3.33% respectively, both roughly 29% of the
  injected fraction, because most entities in this Zipf-skewed population have no
  activity in any given hour to lose in the first place. `tests/spark/test_skew_detector.py`
- **A real bug, not the planned one:** the detector's first live run, before any bug was
  deliberately injected, flagged a reproducible undercount on `count_1d` concentrated on
  the most active synthetic users. Root cause: the online Redis head stored recent events
  in a sorted set keyed only on their column values, and every impression event has
  identical columns, so two impressions in the same window silently collapsed into one
  entry. Fixed by keying each entry on the source event's id instead. See the commit
  titled *"Fix a real head-ZSET collision bug: found by P5's detector on its first live
  run"*.
- **Test suite:** 239 unit tests (no JVM, no containers, no network), 20 integration and
  serving tests against real Redis/Kafka/FastAPI, 31 Spark tests, all passing together in
  one process, which is what CI actually runs. 24 test files, roughly 4,450 lines of test
  code against roughly 5,900 lines of source.
- **A real CI-blocking bug, found and fixed along the way:** `SparkSession.builder.getOrCreate()`
  is a JVM-wide singleton, and two independently built streaming test files each believed
  they owned an isolated, Kafka-jar-carrying session and stopped it at teardown. Running
  the whole `tests/spark` directory together, exactly what CI does, either failed outright
  or corrupted every later test once the shared session was stopped mid-run. Fixed by
  giving every session this project builds the same jar set rather than choosing per job.

## Honest limitations

- **Trailing window edges snap to the tile grid.** Effective window length is in
  `[W, W + g)`, never exactly `W`. The leading edge is exact.
- **Only monoid-mergeable aggregations are supported:** `sum`, `count`, `min`, `max`,
  `avg`. No exact median, no exact distinct count, because tiles cannot represent them.
- **Single-node everything.** No Redis cluster, no Kafka partition rebalancing story
  beyond what one broker gives, no hot-key mitigation beyond what the demo's Zipf
  generator happens to make visible.
- **At-least-once ingestion, made explicit rather than hidden.** The Kafka-to-Redis
  consumer commits offsets only after a batch's Redis writes succeed, so a crash before
  that commit reprocesses the batch on restart. A non-idempotent monoid (`sum`, `count`)
  double-counts on redelivery, demonstrated directly by a test rather than only claimed in
  a comment. Building genuine exactly-once dedup was out of scope.
- **The skew detector's ground truth is the tiled, event-time recomputation, not the
  strict point-in-time one**, by design (see `skew/detector.py`'s module docstring): this
  isolates implementation skew from the semantic leak P2 already measures, but it means a
  clean detector run says nothing about whether the online path also reproduces
  `as_of_known` semantics, only that it correctly reproduces `as_of_event_time`.
- **The freshness probe's own poll interval (30ms) dominates its reported latency.**
  The true event-to-visible latency in steady state is evidently well under that; a finer
  poll interval would sharpen the number at the cost of hammering Redis for a
  measurement, not a workload.

## Running it

```bash
docker compose up -d --wait
```

```bash
uv sync --group dev --all-extras
```

```bash
uv run pytest tests/unit -q
```

```bash
uv run pytest tests/spark tests/integration tests/serving -q
```

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

```bash
uv run python scripts/validate_artifacts.py results/
```

End to end: bring the stack up, run the unit and Spark/integration suites, run
`scripts/run_online_benchmark.py` and `scripts/run_freshness_probe.py` to reproduce the
committed benchmark artifacts, and run `tests/spark/test_skew_detector.py` to watch the
detector catch a deliberately injected bug and correctly ignore a clean run.

## Style

SOLID, DRY, YAGNI. No new abstraction with a single caller. Comments explain why, not
what. No em-dashes, no AI attribution in code, commits, or documentation. Ruff at 100
columns, mypy strict on `src`.
