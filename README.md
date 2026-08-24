# asofline

A point-in-time correct feature store. Iceberg offline, Redis online, one feature
definition compiled to both the batch and the streaming path, and a training-serving
skew detector that is tested for false positives as well as true ones.

Status: in progress. See `docs/` for the design and `results/` for committed evidence.
