# Hardware Profiling

The launch kit includes a synthetic hardware profile that mirrors the telemetry
collection performed in the full Phase III stack. The helper functions provided
by `benchmark_suite.hardware_profile` produce deterministic yet realistic
metadata that is embedded in the launch payload.

## Default profile

`default_hardware_profile()` constructs a `HardwareProfile` instance populated
with representative power and temperature readings. These values contribute to
the derived `efficiency_score` which is calculated using memory capacity,
compute capability, and configured power limits.

```python
from benchmark_suite.hardware_profile import default_hardware_profile

profile = default_hardware_profile()
print(profile.summary())
```

## Custom profiles

Instantiate `HardwareProfile` directly to describe additional nodes. Once
recorded, use `merge_profiles()` to aggregate multiple summaries into a single
structure suitable for serialisation.
