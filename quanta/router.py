def router_temperature_sweep(tau: float) -> float:
    """Return entropy proxy for softmax temp.
    For tau in (0.3..2.0), lower -> sharper, higher -> flatter.
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    # proxy: 1 / tau capped to [0, 10]
    val = 1.0 / tau
    return min(10.0, max(0.0, val))
