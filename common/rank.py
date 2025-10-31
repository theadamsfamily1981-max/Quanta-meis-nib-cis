def rank_schedule(initial: int, steps: int) -> list[int]:
    """Return a simple 64->32->8 style schedule over steps.
    This is a placeholder to be replaced by adaptive scheduling.
    """
    seq = []
    thirds = max(1, steps // 3)
    for i in range(steps):
        if i < thirds:
            seq.append(max(1, initial))
        elif i < 2 * thirds:
            seq.append(max(1, initial // 2))
        else:
            seq.append(max(1, initial // 8))
    return seq
