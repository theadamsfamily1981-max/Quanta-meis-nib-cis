from tfan.pgu import ProofGoalCache, Z3LiteProofChecker


def test_cache_distinguishes_substitutions():
    cache = ProofGoalCache()
    checker = Z3LiteProofChecker(cache)
    goal = "x + y == 2"
    assert checker.check(goal, {"x": 1, "y": 1}) is True
    assert checker.check(goal, {"x": 2, "y": 0}) is True
    # Cache must differentiate the substitution assignments.
    assert len(cache._store) == 2


def test_cache_hits_are_returned_directly():
    cache = ProofGoalCache()
    checker = Z3LiteProofChecker(cache)
    goal = "x * x == y"
    assert checker.check(goal, {"x": 2, "y": 4})
    # Second invocation should be a cache hit and therefore avoid recomputation.
    assert checker.check(goal, {"x": 2, "y": 4})
    assert len(cache._store) == 1
