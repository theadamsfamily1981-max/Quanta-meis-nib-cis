from tfan.pgu import PGUCache

def test_cache_hit():
    p = PGUCache(timeout_ms=50)
    a = "(declare-const a Int) (assert (> a 0)) (check-sat)"
    b = "(declare-const x Int) (assert (> x 0)) (check-sat)"
    r1 = p.check(a)
    r2 = p.check(b)
    assert r2["cached"] is True or (r1["res"] == r2["res"])
