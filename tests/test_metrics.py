from tfan.metrics import fit_power

def test_fit_power():
    xs = [1, 2, 4, 8]
    ys = [2, 4, 8, 16]   # y = 2 * x^1
    a, alpha = fit_power(xs, ys)
    assert abs(a-2.0) < 1e-6 and abs(alpha-1.0) < 1e-6
