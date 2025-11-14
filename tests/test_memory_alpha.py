from tfan.metrics import fit_power

def test_alpha_sublinear():
    xs = [1000, 2000, 4000]
    ys = [1.0, 1.6, 2.3]  # ~ x^0.53
    a, alpha = fit_power(xs, ys)
    assert alpha < 1.0
