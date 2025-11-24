from tfan.fdr import benjamini_hochberg, estimate_fdr_thresholds


def test_benjamini_hochberg_discovers_expected_hypotheses():
    p_values = [0.001, 0.01, 0.02, 0.2, 0.5]
    result = benjamini_hochberg(p_values, alpha=0.05)
    assert result.threshold == p_values[2]
    assert result.discoveries == (0, 1, 2)


def test_estimate_wrapper_supports_named_method():
    p_values = [0.05, 0.1, 0.2]
    result = estimate_fdr_thresholds(p_values, method="benjamini-yekutieli")
    assert result.threshold <= max(p_values)
