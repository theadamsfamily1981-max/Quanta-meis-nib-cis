from tfan.tls import TopologicalLandmarkSelector


def test_landmark_selection_returns_unique_indices():
    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ]
    selector = TopologicalLandmarkSelector()
    indices = selector.select(points, k=2)
    assert len(indices) == len(set(indices))
    assert all(0 <= idx < len(points) for idx in indices)


def test_landmark_selection_coverage_radius():
    selector = TopologicalLandmarkSelector()
    result = selector.select([(0.0,), (2.0,), (4.0,)], k=3, return_result=True)
    assert len(result.indices) == 3
    assert all(radius >= 0 for radius in result.radii)
