import unittest

from tff.topofusion import TopoFusion
from tff.topo_regularizer import TopologyRegularizer


def _flatten(tensor):
    if isinstance(tensor, list):
        for value in tensor:
            yield from _flatten(value)
    else:
        yield float(tensor)


class TopologyTests(unittest.TestCase):
    def test_fusion_preserves_shape_and_is_non_negative(self) -> None:
        curvature = [[[1.0, -2.0], [0.5, 0.1]]]
        flow = [[[0.1, 0.2], [0.3, 0.4]]]
        fused = TopoFusion().fuse(curvature, flow)
        self.assertEqual(len(fused), len(curvature))
        self.assertTrue(all(value >= 0.0 for value in _flatten(fused)))

    def test_curvature_ratio_is_positive(self) -> None:
        tensor = [[[0.5, 0.5], [0.5, 0.5]]]
        ratio = TopologyRegularizer().curvature_ratio(tensor)
        self.assertGreater(ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
