import unittest

from tff.topo_regularizer import TopologyRegularizer
from udk.utcfloss import UTCFLoss


class UTCFLossTests(unittest.TestCase):
    def test_loss_is_non_negative(self) -> None:
        fused = [[[1.0, 1.0], [1.0, 1.0]]]
        safety = [[[0.0, 0.0], [0.0, 0.0]]]
        energy = [[[0.0, 0.0], [0.0, 0.0]]]
        loss = UTCFLoss(TopologyRegularizer())
        metrics = loss(fused, safety, energy)
        self.assertGreaterEqual(metrics["total"], 0.0)
        self.assertGreater(metrics["curvature_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
