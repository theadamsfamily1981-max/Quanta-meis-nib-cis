import unittest

from tff.landmark_attn import LandmarkAttention


def _flatten(tensor):
    if isinstance(tensor, list):
        for value in tensor:
            yield from _flatten(value)
    else:
        yield float(tensor)


def _sum_last_axis(tensor):
    return [sum(row) for batch in tensor for row in batch]


class LandmarkAttentionTests(unittest.TestCase):
    def test_attention_weights_are_probabilities(self) -> None:
        queries = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
        landmarks = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
        weights, summary = LandmarkAttention().attend(queries, landmarks)
        self.assertEqual(len(weights), 1)
        self.assertTrue(all(value >= 0.0 for value in _flatten(weights)))
        self.assertTrue(all(abs(total - 1.0) <= 1e-6 for total in _sum_last_axis(weights)))
        self.assertEqual(len(summary), 1)


if __name__ == "__main__":
    unittest.main()
