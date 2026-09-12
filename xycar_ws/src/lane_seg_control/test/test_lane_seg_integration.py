"""Focused tests for the retained public LR-ASPP perception utilities."""

import unittest

import numpy as np

from lane_seg_control.lraspp_inference_node import (
    masks_from_probabilities,
    prepare_model_input,
)


class LaneSegIntegrationTest(unittest.TestCase):
    def test_input_preparation_returns_normalized_nchw(self):
        frame = np.zeros((12, 20, 3), dtype=np.uint8)
        prepared = prepare_model_input(frame, width=10, height=8)
        self.assertEqual(prepared.shape, (1, 3, 8, 10))
        self.assertTrue(prepared.flags["C_CONTIGUOUS"])

    def test_semantic_masks_are_disjoint(self):
        probabilities = np.zeros((3, 3, 4), dtype=np.float32)
        probabilities[1, 0, 1] = 0.9
        probabilities[2, 2, 3] = 0.9
        white, yellow = masks_from_probabilities(probabilities)
        self.assertEqual(int(white[0, 1]), 255)
        self.assertEqual(int(yellow[2, 3]), 255)
        self.assertEqual(int(np.count_nonzero(white & yellow)), 0)


if __name__ == "__main__":
    unittest.main()
