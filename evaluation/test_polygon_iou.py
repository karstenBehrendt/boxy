#!/usr/bin/env python3
"""
Test polygon intersections

Covers a very limited sample set to check if the iou functions are working
as intended. The tests are too limited to assume correctness. The tests
also only cover quadrilaterals.

"""
# NOTE Ignoring incorrect inputs for first (and likely last) version of tests
# NOTE The tests are in now way covering necessary test cases.

# TODO Negative Values
# TODO Self intersecting polygon
# TODO Different ordering of points
# TODO More challenging shapes

import copy
import unittest

from polygon_iou import polygon_iou as iou


class TestZeroIOU(unittest.TestCase):
    def test_zero_iou(self):
        poly1 = [(0, 0), (3, 0), (3, 3), (0, 3)]
        poly2 = [[coord + 15 for coord in point] for point in poly1]
        self.assertEqual(0, iou(poly1, poly2))


class TestFullIOU(unittest.TestCase):
    def test_full_iou(self):
        poly1 = [(0, 0), (3, 0), (3, 3), (0, 3)]
        self.assertEqual(1, iou(poly1, poly1))


class TestInputUnchanged(unittest.TestCase):
    def test_input_equality(self):
        poly1 = [(1, 3), (3.14, 15), (13, 12), (4, 4)]
        original_poly1 = copy.deepcopy(poly1)
        poly2 = [[1.33, 3.33], [3.14, 15.12], [13.123, 12.13], [4.3, 4]]
        original_poly2 = copy.deepcopy(poly2)
        iou(poly1, poly2)
        iou(poly2, poly1)  # only one may be changed, checking tuple and list
        self.assertSequenceEqual(poly1, original_poly1)
        self.assertSequenceEqual(poly2, original_poly2)


class TestEasySquares(unittest.TestCase):
    def test_easy_square(self):
        poly1 = [(0, 0), (3, 0), (3, 3), (0, 3)]
        poly2 = [[coord / 4.0 for coord in point] for point in poly1]
        self.assertAlmostEqual(0.0625, iou(poly1, poly2))


class TestEasyPhombus(unittest.TestCase):
    def test_easy_rhombus(self):
        poly1 = [(0, 0), (1, 1), (3, 1), (2, 0)]
        poly2 = [(1, 0), (2, 1), (4, 1), (3, 0)]
        # Area each 2, intersection 1, union 3
        self.assertAlmostEqual(1 / 3.0, iou(poly1, poly2))


if __name__ == '__main__':
    unittest.main()
