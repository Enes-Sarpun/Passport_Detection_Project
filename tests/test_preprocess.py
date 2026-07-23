"""
Unit tests for Scripts.detection.preprocess — cropping, grayscale conversion,
deskew, upscaling and the full multi-variant preprocess pipeline.

Synthetic numpy images are used; no real passport data.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from Scripts.detection import preprocess as p


def _bgr_image(h=60, w=300):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


class TestCrop:
    def test_crops_to_box_with_padding(self):
        img = _bgr_image()
        out = p.crop(img, (10, 10, 200, 50))
        # Padding expands the box slightly, so height/width exceed the raw box.
        assert out.shape[0] >= 40
        assert out.shape[1] >= 190
        assert out.ndim == 3

    def test_clamps_to_image_bounds(self):
        img = _bgr_image(h=60, w=300)
        out = p.crop(img, (-20, -20, 400, 400))
        assert out.shape[0] <= 60
        assert out.shape[1] <= 300

    def test_returns_copy(self):
        img = _bgr_image()
        out = p.crop(img, (10, 10, 100, 40))
        out[0, 0] = [0, 0, 0]
        # Mutating the crop must not touch the source image.
        assert not np.array_equal(img[10, 10], np.array([0, 0, 0]))


class TestToGray:
    def test_bgr_becomes_single_channel(self):
        assert p._to_gray(_bgr_image()).ndim == 2

    def test_gray_passthrough(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        assert p._to_gray(gray).ndim == 2


class TestDeskew:
    def test_returns_same_shape(self):
        gray = _bgr_image()[:, :, 0].copy()
        assert p.deskew(gray).shape == gray.shape

    def test_too_few_ink_pixels_returns_input(self):
        gray = np.zeros((60, 300), dtype=np.uint8)
        out = p.deskew(gray)
        assert np.array_equal(out, gray)


class TestUpscale:
    def test_small_lines_upscaled(self):
        gray = np.zeros((20, 300), dtype=np.uint8)
        out = p.upscale(gray, n_lines=2)
        assert out.shape[0] > gray.shape[0]

    def test_large_enough_left_unchanged(self):
        gray = np.zeros((200, 300), dtype=np.uint8)
        out = p.upscale(gray, n_lines=2)
        assert out.shape == gray.shape


class TestPreprocessPipeline:
    def test_returns_seven_variants(self):
        img = _bgr_image()
        variants = p.preprocess(img, (10, 10, 250, 55), n_lines=2)
        assert len(variants) == 7
        assert all(v.ndim == 3 for v in variants)

    def test_toggles_do_not_crash(self):
        img = _bgr_image()
        variants = p.preprocess(
            img, (10, 10, 250, 55), n_lines=2,
            do_deskew=False, do_upscale=False, do_clahe=False,
        )
        assert len(variants) == 7

    def test_three_line_format(self):
        img = _bgr_image(h=90, w=300)
        variants = p.preprocess(img, (5, 5, 290, 85), n_lines=3)
        assert len(variants) == 7
