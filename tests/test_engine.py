"""
Unit tests for Scripts.ocr.engine — pure text/geometry helpers and the
result dataclasses. The actual Tesseract call is not exercised (it needs the
tesseract binary + OCR-B model); only deterministic helpers are covered.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from Scripts.ocr import engine as e


class TestCleanChar:
    def test_valid_letter_unchanged(self):
        assert e._clean_char("A") == "A"

    def test_lowercase_upcased(self):
        assert e._clean_char("a") == "A"

    def test_space_becomes_filler(self):
        assert e._clean_char(" ") == "<"

    def test_unknown_symbol_becomes_filler(self):
        assert e._clean_char("#") == "<"

    def test_junk_map_letter_to_digit(self):
        # lowercase letters that upcase into the valid set stay as letters,
        # but punctuation in the junk map is remapped to a valid char.
        assert e._clean_char(".") == "<"
        assert e._clean_char("-") == "<"


class TestCleanText:
    def test_strips_and_normalises(self):
        assert e._clean_text("  p<ita  ") == "P<ITA"

    def test_symbols_replaced_with_filler(self):
        assert e._clean_text("A B") == "A<B"


class TestSnap:
    def test_pads_short(self):
        assert e._snap("ABC", 5) == "ABC<<"

    def test_truncates_long(self):
        assert e._snap("ABCDEF", 3) == "ABC"

    def test_exact_length(self):
        assert e._snap("ABCD", 4) == "ABCD"


class TestUpscale:
    def test_small_image_is_upscaled(self):
        img = np.zeros((10, 20), dtype=np.uint8)
        out = e._upscale(img, min_h=80)
        assert out.shape[0] == 80

    def test_tall_image_left_unchanged(self):
        img = np.zeros((100, 20), dtype=np.uint8)
        out = e._upscale(img, min_h=80)
        assert out.shape == (100, 20)


class TestResultDataclasses:
    def test_mean_confidence_empty(self):
        assert e.TesseractResult(lines=[]).mean_confidence == 0.0

    def test_mean_confidence_averaged(self):
        result = e.TesseractResult(
            lines=[e.TesseractLine("A", 0.4), e.TesseractLine("B", 0.6)]
        )
        assert result.mean_confidence == 0.5

    def test_default_engine_name(self):
        assert e.TesseractResult(lines=[]).engine == "tesseract_ocrb"


class TestResolvers:
    def test_tesseract_cmd_prefers_env(self, monkeypatch):
        monkeypatch.setenv("TESSERACT_CMD", "/custom/tesseract")
        assert e._resolve_tesseract_cmd() == "/custom/tesseract"

    def test_tesseract_cmd_falls_back(self, monkeypatch):
        monkeypatch.delenv("TESSERACT_CMD", raising=False)
        # Either resolves via PATH or the Windows default; never empty.
        assert e._resolve_tesseract_cmd()

    def test_tessdata_dir_prefers_env(self, monkeypatch):
        monkeypatch.setenv("TESSDATA_PREFIX", "/custom/tessdata")
        assert str(e._resolve_tessdata_dir()) == "/custom/tessdata"

    def test_tessdata_dir_finds_bundled(self, monkeypatch):
        monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
        path = e._resolve_tessdata_dir()
        # The repo bundles ocrb.traineddata next to the engine module.
        assert path.name in ("tessdata", "tessdata_ocrb")
