"""
Unit tests for Scripts.parsing.reconstruct — MRZ line normalisation, alignment,
format detection and validation scoring.

All passport strings are synthetic specimen samples (ITA). No real identity data.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Scripts.parsing import reconstruct as r


# Clean synthetic TD3 specimen used across several tests.
LINE1 = "P<ITAROSSI<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<"
LINE2 = "KF00000016ITA9011012F3307308<<<<<<<<<<<<<<08"


class TestNormalize:
    def test_uppercases(self):
        assert r._normalize("p<ita") == "P<ITA"

    def test_maps_common_filler_chars(self):
        # '(', '[' and ' ' map to '<' (text is upcased before translation,
        # so only the non-alphabetic filler substitutions take effect).
        assert r._normalize("([ ") == "<<<"

    def test_invalid_chars_become_filler(self):
        assert r._normalize("A#B%") == "A<B<"

    def test_valid_chars_preserved(self):
        text = "ABCXYZ0189<"
        assert r._normalize(text) == text


class TestSnapLine:
    def test_pads_short_line_with_filler(self):
        assert r._snap_line("ABC", 6) == "ABC<<<"

    def test_truncates_long_line(self):
        assert r._snap_line("ABCDEFG", 4) == "ABCD"

    def test_exact_length_unchanged(self):
        assert r._snap_line("ABCD", 4) == "ABCD"


class TestAlignLine:
    def test_output_has_exact_length(self):
        out = r._align_line("random text here", 44, 0)
        assert len(out) == 44

    def test_empty_input_returns_all_filler(self):
        assert r._align_line("", 5, 0) == "<" * 5

    def test_all_filler_input_returns_filler(self):
        assert r._align_line("<<<<<", 5, 0) == "<" * 5

    def test_strips_leading_filler_and_finds_data(self):
        out = r._align_line("<<<ITARO", 5, 0)
        assert out == "ITARO"

    def test_sliding_window_maximises_data(self):
        # The densest 4-char window of data should be selected.
        out = r._align_line("AB<<CDEF", 4, 0)
        assert out == "CDEF"


class TestDetectFormat:
    def test_three_lines_is_td1(self):
        assert r.detect_format(["a" * 30] * 3) == ("TD1", 30)

    def test_two_long_lines_is_td3(self):
        assert r.detect_format(["a" * 44, "b" * 44]) == ("TD3", 44)

    def test_two_short_lines_is_td2(self):
        assert r.detect_format(["a" * 36, "b" * 36]) == ("TD2", 36)

    def test_td3_boundary_at_40(self):
        assert r.detect_format(["a" * 40, "b" * 40]) == ("TD3", 44)

    def test_single_line_falls_back_to_closest(self):
        fmt, length = r.detect_format(["a" * 44])
        assert fmt == "TD3" and length == 44

    def test_single_short_line_falls_back_to_td1(self):
        fmt, length = r.detect_format(["a" * 29])
        assert fmt == "TD1" and length == 30


class TestValidationScore:
    def test_empty_lines(self):
        assert r._validation_score([]) == (0, 0.0, 0.0)

    def test_all_filler_low_density(self):
        assert r._validation_score(["<" * 44, "<" * 44]) == (0, 0.0, 0.0)

    def test_clean_specimen_scores_well(self):
        passes, composite_bonus, density = r._validation_score([LINE1, LINE2])
        assert passes > 0
        assert composite_bonus == 1.0
        assert 0.0 < density <= 1.0

    def test_density_reflects_ink_ratio(self):
        # Half-filled lines -> density around 0.5.
        line = "A" * 22 + "<" * 22
        _, _, density = r._validation_score([line, line])
        assert abs(density - 0.5) < 0.01


class TestBestAlignedPair:
    def test_recovers_clean_pair_with_leading_noise(self):
        line1, line2 = r._best_aligned_pair("<<" + LINE1, LINE2, 44)
        assert line1 == LINE1
        assert line2 == LINE2

    def test_output_lengths(self):
        line1, line2 = r._best_aligned_pair(LINE1, LINE2, 44)
        assert len(line1) == 44
        assert len(line2) == 44
