"""
Unit tests for GroundTruth.evaluate — the accuracy metrics (Levenshtein, CER,
field accuracy) and the report/summary helpers. The image pipeline itself is
not run; only the deterministic scoring logic is covered.
"""
from __future__ import annotations
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from GroundTruth import evaluate as ev


LINE1 = "P<ITAROSSI<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<"
LINE2 = "KF00000016ITA9011012F3307308<<<<<<<<<<<<<<08"


class TestLevenshtein:
    def test_identical(self):
        assert ev.levenshtein("abc", "abc") == 0

    def test_empty_first(self):
        assert ev.levenshtein("", "abc") == 3

    def test_empty_second(self):
        assert ev.levenshtein("abc", "") == 3

    def test_classic_kitten_sitting(self):
        assert ev.levenshtein("kitten", "sitting") == 3

    def test_symmetric(self):
        assert ev.levenshtein("flaw", "lawn") == ev.levenshtein("lawn", "flaw")


class TestCer:
    def test_perfect_match_is_zero(self):
        assert ev.cer([LINE1, LINE2], [LINE1, LINE2]) == 0.0

    def test_single_substitution(self):
        assert ev.cer(["ABCD"], ["ABXD"]) == 0.25

    def test_missing_line_counts_as_errors(self):
        # Second predicted line absent -> all its chars are errors.
        assert ev.cer(["abc", "def"], ["abc"]) == 0.5

    def test_empty_inputs(self):
        assert ev.cer([], []) == 0.0


class TestFieldsFromLines:
    def test_parses_specimen(self):
        fields = ev._fields_from_lines([LINE1, LINE2])
        assert fields["surname"] == "ROSSI"
        assert fields["nationality"] == "ITA"
        assert set(fields) == set(ev._FIELDS)

    def test_unparseable_returns_empty_fields(self):
        fields = ev._fields_from_lines([])
        assert fields == {f: "" for f in ev._FIELDS}


class TestFieldAccuracy:
    def test_all_match(self):
        fields = ev._fields_from_lines([LINE1, LINE2])
        acc, failed = ev.field_accuracy(fields, fields)
        assert acc == 1.0
        assert failed == []

    def test_one_mismatch(self):
        fields = ev._fields_from_lines([LINE1, LINE2])
        wrong = {**fields, "surname": "WRONG"}
        acc, failed = ev.field_accuracy(fields, wrong)
        assert failed == ["surname"]
        assert acc == 1.0 - 1 / len(ev._FIELDS)

    def test_missing_key_counts_as_failure(self):
        fields = ev._fields_from_lines([LINE1, LINE2])
        acc, failed = ev.field_accuracy(fields, {})
        assert set(failed) == set(ev._FIELDS)
        assert acc == 0.0


class TestWriteReport:
    def test_writes_csv_with_header(self, tmp_path, monkeypatch):
        report = tmp_path / "report.csv"
        monkeypatch.setattr(ev, "_REPORT_PATH", report)
        rows = [{
            "stem": "s1", "country": "ITA", "status": "ok",
            "cer": 0.0, "field_acc": 1.0, "failed_fields": "",
        }]
        ev.write_report(rows)
        with open(report, newline="", encoding="utf-8") as f:
            read_rows = list(csv.DictReader(f))
        assert len(read_rows) == 1
        assert read_rows[0]["stem"] == "s1"
        assert read_rows[0]["status"] == "ok"


class TestPrintSummary:
    def test_no_scored_rows(self, capsys):
        ev.print_summary([{"status": "no_mrz", "cer": ""}])
        assert "No scored rows." in capsys.readouterr().out

    def test_summary_reports_metrics(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setattr(ev, "_REPORT_PATH", tmp_path / "r.csv")
        rows = [
            {"stem": "a", "country": "ITA", "status": "ok",
             "cer": 0.0, "field_acc": 1.0, "failed_fields": ""},
            {"stem": "b", "country": "DEU", "status": "ok",
             "cer": 0.1, "field_acc": 0.8, "failed_fields": "surname"},
            {"stem": "c", "country": "FRA", "status": "no_mrz",
             "cer": "", "field_acc": "", "failed_fields": ""},
        ]
        ev.print_summary(rows)
        out = capsys.readouterr().out
        assert "ACCURACY SUMMARY" in out
        assert "Scored images:        2" in out
        assert "surname" in out
