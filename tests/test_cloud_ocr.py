"""
Unit tests for Scripts.ocr.cloud_ocr — the optional Google Vision fallback.

The network call is fully mocked; no real API key or request is made.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import requests

from Scripts.ocr import cloud_ocr as c


LINE1 = "P<ITAROSSI<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<"
LINE2 = "KF00000016ITA9011012F3307308<<<<<<<<<<<<<<08"


def _blank_image():
    return np.zeros((40, 400, 3), dtype=np.uint8)


class _FakeResponse:
    def __init__(self, payload, raise_error=False):
        self._payload = payload
        self._raise = raise_error

    def raise_for_status(self):
        if self._raise:
            raise requests.HTTPError("boom")

    def json(self):
        return self._payload


class TestIsAvailable:
    def test_false_without_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_VISION_API_KEY", raising=False)
        assert c.is_available() is False

    def test_true_with_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_VISION_API_KEY", "secret")
        assert c.is_available() is True


class TestLooksLikeMrz:
    def test_short_line_rejected(self):
        assert c._looks_like_mrz("ABC") is False

    def test_mrz_line_accepted(self):
        assert c._looks_like_mrz(LINE1) is True

    def test_symbol_heavy_line_rejected(self):
        assert c._looks_like_mrz("!" * 30) is False


class TestExtractMrzLines:
    def test_extracts_and_normalises(self):
        text = f"{LINE1}\n{LINE2}\nshort junk"
        lines = c._extract_mrz_lines(text)
        assert LINE1 in lines
        assert LINE2 in lines
        assert all(len(l) in (30, 36, 44) for l in lines)

    def test_no_mrz_returns_empty(self):
        assert c._extract_mrz_lines("hello world\nfoo bar") == []

    def test_caps_at_three_lines(self):
        text = "\n".join([LINE1] * 5)
        assert len(c._extract_mrz_lines(text)) <= 3


class TestGoogleVisionText:
    def test_returns_none_without_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_VISION_API_KEY", raising=False)
        assert c._google_vision_text(_blank_image()) is None

    def test_returns_full_text_annotation(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_VISION_API_KEY", "k")
        payload = {"responses": [{"fullTextAnnotation": {"text": "HELLO"}}]}
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(payload))
        assert c._google_vision_text(_blank_image()) == "HELLO"

    def test_falls_back_to_text_annotations(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_VISION_API_KEY", "k")
        payload = {"responses": [{"textAnnotations": [{"description": "WORLD"}]}]}
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(payload))
        assert c._google_vision_text(_blank_image()) == "WORLD"

    def test_empty_responses_returns_none(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_VISION_API_KEY", "k")
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse({"responses": []}))
        assert c._google_vision_text(_blank_image()) is None

    def test_network_error_returns_none(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_VISION_API_KEY", "k")

        def _boom(*a, **k):
            raise requests.ConnectionError("no network")

        monkeypatch.setattr(requests, "post", _boom)
        assert c._google_vision_text(_blank_image()) is None

    def test_http_error_returns_none(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_VISION_API_KEY", "k")
        monkeypatch.setattr(
            requests, "post", lambda *a, **k: _FakeResponse({}, raise_error=True)
        )
        assert c._google_vision_text(_blank_image()) is None


class TestReadMrz:
    def test_returns_none_when_unavailable(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_VISION_API_KEY", raising=False)
        assert c.read_mrz(_blank_image()) is None

    def test_returns_none_when_no_text(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_VISION_API_KEY", "k")
        monkeypatch.setattr(c, "_google_vision_text", lambda img: None)
        assert c.read_mrz(_blank_image()) is None

    def test_returns_lines_on_success(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_VISION_API_KEY", "k")
        monkeypatch.setattr(c, "_google_vision_text", lambda img: f"{LINE1}\n{LINE2}")
        lines = c.read_mrz(_blank_image())
        assert lines is not None
        assert LINE1 in lines and LINE2 in lines

    def test_returns_none_when_no_mrz_in_text(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_VISION_API_KEY", "k")
        monkeypatch.setattr(c, "_google_vision_text", lambda img: "no mrz here")
        assert c.read_mrz(_blank_image()) is None
