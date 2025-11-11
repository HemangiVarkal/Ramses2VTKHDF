"""
Unit tests for Chhavi parsing helper functions

Validates CLI argument parsing utilities: output numbers, normalized ranges, and fields.

"""

import pytest
from chhavi.converter import parse_output_numbers, parse_norm_range, parse_fields_arg


# ──────────────────────────────────────────────────────────────
# Output numbers parsing
# ──────────────────────────────────────────────────────────────

def test_parse_output_numbers_single():
    assert parse_output_numbers("5") == [5]


def test_parse_output_numbers_range():
    assert parse_output_numbers("2-4") == [2, 3, 4]


def test_parse_output_numbers_list():
    assert parse_output_numbers("1,3,7") == [1, 3, 7]


# ──────────────────────────────────────────────────────────────
# Normalized range parsing
# ──────────────────────────────────────────────────────────────

def test_parse_norm_range_valid():
    assert parse_norm_range("0.2:0.8") == (0.2, 0.8)


def test_parse_norm_range_colon_variants():
    assert parse_norm_range(":0.6") == (0.0, 0.6)
    assert parse_norm_range("0.4:") == (0.4, 1.0)
    assert parse_norm_range(":") == (0.0, 1.0)


# ──────────────────────────────────────────────────────────────
# Fields argument parsing
# ──────────────────────────────────────────────────────────────

def test_parse_fields_arg_none():
    assert parse_fields_arg(None) is None


def test_parse_fields_arg_valid():
    assert parse_fields_arg("density,velocity") == ["density", "velocity"]
