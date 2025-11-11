"""
Unit tests for Chhavi package import.

These tests verify that:
1. The package can be imported without errors
2. The package exposes version metadata

"""

# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────

def test_import_and_version():
    """Ensure the package loads and __version__ attribute exists."""
    import chhavi
    assert hasattr(chhavi, "__version__")
    assert isinstance(chhavi.__version__, str)
