
"""

Unit tests for Ramses2VTKHDF package import.

These tests verify that:
1. The package can be imported without errors
2. The package exposes version metadata

"""

# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────

def test_import_and_version():
    
    """Ensure the package loads and __version__ attribute exists."""
    
    import ramses_to_vtkhdf
    assert hasattr(ramses_to_vtkhdf, "__version__")
    assert isinstance(ramses_to_vtkhdf.__version__, str)
