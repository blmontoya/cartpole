def test_import_package():
    """Test that the package has the expected structure."""
    import cartpole

    # Check that __version__ exists
    assert hasattr(cartpole, "__version__"), "Package should have __version__ attribute"


def test_import_core():
    """Test that the core module can be imported."""
    from cartpole import core

    assert core is not None
