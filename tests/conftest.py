import pytest

from wetware import MEA, Culture


@pytest.fixture
def small_culture() -> Culture:
    """A small, fast culture used across tests."""
    return Culture(n_neurons=300, seed=7, mea=MEA(rows=4, cols=4, pitch_um=200.0))
