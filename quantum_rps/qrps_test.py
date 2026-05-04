import numpy as np
import pytest
from circuit import expected_payoff, fourier_3, shift_3

# --- Helpers for classical strategies ---
def classical_strategy(choice):
    """Return unitary that always outputs a fixed classical choice.
    choice: 0=Rock, 1=Paper, 2=Scissors
    """
    X = shift_3()
    return np.linalg.matrix_power(X, choice)

@pytest.fixture
def J_identity():
    """Identity J gate -> classical game (no entanglement)."""
    return np.eye(9, dtype=complex)

def test_balanced_strategy_is_fair(J_identity):
    F = fourier_3()  # balanced strategy
    payoff = expected_payoff(F, classical_strategy(0), J=J_identity)
    assert abs(payoff) < 1e-10

    payoff = expected_payoff(F, classical_strategy(1), J=J_identity)
    assert abs(payoff) < 1e-10

    payoff = expected_payoff(F, classical_strategy(2), J=J_identity)
    assert abs(payoff) < 1e-10

def test_rock_beats_scissors(J_identity):
    payoff = expected_payoff(classical_strategy(0), classical_strategy(2), J=J_identity)
    assert pytest.approx(payoff, abs=1e-10) == 1.0

def test_rock_loses_to_paper(J_identity):
    payoff = expected_payoff(classical_strategy(0), classical_strategy(1), J=J_identity)
    assert pytest.approx(payoff, abs=1e-10) == -1.0

def test_same_choice_is_draw(J_identity):
    payoff = expected_payoff(classical_strategy(1), classical_strategy(1), J=J_identity)
    assert pytest.approx(payoff, abs=1e-10) == 0.0
