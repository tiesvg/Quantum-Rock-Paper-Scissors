import numpy as np

# -------------------------------------------------------------
# Qutrit primitives
# -------------------------------------------------------------

def fourier_3():
    """3x3 discrete Fourier transform matrix (qutrit Hadamard)."""
    omega = np.exp(2j * np.pi / 3)
    F = np.array([[1, 1, 1],
                  [1, omega, omega**2],
                  [1, omega**2, omega]], dtype=complex)
    return F / np.sqrt(3)

def shift_3():
    """Qutrit cyclic shift X: |0>->|1>, |1>->|2>, |2>->|0>."""
    X = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]], dtype=complex)
    return X

def add_3():
    """Controlled-ADD gate for qutrits (9x9 matrix).
    ADD_3 |a,b> = |a, b+a mod 3>.
    """
    d = 3
    U = np.zeros((d*d, d*d), dtype=complex)
    for a in range(d):
        for b in range(d):
            inp = a*d + b
            out = a*d + ((b + a) % d)
            U[out, inp] = 1
    return U

# -------------------------------------------------------------
# J gate (based on ADD_3 and F_3)
# -------------------------------------------------------------

def J_gate_maximal_entanglement():
    """Construct the J gate as (F† ⊗ I) ADD_3 (F ⊗ I)."""
    F = fourier_3()
    I = np.eye(3)
    ADD = add_3()
    U = np.kron(F.conj().T, I) @ ADD @ np.kron(F, I)
    return U

# -------------------------------------------------------------
# Quantum RPS circuit
# -------------------------------------------------------------

def qrps_circuit(S1, S2, J=None):
    """Return the 9x9 unitary for the full Q-RPS circuit.
    U = J^† (S1 ⊗ S2) J
    """
    if J is None:
        J = J_gate_maximal_entanglement()
    U = J.conj().T @ (np.kron(S1, S2)) @ J
    return U

# -------------------------------------------------------------
# Reward function
# -------------------------------------------------------------

def expected_payoff(S1, S2, J=None):
    """Compute expected payoff for Player1 given strategies S1, S2.
    
    1. Build the full circuit U
    2. Apply to |00>
    3. Compute measurement probabilities
    4. Evaluate Player 1's (1st qtrit) payoff with RPS reward matrix
    """
    # Initial state |00> in 9-dim space
    ket00 = np.zeros((9, 1), dtype=complex)
    ket00[0, 0] = 1.0

    # Apply circuit
    U = qrps_circuit(S1, S2, J)
    final_state = U @ ket00

    # Measurement probabilities
    probs = np.abs(final_state.flatten())**2
    probs = probs.reshape((3, 3))  # shape (p1_choice, p2_choice)

    # RPS payoff matrix for player1
    payoff = np.array([[ 0, -1,  1],
                       [ 1,  0, -1],
                       [-1,  1,  0]])

    # Expected value of the payoff after measurement
    expected = np.sum(probs * payoff)
    return expected

# -------------------------------------------------------------
# Example usage
# -------------------------------------------------------------
if __name__ == "__main__":
    F = fourier_3()
    I = np.eye(3)

    # Both players use the balanced strategy (Fourier)
    print("Expected payoff with both players = F:", expected_payoff(F, F))

    # Player1 deviates to identity while player2 plays F
    print("Expected payoff with (I, F):", expected_payoff(I, F))
