import numpy as np
from scipy.linalg import expm

payoff_matrix = {
    "CC": (3, 3),
    "DC": (5, 0),
    "CD": (0, 5),
    "DD": (1, 1)
}

state_space = {
    'co-operate': np.array([1,0]),
    'defect': np.array([0,1])
}

CC, CD, DC, DD = np.kron(state_space['co-operate'], state_space['co-operate']), \
                 np.kron(state_space['co-operate'], state_space['defect']), \
                 np.kron(state_space['defect'], state_space['co-operate']), \
                 np.kron(state_space['defect'], state_space['defect'])

basis = np.column_stack([CC, CD, DC, DD])

def calculate_strategy_matrix(theta, phi):
    U = np.array([[np.exp(1j*phi)*np.cos(theta/2), np.sin(theta/2)],
                    [-np.sin(theta/2), np.exp(-1j*phi)*np.cos(theta/2)]])
    return U


def calculate_entagling_matrix(gamma = 0) -> np.ndarray:
    defect_matrix = calculate_strategy_matrix(np.pi, 0)
    J = expm(1j * gamma * np.kron(defect_matrix, defect_matrix) / 2)
    return J

def calculate_non_quantum_tensor_operators():
    defect = calculate_strategy_matrix(np.pi, 0)
    cooperate = calculate_strategy_matrix(0, 0)

    tensor_products ={
    "CC": np.round(np.kron(cooperate, cooperate).real).astype(int),
    "DC": np.round(np.kron(defect, cooperate).real).astype(int),
    "CD": np.round(np.kron(cooperate, defect).real).astype(int),
    "DD": np.round(np.kron(defect, defect).real).astype(int)
    }

    return tensor_products

def probability_distribution(strategy_A, strategy_B, initial_state = basis[0], J = calculate_entagling_matrix(), basis = basis): 
    modified_state = J.conj().T @ np.kron(strategy_A, strategy_B) @ J @ initial_state
    probability_distribution = { 
    "CC": np.abs(np.dot(basis[0], modified_state))**2,
    "DC": np.abs(np.dot(basis[1], modified_state))**2,
    "CD": np.abs(np.dot(basis[2], modified_state))**2,
    "DD": np.abs(np.dot(basis[3], modified_state))**2}
    return probability_distribution

def expected_payoffs(probability_distribution, payoff_matrix):
    expected_payoff_A = sum([probability_distribution[outcome] * payoff_matrix[outcome][0] for outcome in payoff_matrix])
    expected_payoff_B = sum([probability_distribution[outcome] * payoff_matrix[outcome][1] for outcome in payoff_matrix])
    return expected_payoff_A, expected_payoff_B

