from classes import *
from numpy import kron, pi, diagflat
import pprint
import plotly.graph_objects as go

angles_classical_strategies = dict(rock = (0,0), paper = (2 * pi/3, 4 * pi/3), scissors = (4 * pi/3, 2 * pi/3))

def param_space():
    return {"alpha_A": np.linspace(0, 2*np.pi, 50),
            "beta_A": np.linspace(0, 2*np.pi, 50),
            "alpha_B": np.linspace(0, 2*np.pi, 50),
            "beta_B": np.linspace(0, 2*np.pi, 50)}

def combined_state(state_A, state_B):
    return kron(state_A, state_B)

def probability_distribution(strategy_A, strategy_B, J, initial_state = TensorProductBasis()[0]):
    probabilities = Distribution(len(TensorProductBasis().basis))
    probabilities.str_to_idx = TensorProductBasis().str_to_idx
    
    combined_strategy = kron(strategy_A, strategy_B)
    final_state = J.conj().T @ combined_strategy @ J @ initial_state
    
    for i, vec in enumerate(TensorProductBasis().basis):
        probabilities.probs[i] = np.abs(np.dot(vec, final_state))**2
        
    return probabilities

def expected_payoff(probabilities, payoff_matrix, player = "A"):
    expected_payoff_A = 0
    expected_payoff_B = 0
    
    for key, prob in zip(probabilities.str_to_idx.keys(), probabilities.probs):
        if prob < 1e-10:
            continue
        key = tuple(key.split("_"))
        
        expected_payoff_A += prob * payoff_matrix[key][0]
        expected_payoff_B += prob * payoff_matrix[key][1]
    
    if player == "A": return expected_payoff_A
    else: return expected_payoff_B

def plot_reward_surface_one_player_fixed(param_space: tuple[np.ndarray, np.ndarray], fixed_play: tuple[str, str] = ("B", "rock"), 
                                         stragegy_matrix_family = StrategyMatrix(), J = EntanglementMatrix(0).matrix):
    alpha_arr = param_space[0]
    beta_arr = param_space[1]

    fixed_player = fixed_play[0]
    fixed_angles = angles_classical_strategies[fixed_play[1]]
    
    if fixed_player == "A": active_player = "B"
    else: active_player = "A"
    
    X, Y = np.meshgrid(alpha_arr, beta_arr)
    Z = np.zeros((len(alpha_arr), len(beta_arr)))

    for i, alpha in enumerate(alpha_arr):
        for j, beta in enumerate(beta_arr):
            if fixed_player == "A":
                strategy_A = stragegy_matrix_family[fixed_angles]
                strategy_B = stragegy_matrix_family[alpha, beta]  
            
            elif fixed_player == "B":
                strategy_A = stragegy_matrix_family[alpha, beta]
                strategy_B = stragegy_matrix_family[fixed_angles]
            
            probs = probability_distribution(strategy_A, strategy_B, J)
            Z[j, i] = expected_payoff(probs, PayoffMatrix(), player = active_player)
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="RdBu", opacity = 0.5)])
    
    fig.update_layout(
        title = f"Expected Payoff Surface (Quantum RPS) for Player {active_player} while the strategy of Player {fixed_player} is fixed at playing {fixed_play[1]}",
        scene = dict(
            xaxis_title = f"Player {active_player} Alpha",
            yaxis_title = f"Player {active_player} Beta",
            zaxis_title = f"Player {active_player} Expected Payoff"))
    
    return fig

def plot_reward_surface_one_parameter_fixed(param_space: tuple[np.ndarray, np.ndarray], fixed: tuple[str, float], 
                                            stragegy_matrix_family = StrategyMatrix(), J = EntanglementMatrix(0).matrix):
    angles_A_arr = param_space[0]
    angles_B_arr = param_space[1]

    fixed_parameter = fixed[0]
    fixed_angle = fixed[1]

    X, Y = np.meshgrid(angles_A_arr, angles_B_arr)
    Z = np.zeros((len(angles_A_arr), len(angles_B_arr)))

    for i, angle_A in enumerate(angles_A_arr):
        for j, angle_B in enumerate(angles_B_arr):
            if fixed_parameter == "alpha":
                strategy_A = stragegy_matrix_family[fixed_angle, angle_A]
                strategy_B = stragegy_matrix_family[fixed_angle, angle_B]
                variable_parameter = "beta"
            elif fixed_parameter == "beta":
                strategy_A = stragegy_matrix_family[angle_A, fixed_angle]
                strategy_B = stragegy_matrix_family[angle_B, fixed_angle]
                variable_parameter = "alpha"
            probs = probability_distribution(strategy_A, strategy_B, J)
            Z[j, i] = expected_payoff(probs, PayoffMatrix())
        
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="RdBu", opacity = 0.5)])
    
    fig.update_layout(
        title = f"Expected Payoff Surface (Quantum RPS) for Player A against angles of {variable_parameter} of both players. Variable {fixed_parameter} is fixed at {fixed_angle}",
        scene = dict(
            xaxis_title = f"Player A {variable_parameter}",
            yaxis_title = f"Player B {variable_parameter}",
            zaxis_title = f"Player A Expected Payoff"))
    
    return fig

def add_points(fig, points, colours = ["red", "green", "blue", "orange", "purple", "black"]):
    i = 0
    for key, (x, y, z) in points.items():
        fig.add_trace(go.Scatter3d(x = [x], y = [y], z = [z], mode = 'markers',
                                   line = dict(color = colours[i % len(colours)], width = 2),
                                   name = key,
                                   showlegend = False))
        i += 1

    return fig

def cost_from_phase(s, player = "A"):
    alpha_A, alpha_B, beta_A, beta_B = s
    U = StrategyMatrix()
    J = EntanglementMatrix(0).matrix
    COST_A = np.diagflat([0,1,-1,-1,0,1,1,-1,0])
    COST_B = np.diagflat([0,-1,1,1,0,-1,-1,1,0])
    
    strategy_A = U[alpha_A, beta_A]
    strategy_B = U[alpha_B, beta_B]

    FINAL_STATE = J.conj().T @ kron(strategy_A, strategy_B) @ J @ TensorProductBasis().basis[0]
    f_A = FINAL_STATE.conj().T @ COST_A @ FINAL_STATE
    f_B = FINAL_STATE.conj().T @ COST_B @ FINAL_STATE
    if player == "A":
        return np.real(f_A)
    else:
        return np.real(f_B)