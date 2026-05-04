import numpy as np
import plotly.graph_objects as go
from circuit import expected_payoff

# === Example restricted strategy families ===
def phase_strategy(theta):
    """
    Strategy: diagonal phase matrix in computational basis.
    param theta: single real parameter (phase for |1>)
    """
    return np.diag([1, np.exp(1j * theta), 1])

def rotation_strategy(theta):
    """
    Strategy: rotation in the (|0>, |1>) subspace.
    Leaves |2> unchanged.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ], dtype=complex)

# === Visualization function ===
def plot_reward_surface(S1_family, S2_family, J, param_range=(0, np.pi), steps=50):
    """
    Plots the reward surface for restricted strategy families in 3D.

    S1_family: function(theta1) -> 3x3 unitary
    S2_family: function(theta2) -> 3x3 unitary
    J: entangling matrix (9x9)
    param_range: range of parameter values (for both players)
    steps: resolution of grid
    invert_player2: if True, flips Player 2's parameter axis so Nash
                    equilibria appear more like extrema.
    """
    theta_vals = np.linspace(param_range[0], param_range[1], steps)
    payoff_matrix = np.zeros((steps, steps))

    for i, theta1 in enumerate(theta_vals):
        for j, theta2 in enumerate(theta_vals):
            S1 = S1_family(theta1)
            S2 = S2_family(theta2)
            payoff_matrix[i, j] = expected_payoff(S1, S2, J=J).real

    # Create 3D surface
    X, Y = np.meshgrid(theta_vals, theta_vals)
    Z = payoff_matrix

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="RdBu")])
    fig.update_layout(
        title="Expected Payoff Surface (Quantum RPS)",
        scene=dict(
            xaxis_title="Player 2 parameter",
            yaxis_title="Player 1 parameter",
            zaxis_title="Player 1 Expected Payoff",
        )
    )
    return fig


def compute_nsi_landscape(S1_family, S2_family, J, param_range=(0, np.pi), steps=50, h=1e-3):
    """
    Computes NSI (Nash Stability Index) values over the area of Player 1 and Player 2's free parameters.
    The NSI measure is based on Ratliff et al. (2016) and combines the first derivatives (gradient) with the
    second derivatives (Hessian) of the payoff function in relation to the players' strategies parameters.

    The necessary conditions for a point x in the continuous parameter space to be a Nash Equilibrium are:

    1. Stationarity: 
        For each player i, the gradient of their payoff with respect to their own parameter must vanish ->
        df_i / dtheta_i = 0.

    2. Second-order conditions:
        For each player i, the second derivative of their payoff with respect to their own parameter must
        be nonnegative. This ensures each player is at a local minimum of their payoff function.

    The NSI value is constructed as: NSI(theta1, theta2) = ||gradient|| + penalty(second_derivatives).
    NSI value being 0 at some point is a necessary condition for it to be a Nash Equilibrium.

    We approximate values of the derivatives with final differences:
    https://en.wikipedia.org/wiki/Finite_difference
    
    Args:
        S1_family, S2_family: functions mapping a scalar -> 3x3 unitary
        J: entangling gate (9x9 matrix)
        param_range: (min, max) range for both theta1 and theta2
        steps: grid resolution
        h: step size for finite difference
    
    Returns:
        X, Y, G: meshgrid arrays with computed values
    """
    theta_vals = np.linspace(param_range[0], param_range[1], steps)
    G = np.zeros((steps, steps))

    for i, theta1 in enumerate(theta_vals):
        for j, theta2 in enumerate(theta_vals):
            # Value of the payoff function at exact coordinates
            R = expected_payoff(S1_family(theta1), S2_family(theta2), J=J).real

            # Player 1's axis of freedom
            R_t1_plus = expected_payoff(S1_family(theta1 + h), S2_family(theta2), J=J).real
            R_t1_minus = expected_payoff(S1_family(theta1 - h), S2_family(theta2), J=J).real
            # Gradient of the payoff function in relation to Player 1's parameter choice: dR/dTheta_1
            g_1 = (R_t1_plus - R_t1_minus) / (2*h)
            # Curvature (Hessian): d/dTheta_1 * dR/dTheta_1
            H_11 = (R_t1_plus - 2*R + R_t1_minus) / (h*h)

            # Player 2's axis of freedom
            R_t2_plus = expected_payoff(S1_family(theta1), S2_family(theta2 + h), J=J).real
            R_t2_minus = expected_payoff(S1_family(theta1), S2_family(theta2 - h), J=J).real
            # Gradient of the payoff function in relation to Player 2's parameter choice: dR/dTheta_2
            g_2 = (R_t2_plus - R_t2_minus) / (2*h)
            # Curvature (Hessian): d/dTheta_2 * dR/dTheta_2
            H_22 = (R_t2_plus - 2*R + R_t2_minus) / (h*h)

            # Compute NSI (Nash Stability Index)
            balance_parameter = 1.0
            gradient_norm = np.sqrt(g_1 * g_1 + g_2 * g_2)
            curvature_penalty = np.maximum(0.0, H_11) + np.maximum(0.0, -H_22)

            G[i, j] = gradient_norm + balance_parameter * curvature_penalty

    X, Y = np.meshgrid(theta_vals, theta_vals)
    return X, Y, G


def plot_nsi_landscape(S1_family, S2_family, J, param_range=(0, np.pi), steps=50):
    """
    Plot the NSI landscape surface for Q-RPS.
    """
    X, Y, G = compute_nsi_landscape(S1_family, S2_family, J, param_range, steps)

    fig = go.Figure(data=[go.Surface(z=G, x=X, y=Y, colorscale="Viridis")])
    fig.update_layout(
        title="Nash Stability Index (NSI=0 => NE candidate)",
        scene=dict(
            xaxis_title="Player 2 parameter (θ2)",
            yaxis_title="Player 1 parameter (θ1)",
            zaxis_title="NSI"
        )
    )
    return fig

# === Example usage ===
if __name__ == "__main__":
    J = np.eye(9)  # classical RPS case - no entanglement
    strategy = rotation_strategy # Players can assign probabilities to Rock and Paper

    # # Plots the raw payoff function
    # fig = plot_reward_surface(strategy, strategy, J, steps=60)

    # Plots the NSI landscape
    fig = plot_nsi_landscape(strategy, strategy, J, steps=60)

    fig.show()

