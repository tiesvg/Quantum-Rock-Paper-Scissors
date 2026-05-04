import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D 
from helper_functions import calculate_strategy_matrix, calculate_entagling_matrix, probability_distribution, expected_payoffs
from helper_functions import basis, payoff_matrix

n = 20
gamma = 0
player = "A"
theta_a_array = np.linspace(0, np.pi, n)
theta_b_array = np.linspace(0, np.pi, n)

TA, TB = np.meshgrid(theta_a_array, theta_b_array, indexing='xy')

phi_array = np.linspace(0, np.pi/2, n)

def compute_surface(phi_a, phi_b, gamma=0, player="A"):
    Z = np.empty(shape=(n,n))
    
    for i in range(n):
        theta_a = theta_a_array[i]
        for j in range(n):
            theta_b = theta_b_array[j]
            A, B = expected_payoffs(
                probability_distribution(
                    strategy_A = calculate_strategy_matrix(theta_a, phi_a),
                    strategy_B = calculate_strategy_matrix(theta_b, phi_b),
                    initial_state = basis[0],
                    J=calculate_entagling_matrix(gamma=gamma),
                    basis = basis),
                payoff_matrix)
            Z[i, j] = A if player.upper() == 'A' else B
    return Z

Z0 = compute_surface(0, 0, gamma = gamma, player = "A")

plt.close('all')
fig = plt.figure(figsize = (9, 7))
ax = fig.add_subplot(111, projection = '3d')

surf = ax.plot_surface(TA, TB, Z0, rstride = 1, cstride = 1, linewidth = 0, antialiased = True)

ax.set_xlabel(r'$\theta_A$')
ax.set_ylabel(r'$\theta_B$')
ax.set_xlim(0,np.pi)
ax.set_ylim(np.pi,0)
ax.set_zlabel(f"Payoff {player}")
ax.set_title(rf"Quantum PD payoff surface for player {player}   ($\gamma={gamma:.2f}$)")

plt.subplots_adjust(bottom = 0.18)

plt.show()