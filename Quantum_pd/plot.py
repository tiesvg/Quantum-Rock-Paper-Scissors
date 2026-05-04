import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D 
from helper_functions import calculate_strategy_matrix, calculate_entagling_matrix, probability_distribution, expected_payoffs
from helper_functions import basis, payoff_matrix

n = 20
gamma = np.pi / 2
player = "A"
theta_a_array = np.linspace(0, np.pi, n)
theta_b_array = np.linspace(0, np.pi, n)
TA, TB = np.meshgrid(theta_a_array, theta_b_array, indexing='xy')

phi_array = np.linspace(0, np.pi/2, n)

def compute_surface(phi_a, phi_b, gamma=0, player="A"):
    Z = np.empty_like(TA)

    for i in range(TA.shape[0]):
        for j in range(TA.shape[1]):
            theta_a, theta_b = TA[i, j], TB[i, j]
            
            A, B = expected_payoffs(
                probability_distribution(
                    strategy_A = calculate_strategy_matrix(theta_a, phi_a),
                    strategy_B = calculate_strategy_matrix(theta_b, phi_b),
                    J=calculate_entagling_matrix(gamma=gamma)),
                payoff_matrix)
            Z[i, j] = A if player.upper() == 'A' else B
    return Z

Z0 = compute_surface(0, 0, gamma = gamma, player = "A")

plt.close('all')
fig = plt.figure(figsize = (9, 7))
ax = fig.add_subplot(111, projection = '3d')

surf = ax.plot_surface(TA, TB, Z0.T, rstride = 1, cstride = 1, linewidth = 0, antialiased = True)
surf = ax.plot_surface(TA, TB, Z0, rstride = 1, cstride = 1, linewidth = 0, antialiased = True)

ax.set_xlabel(r'$\theta_A$')
ax.set_ylabel(r'$\theta_B$')
ax.set_zlabel(f"Payoff {player}")
ax.set_title(rf"Quantum PD payoff surface for player {player}   ($\gamma={gamma:.2f}$)")

# plt.subplots_adjust(bottom = 0.18)

ax_phi_a = plt.axes([0.15, 0.08, 0.7, 0.03])  
ax_phi_b = plt.axes([0.15, 0.03, 0.7, 0.03])

s_phi_a = Slider(ax=ax_phi_a, label = r'$\phi_A$', valmin = phi_array[0], valmax = phi_array[-1], valinit = 0)
s_phi_b = Slider(ax=ax_phi_b, label = r'$\phi_B$', valmin = phi_array[0], valmax = phi_array[-1], valinit = 0)
print(s_phi_a.val, s_phi_b.val)

def update(_):
    global surf

    phi_a = float(s_phi_a.val)
    phi_b = float(s_phi_b.val)
    Z = compute_surface(phi_a, phi_b, gamma = gamma, player = player)

    surf.remove()
    surf = ax.plot_surface(TA, TB, Z, rstride = 1, cstride = 1, linewidth = 0, antialiased=True)

    ax.set_zlim(np.nanmin(Z), np.nanmax(Z))

    fig.canvas.draw_idle()

s_phi_a.on_changed(update)
s_phi_b.on_changed(update)

plt.show()