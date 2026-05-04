from classes import *
from helper_functions import *
import numpy as np

BASIS_OBJECT = Basis()
BASIS_LIST = BASIS_OBJECT.basis
TENSORBASIS = TensorProductBasis()
INITIAL_STATE = TENSORBASIS[0]

PAYOFF = PayoffMatrix()
U = StrategyMatrix()
J = EntanglementMatrix(0).matrix

xyz_points = dict.fromkeys(angles_classical_strategies)
fixed_play = ("B", "scissors")

for move, (x,y) in angles_classical_strategies.items():
    fixed_angles = angles_classical_strategies[fixed_play[1]]
    probs = probability_distribution(U[x,y], U[fixed_angles], J)
    z = expected_payoff(probs, PAYOFF)
    xyz_points[move] = (x,y,z)

standard_param_space = param_space()
param_space1 = standard_param_space["alpha_A"], standard_param_space["beta_A"]
param_space2 = standard_param_space["alpha_A"], standard_param_space["alpha_B"]

fig = plot_reward_surface_one_player_fixed(param_space1, fixed_play)
fig = add_points(fig, xyz_points)
fig.show()

