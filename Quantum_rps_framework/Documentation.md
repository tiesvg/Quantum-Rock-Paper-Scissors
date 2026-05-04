# Documentation for classes.py

## Basis Class

Class that contains vectors that span the Hilbert Space for the moves/strategies for a single player.
- The basis represents states |R>, |P>, |S>. 
- The space is 3-dimensional, thus they states can be seen as representing a qutrit.


Initialising creates an instance of the class, e.g. "BASIS_OBJECT = Basis()"
- This instance IS NOT THE BASIS ITSELF as an ordered list, rather it is an object that contains the individual vectors and basis as attributes.
- One can acces a single vector either trough (a) attributes ("ROCK = BASIS_OBJECT.rock") or (b) indexing (see __getitem__ method below).
- To retrieve the basis as ordered list, simply use the .basis method, e.g. "BASIS_LIST = BASIS_OBJECT.basis".


Performing a simple call on an instance of the class does return the basis as an ordered list. 
- Let "BASIS_OBJECT = Basis()", then BASIS_OBJECT is an instance of the class.
- If you call this instance, it returns the basis as an ordered list, e.g. BASIS_OBJECT() -> BASIS_LIST 
- Thus one could also immidiately write "BASIS_LIST = TensorProductBasis()()"


Note on the __getitem__ method:
- Elements can be called either by string or by integer index,
  thus both "BASIS_OBJECT["rock"]" and "BASIS_OBJECT[0]" return the vector corresponding to state |R>, e.g. [1,0,0].

## TensorProductBasis Class

Class that contains vectors that span the Tensor Product Space for the combined moves/strategies of both players.
- The basis has vectors of length 9 that represent states |RR>, |RP>, |RS>, |PR>, |PP>, |PS>, |SR>, |SP>, |SS>. 
- As all vectors are kronecker products of the single player basis vectors, the space is 9-dimensional.


Initialising creates an instance of the class, e.g. "TENSORPRODUCTVECTORS = TensorProductVectors()"
- This instance IS NOT THE BASIS ITSELF as an ordered list, rather it is an object that contains the individual vectors and basis as attributes.
- One can acces a single vector either trough (a) attributes ("ROCK_ROCK = TENSORPRODUCTVECTORS.rock_rock") or (b) indexing (see __getitem__ method below).
- To retrieve the basis as ordered list, simply use the .basis method, e.g. "TENSORBASIS = TENSORPRODUCTVECTORS.basis".


Performing a simple call on an instance of the class does return the basis as an ordered list. 
- Let "TENSOR_BASIS_OBJECT = TensorProductBasis()", then TENSOR_BASIS_OBJECT is an instance of the class.
- If you call this instance, it returns the basis as an ordered list, e.g. TENSOR_BASIS_OBJECT() -> TENSOR_BASIS_LIST 
- Thus one could also immidiately write "TENSOR_BASIS_LIST = TensorProductBasis()()"


Note on the __getitem__ method:
- Elements can be called either by string or by integer index,
  thus both "TENSORPRODUCTVECTORS["rock_rock"]" and "TENSORPRODUCTVECTORS[0]" return the vector corresponding to state |RR>, e.g. [1,0,0,0,0,0,0,0,0].

## PayoffMatrix Class

Class that contains the payoff matrix for a game of rock-paper-scissors.


Initialising creates an instance of the class, e.g. "PAYOFF = PayoffMatrix()"
- This instance IS NOT THE PAYOFF MATRIX ITSELF as a 2D array, rather it is an object that contains the matrix as an attribute.
- To retrieve the matrix itself, simply use the .matrix method.


Performing a simple call on an instance of the class does return the payoff matrix as a 2D array. 
- Let "PAYOFF = PayoffMatrix()", then PAYOFF is an instance of the class.
- If you call this instance, it returns the payoff matrix as a 2D array, e.g. PAYOFF() -> PAYOFF_MATRIX 
- Thus one could also immidiately write "PAYOFF_MATRIX = PayoffMatrix()()"


Note on the __getitem__ method:
- Elements can be called either by string or by integer index, thus both "PAYOFF[("rock","paper")]" and "PAYOFF[(0,1)]" 
  return the payoff for player A when player A plays rock and player B paper, e.g. -1.
- By default, the method returns a tuple with the payoffs for both players.
- If "both" is set to False, only the payoff for player A is returned.

## StrategyMatrix Class

Class that contains the family of strategy matrices for a quantum game of rock-paper-scissors.
- The matrices are parameterised by two angles alpha and beta.
- The matrices are 3x3 unitary matrices.


Initialising creates an instance of the class, e.g. "U = StrategyMatrix()"
- This instance IS NOT A STRATEGY MATRIX ITSELF, rather an object containing the family of matrices as a method.


To get a specific strategy matrix, use indexing on an instance of the class.
- Indexing requires a tuple of two angles (alpha, beta), e.g. "U[(0, 2*np.pi/3)]" returns the matrix for alpha = 0 and beta = 2*pi/3.

## EntanglementMatrix Class

Class that contains the entanglement matrix for a quantum game of rock-paper-scissors.
- The matrix is parameterised by a single angle gamma, that determines the degree of entanglement.
- For a non-entangled game, gamma = 0 and the matrix is the identity.


A simple call on an instance of the class returns the entanglement matrix itself.

---

# Documentation for helper_functions.py

## Global Variables

**angles_classical_strategies**
Dictionary that maps classical rock-paper-scissors strategies to their corresponding quantum strategy angles (alpha, beta).
- "rock": (0, 0)
- "paper": (2π/3, 4π/3) 
- "scissors": (4π/3, 2π/3)

## Helper Functions

**param_space()**
Creates a standard parameter space for quantum strategies.
- Returns a dictionary with alpha and beta angle arrays for both players A and B
- Each array contains 50 linearly spaced values from 0 to 2π
- Used as input for plotting functions and parameter sweeps

**combined_state(state_A, state_B)**
Combines two single-player quantum states into a tensor product state.
- Takes two numpy arrays representing individual player states
- Returns the Kronecker product of the two states
- Essential for creating composite quantum states in the game

**probability_distribution(strategy_A, strategy_B, J, initial_state)**
Calculates the probability distribution of measurement outcomes for a quantum game.
- strategy_A, strategy_B: Strategy matrices for players A and B
- J: Entanglement matrix
- initial_state: Initial quantum state (defaults to first tensor product basis vector)
- Returns a Distribution object containing probabilities for each possible outcome
- Applies the quantum evolution: J† @ (strategy_A ⊗ strategy_B) @ J @ initial_state

**expected_payoff(probabilities, payoff_matrix, player)**
Calculates the expected payoff for a given player from a probability distribution.
- probabilities: Distribution object containing outcome probabilities
- payoff_matrix: PayoffMatrix object defining game payoffs
- player: Either "A" or "B" to specify which player's payoff to calculate
- Returns the expected payoff value weighted by outcome probabilities

## Plotting Functions

**plot_reward_surface_one_player_fixed(param_space, fixed_play, strategy_matrix_family, J)**
Creates a 3D surface plot of expected payoffs when one player uses a fixed classical strategy.
- param_space: Tuple of (alpha_array, beta_array) for the variable player
- fixed_play: Tuple of (player, strategy) e.g., ("B", "rock")
- strategy_matrix_family: StrategyMatrix object for generating strategies
- J: Entanglement matrix
- Returns a Plotly Figure with 3D surface visualization
- Shows how the active player's payoff varies with their strategy parameters

**plot_reward_surface_one_parameter_fixed(param_space, fixed, strategy_matrix_family, J)**
Creates a 3D surface plot when one strategy parameter (alpha or beta) is fixed for both players.
- param_space: Tuple of angle arrays for both players
- fixed: Tuple of (parameter_name, value) e.g., ("alpha", π/2)
- strategy_matrix_family: StrategyMatrix object
- J: Entanglement matrix
- Returns a Plotly Figure showing Player A's payoff surface
- Useful for analyzing the effect of varying one parameter while keeping another constant

**add_points(fig, points, colours)**
Adds scatter points to an existing 3D plot.
- fig: Plotly Figure object to modify
- points: Dictionary mapping point names to (x, y, z) coordinates
- colours: List of colors to cycle through for different points
- Returns the modified figure with added marker points
- Useful for highlighting specific strategies or equilibria on payoff surfaces

## Usage Patterns

These helper functions work together to analyze quantum rock-paper-scissors games:

1. **Setup**: Use param_space() to define strategy parameter ranges
2. **State Evolution**: Use combined_state() and probability_distribution() to calculate quantum evolution
3. **Analysis**: Use expected_payoff() to evaluate strategy performance  
4. **Visualization**: Use plotting functions to visualize payoff landscapes
5. **Enhancement**: Use add_points() to highlight important strategies or equilibria