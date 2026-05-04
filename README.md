# Quantum Rock Paper Scissors with Superposition Moves

An experiment in extending classical Rock Paper Scissors with "superposition" moves — weighted combinations of the three base moves treated as new actions — and using gradient descent to ask whether a different optimal strategy emerges.

## Background

Standard Rock Paper Scissors has a clean game-theoretic answer: the unique Nash equilibrium is the uniform mixed strategy `(⅓, ⅓, ⅓)`. This project widens the action space and re-runs the question. If players can choose moves that are themselves convex combinations of Rock, Paper, and Scissors, does the strategic landscape change, or does optimization fall back to the classical equilibrium?

## What's in here

- A payoff matrix for an extended RPS game where the action set includes both the pure moves and a configurable set of superposition moves.
- A gradient descent loop that optimizes a player's mixed strategy over this extended action set against an opponent strategy.
- Tools for inspecting how the optimization converges and what distribution it lands on.
