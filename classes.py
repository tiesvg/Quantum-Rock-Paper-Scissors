import numpy as np
from typing import TypeVar

Idx = TypeVar("Idx", int, str)

class Basis:
    def __init__(self):
        self.rock = np.array([1,0,0])
        self.paper = np.array([0,1,0])
        self.scissors = np.array([0,0,1])

        self.str_to_idx = {"rock": 0, "paper": 1, "scissors": 2}
        
        self.basis = [self.rock, self.paper, self.scissors]
    
    def __call__(self):
        return self.basis
    
    def __getitem__(self, key: str | int) -> np.ndarray:
        if not(isinstance(key, str)) and not(isinstance(key, int)):
            raise TypeError("Index must be either str or int")
        
        if isinstance(key, str):
            if key not in self.str_to_idx.keys():
                raise KeyError(f"String index must be one of {list(self.str_to_idx.keys())}")
            return self.basis[self.str_to_idx[key]]
        
        if isinstance(key, int):
            if key not in range(len(self.basis) - 1):
                raise IndexError(f"Integer index must can't be larger than dimension ( = {len(self.basis)}) of basis: max allowed index is {len(self.basis)-1}")
            return self.basis[key]

class TensorProductBasis:
    def __init__(self):
        self.rock_rock = np.kron(np.array([1,0,0]), np.array([1,0,0]))
        self.rock_paper = np.kron(np.array([1,0,0]), np.array([0,1,0]))
        self.rock_scissors = np.kron(np.array([1,0,0]), np.array([0,0,1]))
        
        self.paper_rock = np.kron(np.array([0,1,0]), np.array([1,0,0]))
        self.paper_paper = np.kron(np.array([0,1,0]), np.array([0,1,0]))
        self.paper_scissors = np.kron(np.array([0,1,0]), np.array([0,0,1]))
        
        self.scissors_rock = np.kron(np.array([0,0,1]), np.array([1,0,0]))
        self.scissors_paper = np.kron(np.array([0,0,1]), np.array([0,1,0]))
        self.scissors_scissors = np.kron(np.array([0,0,1]), np.array([0,0,1]))
        
        self.str_to_idx = {"rock_rock": 0, "rock_paper": 1, "rock_scissors": 2,
                           "paper_rock": 3,"paper_paper": 4, "paper_scissors": 5,
                           "scissors_rock": 6, "scissors_paper": 7, "scissors_scissors": 8}

        self.basis = [self.rock_rock, self.rock_paper, self.rock_scissors,
                      self.paper_rock, self.paper_paper, self.paper_scissors,
                      self.scissors_rock, self.scissors_paper, self.scissors_scissors]
    
    def __call__(self):
        return self.basis
    
    def __getitem__(self, key: str | int) -> np.ndarray:
        if not(isinstance(key, str)) and not(isinstance(key, int)):
            raise TypeError("Index must be either str or int")
        
        if isinstance(key, str):
            if key not in self.str_to_idx.keys():
                raise KeyError(f"String index must be one of {list(self.str_to_idx.keys())}")
            return self.basis[self.str_to_idx[key]]
        
        if isinstance(key, int):
            if key not in range(len(self.basis) - 1):
                raise IndexError(f"Integer index must can't be larger than dimension ( = {len(self.basis)}) of basis: max allowed index is {len(self.basis)-1}")
            return self.basis[key]

class PayoffMatrix:
    def __init__(self):
        self.matrix = np.array([[0, -1, 1],
                                [1, 0, -1],
                                [-1, 1, 0]])
        
        self.str_to_idx = {"rock": 0, "paper": 1, "scissors": 2}
    
    def __call__(self):
        return self.matrix
    
    def __getitem__(self, key: tuple[Idx, Idx], both = True) -> tuple[float, float] | float | None:
        if not(isinstance(any(key), str)) and not(isinstance(any(key), int)):
            raise TypeError(f"Indices must be either int or str")
        
        if not(isinstance(key, tuple)) or len(key) != 2:
            raise KeyError("Index must be a pair such as (0, 1) or ('rock','paper')")
        
        if type(key[0]) != type(key[1]):
            raise KeyError("Both indices must be of the same type, either int or str")
        
        if isinstance(key[0], str) and isinstance(key[1], str):
            pay_off_A = self.matrix[self.str_to_idx[key[0]], self.str_to_idx[key[1]]]
            
            if both: return pay_off_A, -pay_off_A
            else: return pay_off_A
        
        elif isinstance(key[0], int) and isinstance(key[1], int):
            pay_off_A = self.matrix[key[0], key[1]]
            
            if both: return pay_off_A, -pay_off_A
            else: return pay_off_A
                
class StrategyMatrix:
    def __init__(self):
        self.omega = np.exp(2j * np.pi / 3)
   
    def __getitem__(self, key: tuple[float, float]) -> np.ndarray:
        alpha, beta = key
        U = np.array([[1 + np.exp(1j*alpha) + np.exp(1j*beta), 1 + self.omega * np.exp(1j*alpha) + self.omega**2 * np.exp(1j*beta), 1 + self.omega**2 * np.exp(1j*alpha) + self.omega * np.exp(1j*beta)],
                      [1 + self.omega**2 * np.exp(1j*alpha) + self.omega * np.exp(1j*beta), 1 + np.exp(1j*alpha) + np.exp(1j*beta), 1 + self.omega * np.exp(1j*alpha) + self.omega**2 * np.exp(1j*beta)],
                      [1 + self.omega * np.exp(1j*alpha) + self.omega**2 * np.exp(1j*beta), 1 + self.omega**2 * np.exp(1j*alpha) + self.omega * np.exp(1j*beta), 1 + np.exp(1j*alpha) + np.exp(1j*beta)]])
        return U / 3

class EntanglementMatrix:
    def __init__(self, gamma = 0):
        self.gamma = gamma
        if gamma == 0: 
            self.matrix = np.identity(9)
    
    def __call__(self):
        return self.matrix

class Distribution:
    def __init__(self, dim):    
        self.probs = np.zeros(dim)
        self.str_to_idx = {}    
