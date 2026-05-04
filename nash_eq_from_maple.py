import re
import numpy as np
from scipy.optimize import root

arg_names = ["a_A", "b_A", "a_B", "b_B"]

def clean_maple_output(input_path):
    with open(input_path, "r") as file:
        text = "".join(line.strip() for line in file)
    text = text.rstrip()
    
    text = re.sub(r"abs\s*\(\s*1\s*,\s*(.*?)\)", r"abs(\1)", text)
    text = text.replace("^", "**")
    text = re.sub(r"\bcos\b", "np.cos", text)
    text = re.sub(r"\bsin\b", "np.sin", text)
    text = re.sub(r"\bexp\b", "np.exp", text)
    text = re.sub(r"\bI\b", "1j", text)
    text = re.sub(r"\bpi\b", "np.pi", text)
    text = re.sub(r"\breal\b", "np.real", text)
    
    lambda_src = f"lambda {', '.join(arg_names)}: {text}"
    func = eval(lambda_src, {"np": np}, {})
    return func, text

Gra_A1 = clean_maple_output("maple_output/Gra_A1.txt")[0]
Gra_A2 = clean_maple_output("maple_output/Gra_A2.txt")[0]
Gra_B1 = clean_maple_output("maple_output/Gra_B1.txt")[0]
Gra_B2 = clean_maple_output("maple_output/Gra_B2.txt")[0]

omega = lambda s: np.real(np.array([Gra_A1(*s), 
                                    Gra_A2(*s), 
                                    Gra_B1(*s), 
                                    Gra_B2(*s)]))

guess = np.array([1/2,1/2,1/2,1/2])

nash_eq = root(omega, x0=guess, method="hybr")



