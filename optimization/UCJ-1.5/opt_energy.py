"""
Calculate the optimized UCJ energy from a pickled UCJ operator.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyscf
import ffsim

try:
    from fermionic_backpropagation import UCJBackPropagator
except ImportError:
    sys.exit("Could not import the required fermionic_backpropagation module. Please install it via `pip install ./fermionic-backpropagation`.")

ucj_op = Path(__file__).parent / "ucj_optimized.pkl"

with open(ucj_op, "rb") as f:
    cached = pickle.load(f)

backprop = UCJBackPropagator(
    cached["ucj_op"],
    nelec=cached["nelec"],
    num_orb=cached["num_orb"],
    h1e=cached["h1e"],
    h2e=cached["h2e"],
    ecore=cached["ecore"],
)
energy = backprop.propagate()
print(f"Optimized UCJ energy: {energy:.6f} Ha")