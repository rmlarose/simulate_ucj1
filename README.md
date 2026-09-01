# Simulate UCJ1

[![Run in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rmlarose/simulate_ucj1/blob/main/simulate_ucj1.ipynb)

Companion repository for https://arxiv.org/abs/2607.21337. Simulate depth one unitary cluster Jastrow (UCJ) circuits in polynomial time, notably the $n = 72$ qubit iron sulfur cluster UCJ circuit from [*Sci. Adv.* **11** 25 (2025)](https://www.science.org/doi/10.1126/sciadv.adu9991).

## Quickstart

1. Run [this notebook](./simulate_ucj1.ipynb) to compute the energy of the $n = 72$ qubit iron sulfur cluster UCJ circuit from [*Sci. Adv.* **11** 25 (2025)](https://www.science.org/doi/10.1126/sciadv.adu9991).
1. Run [this script](./simulate_ucj1_hchains.py) to compute the energy of hydrogen chains up to $n = 160$ qubits.

## Optimization 

The `optimization/` directory contains scripts and the results of optimization experiments. The `run_UCJ.py` scripts optimize the UCJ ansatz from a CCSD-initialized parameters `ucj_initial.pkl`, and saves the final results to `UCJ_results.npz`. We have also included the best UCJ operator found during our optimization runs, saved to `ucj_optimized.pkl`. The `opt_energy.py` scripts calculate the optimized UCJ energy from `ucj_optimized.pkl`.

This uses the `fermionic-backpropagation/` directory which contains code with a JAX implementation of the backpropagation algorithm, enabling the variational optimization of UCJ circuits. It can be installed via pip: 
```bash
pip install ./fermionic-backpropagation
```

## Benchmarking

[`simulate_ucj1_appendix_h.py`](./simulate_ucj1_appendix_h.py) contains an implementation of the Appendix H algorithm from https://arxiv.org/abs/2503.21041, and [`simulate_ucj1_appendix_h_cumulative.py`](./simulate_ucj1_appendix_h_cumulative.py) contains a modified version of this implementation that allows one to plot convergence vs. the number of samples.

## Plotting and data

The [plot/](./plot/) contains data and a script to reproduce plots.

## AI statement

We acknowledge the use of Clause Opus 5 and GPT 5.6 Sol for assistance with writing code in this repository.
