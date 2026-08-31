# Simulate UCJ1

[![Run in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rmlarose/simulate_ucj1/blob/main/simulate_ucj1.ipynb)

Simulate depth one unitary cluster Jastrow (UCJ) circuits in polynomial time, notably the $n = 72$ qubit iron sulfur cluster UCJ circuit from [*Sci. Adv.* **11** 25 (2025)](https://www.science.org/doi/10.1126/sciadv.adu9991).

# Variational optimization of UCJ anstaze
The `fermionic-backpropagation/` directory contains code with a JAX implementation of the backpropagation algorithm, enabling the variational optimization of UCJ circuits. It can be installed via pip: 
```bash
pip install ./fermionic-backpropagation
```

## Experiments 
The `experiments/` directory contains scripts and the results of experiments from https://arxiv.org/abs/2607.21337, namely the variational optimization of UCJ ansatze for the $n = 72$ qubit iron sulfur cluster, with 1 and 1.5 layers:
```bash 
.
├── H-chains
│   ├── n12
│   │   ├── UCJ_checkpoint.npz
│   │   ├── UCJ_results.npz
│   │   └── ucj_initial.pkl
│   ├── n16
│   │   ├── UCJ_checkpoint.npz
│   │   ├── UCJ_results.npz
│   │   └── ucj_initial.pkl
│   ├── n20
│   │   ├── UCJ_checkpoint.npz
│   │   ├── UCJ_results.npz
│   │   └── ucj_initial.pkl
│   ├── n4
│   │   ├── UCJ_checkpoint.npz
│   │   ├── UCJ_results.npz
│   │   └── ucj_initial.pkl
│   ├── n8
│   │   ├── UCJ_checkpoint.npz
│   │   ├── UCJ_results.npz
│   │   └── ucj_initial.pkl
│   └── run_UCJ.py
├── UCJ-1
│   ├── UCJ_checkpoint.npz
│   ├── UCJ_results.npz
│   ├── opt_energy.py
│   ├── run_UCJ.py
│   ├── ucj_initial.pkl
│   └── ucj_optimized.pkl
├── UCJ-1.5
│   ├── UCJ_checkpoint.npz
│   ├── UCJ_results.npz
│   ├── opt_energy.py
│   ├── run_UCJ.py
│   ├── ucj_initial.pkl
│   └── ucj_optimized.pkl
└── fcidump_Fe4S4_MO.txt # FCIDUMP file for the $n = 72$ qubit iron sulfur cluster, from https://github.com/jrm874/sqd_data_repository
```

The `run_UCJ.py` scripts optimize the UCJ ansatz from a CCSD-initialized parameters `ucj_initial.pkl`, and saves the final results to `UCJ_results.npz`. We have also included the best UCJ operator found during our optimization runs, saved to `ucj_optimized.pkl`. The `opt_energy.py` scripts calculate the optimized UCJ energy from `ucj_optimized.pkl`.