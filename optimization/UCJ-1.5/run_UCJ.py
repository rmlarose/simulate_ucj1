"""
Run the polynomial-time UCJ backpropagation energy calculation
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

# Run 1.5 layer of UCJ with an unrestricted set of interaction pairs                
alpha_alpha_indices = lambda norb: None
alpha_beta_indices  = lambda norb: None

optimizer_method = "L-BFGS-B"
optimizer_options = {"maxiter": 10000, "gtol": 1e-9, "ftol": 1e-9}

# Since we have O(N^4) two-body terms, we might not be able to fit all of them in memory at once. This chunk size controls the batching of the 
# two-body terms, and should be set to a value that fits on the target GPU. These experiments were run on a GH200 with 96 GB of memory. 
optimizer_chunk_size = None  # set below to num_orb**2 once num_orb is known

fcidump_filename = Path(__file__).parent.parent / "fcidump_Fe4S4_MO.txt"
ucj_op_path = Path(__file__).parent / "ucj_initial.pkl"

# If we've already run HF/CCSD, and saved the UCJ operator, load it from disk. 
if ucj_op_path.exists():
    print(f"Loading cached HF/CCSD/UCJ data from {ucj_op_path}")
    with open(ucj_op_path, "rb") as f:
        cached = pickle.load(f)
    hf_energy = cached["hf_energy"]
    ccsd_energy = cached["ccsd_energy"]
    eccsd = cached["ccsd_corr_energy"]
    h1e = cached["h1e"]
    h2e = cached["h2e"]
    num_orb = cached["num_orb"]
    n_qubits = cached["n_qubits"]
    nelec = cached["nelec"]
    constant = cached["ecore"]
    base_op = cached["ucj_op"]
else:
    # Run Hartree-Fock.
    mf_as = pyscf.tools.fcidump.to_scf(fcidump_filename)
    mf_as.max_cycle = 100
    mf_as.conv_tol = 1e-9
    mf_as = mf_as.newton()
    mf_as.kernel()
    assert mf_as.converged, "SCF did not converge"

    # Run CCSD.
    ccsd = pyscf.cc.CCSD(mf_as)
    eccsd, *_ = ccsd.kernel()

    # Extract second-quantized Hamiltonian and Hamiltonian parameters.
    constant = pyscf.tools.fcidump.read(fcidump_filename).get("ECORE", 0.0)
    h1e = mf_as.get_hcore()
    num_orb = h1e.shape[0]
    n_qubits = 2 * num_orb
    h2e = pyscf.ao2mo.restore(1, mf_as._eri, num_orb)
    nelec = pyscf.tools.fcidump.read(fcidump_filename)["NELEC"]
    nelec = (nelec // 2, nelec // 2)  # Convert to (n_alpha, n_beta) tuple.

    # Construct the 1 1/2 layer operator
    _ = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=ccsd.t2, n_reps=2,
        interaction_pairs=(alpha_alpha_indices(num_orb), alpha_beta_indices(num_orb)),
    )

    base_op = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=_.diag_coulomb_mats[:1],
        orbital_rotations=_.orbital_rotations[:1],
        final_orbital_rotation=_.orbital_rotations[1].conj().T,
    )

    hf_energy = mf_as.e_tot
    ccsd_energy = ccsd.e_tot

    # Pickle the UCJ operator and energies to disk
    with open(ucj_op_path, "wb") as f:
        pickle.dump({
            "hf_energy": hf_energy,
            "ccsd_energy": ccsd_energy,
            "ccsd_corr_energy": eccsd,
            "h1e": h1e,
            "h2e": h2e,
            "num_orb": num_orb,
            "n_qubits": n_qubits,
            "nelec": nelec,
            "ecore": constant,
            "ucj_op": base_op,
        }, f)
    print(f"Saved HF/CCSD/UCJ data to {ucj_op_path}")

print(f"Number of spatial orbitals: {num_orb}, Number of qubits: {n_qubits}")
print(f"Hartree-Fock energy: {hf_energy:.6f} Ha")
print(f"CCSD correlation energy: {eccsd:.6f} Ha")
print(f"CCSD total energy: {ccsd_energy:.6f} Ha")

if optimizer_chunk_size is None:
    optimizer_chunk_size = num_orb ** 3 # This is the chunking that fits on a GH200

backprop = UCJBackPropagator(base_op, nelec=nelec, num_orb=num_orb, h1e=h1e, h2e=h2e, ecore=constant)

# CCSD-parameterized UCJ energy, before variational optimization.
ucj_ccsd_energy = backprop.propagate()

print(f"CCSD-parameterized UCJ energy: {ucj_ccsd_energy:.6f} Ha")

# Restart from a checkpoint if it exists, otherwise start from the CCSD parameters
checkpoint_path = Path(__file__).parent / "UCJ_checkpoint.npz"
optimize_start = time.perf_counter()
result = backprop.optimize_jax(
    interaction_pairs=(alpha_alpha_indices(num_orb), alpha_beta_indices(num_orb)),
    chunk_size=optimizer_chunk_size,
    method=optimizer_method,
    options=optimizer_options,
    checkpoint_path=checkpoint_path,
)
optimize_time = time.perf_counter() - optimize_start

print(f"Optimizer iterations: {result.nit}, function evals: {result.nfev}, gradient evals: {result.get('njev', result.nfev)}")
print(f"Optimization wall time: {optimize_time:.2f} s")

# Optimizing the ansatz updates the UCJ parameters in place, so we can just call propagate() again with those parameters 
# to get the variationally optimized energy.
ucj_optimized_energy = backprop.propagate(show_progress=True)

print(f"Hartree-Fock energy: {hf_energy:.6f} Ha")
print(f"CCSD energy: {ccsd_energy:.6f} Ha")
print(f"CCSD-parameterized UCJ energy: {ucj_ccsd_energy:.6f} Ha")
print(f"Variationally optimized UCJ energy: {ucj_optimized_energy:.6f} Ha")

out_path = Path(__file__).parent / "UCJ_results.npz"
np.savez(
    out_path,
    hf_energy=hf_energy,
    ccsd_energy=ccsd_energy,
    ucj_ccsd_energy=ucj_ccsd_energy,
    ucj_optimized_energy=ucj_optimized_energy,
    optimizer_nit=result.nit,
    optimizer_nfev=result.nfev,
    optimizer_njev=result.get("njev", result.nfev),
    optimize_time_seconds=optimize_time,
)
print(f"Saved results to {out_path}")
