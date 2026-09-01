"""Run the polynomial-time UCJ backpropagation energy estimate"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyscf
import pyscf.cc
import ffsim

try:
    from fermiprop import UCJBackPropagator
except ImportError:
    sys.exit("Could not import the required fermiprop module. Please install it via `pip install ./fermionic-backpropagation`.")

# Molecule parameters.
atom = "H"
natoms_values = [int(n) for n in np.arange(4, 21, 4)]
atomic_distance = 0.774 

# Parameters of the (L)UCJ ansatz.
half_layer = False                       # If True, appends a final rotation to the circuit
alpha_alpha_indices = lambda norb: None  
alpha_beta_indices  = lambda norb: None 

# Variational optimization settings
optimizer_method = "L-BFGS-B"
optimizer_options = {"maxiter": 10000, "gtol": 1e-9, "ftol": 1e-9}
optimizer_chunk_size = None
n_timing_trials = 10


def generate_linear_geometry(atom: str, natoms: int, atomic_distance: float = 1.0) -> str:
    return "; ".join([f"{atom} 0 0 {i * atomic_distance}" for i in range(natoms)])


for natoms in natoms_values:
    results_dir = Path(f"n{natoms}")
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "UCJ_results.npz"
    if results_path.exists():
        print(f"Skipping natoms={natoms}: {results_path} already exists")
        continue

    ucj_op_path = results_dir / "ucj_initial.pkl"

    # If we've already run HF/CCSD and built the UCJ operator for this chain
    # length, load it from disk instead of recomputing.
    if ucj_op_path.exists():
        print(f"Loading cached HF/CCSD/UCJ data from {ucj_op_path}")
        with open(ucj_op_path, "rb") as f:
            cached = pickle.load(f)
        hf_energy = cached["hf_energy"]
        ccsd_energy = cached["ccsd_energy"]
        ccsd_runtime = cached.get("ccsd_runtime")
        h1e = cached["h1e"]
        h2e = cached["h2e"]
        num_orb = cached["num_orb"]
        n_qubits = cached["n_qubits"]
        nelec = cached["nelec"]
        constant = cached["ecore"]
        ucj_op = cached["ucj_op"]
    else:
        mol = pyscf.gto.Mole()
        mol.build(
            atom=generate_linear_geometry(atom, natoms, atomic_distance),
            basis="sto-6g",
        )

        n_frozen = 0
        active_space = range(n_frozen, mol.nao_nr())

        scf = pyscf.scf.RHF(mol).run()

        n_electrons = int(sum(scf.mo_occ[active_space]))
        n_alpha = (n_electrons + mol.spin) // 2
        n_beta = (n_electrons - mol.spin) // 2
        nelec = (n_alpha, n_beta)

        mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
        num_orb = mol_data.norb
        n_qubits = 2 * num_orb
        h1e = mol_data.one_body_integrals
        h2e = pyscf.ao2mo.restore(1, mol_data.two_body_integrals, num_orb)
        constant = mol_data.core_energy

        # Run CCSD.
        t_start = time.perf_counter()
        ccsd = pyscf.cc.CCSD(scf)
        ccsd.max_cycle = 200
        eccsd, *_ = ccsd.kernel()
        ccsd_runtime = time.perf_counter() - t_start
        assert ccsd.converged, "CCSD did not converge"

        base_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=ccsd.t2, n_reps=(2 if half_layer else 1),
            interaction_pairs=(alpha_alpha_indices(num_orb), alpha_beta_indices(num_orb)),
        )
        if half_layer:
            ucj_op = ffsim.UCJOpSpinBalanced(
                diag_coulomb_mats=base_op.diag_coulomb_mats[:1],
                orbital_rotations=base_op.orbital_rotations[:1],
                final_orbital_rotation=base_op.orbital_rotations[1].conj().T,
            )
        else:
            ucj_op = base_op

        hf_energy = scf.e_tot
        ccsd_energy = ccsd.e_tot

        with open(ucj_op_path, "wb") as f:
            pickle.dump({
                "hf_energy": hf_energy,
                "ccsd_energy": ccsd_energy,
                "ccsd_runtime": ccsd_runtime,
                "h1e": h1e,
                "h2e": h2e,
                "num_orb": num_orb,
                "n_qubits": n_qubits,
                "nelec": nelec,
                "ecore": constant,
                "ucj_op": ucj_op,
            }, f)
        print(f"Saved HF/CCSD/UCJ data to {ucj_op_path}")

    chunk_size = optimizer_chunk_size if optimizer_chunk_size is not None else num_orb ** 3
    num_parameters = ucj_op.to_parameters(
        interaction_pairs=(alpha_alpha_indices(num_orb), alpha_beta_indices(num_orb))
    ).size

    print(f"natoms={natoms}, atomic distance: {atomic_distance} Angstrom")
    print(f"Number of spatial orbitals: {num_orb}, Number of qubits: {n_qubits}")
    print(f"Number of parameters in the ansatz: {num_parameters}")

    backprop = UCJBackPropagator(ucj_op, nelec=nelec, num_orb=num_orb, h1e=h1e, h2e=h2e, ecore=constant)

    # CCSD-parameterized UCJ energy, before variational optimization.
    t_start = time.perf_counter()
    ucj_ccsd_energy = backprop.propagate()
    propagate_runtime = time.perf_counter() - t_start

    # Variationally optimize the circuit parameters starting from the CCSD-derived
    # parameters. Repeat the optimization `n_timing_trials` times
    checkpoint_path = results_dir / "UCJ_checkpoint.npz"
    initial_parameters = ucj_op.to_parameters(
        interaction_pairs=(alpha_alpha_indices(num_orb), alpha_beta_indices(num_orb))
    )
    optimize_runtimes = np.empty(n_timing_trials)
    result = None
    for trial in range(n_timing_trials):
        t_start = time.perf_counter()
        result = backprop.optimize_jax(
            x0=initial_parameters,
            interaction_pairs=(alpha_alpha_indices(num_orb), alpha_beta_indices(num_orb)),
            chunk_size=chunk_size,
            method=optimizer_method,
            options=optimizer_options,
            checkpoint_path=checkpoint_path,
        )
        optimize_runtimes[trial] = time.perf_counter() - t_start
    optimize_runtime_mean = optimize_runtimes.mean()
    optimize_runtime_sem = optimize_runtimes.std(ddof=1) / np.sqrt(n_timing_trials)
    ucj_optimized_energy = backprop.propagate(show_progress=False)

    print(f"Hartree-Fock energy: {hf_energy:.10f} Ha")
    print(f"CCSD energy: {ccsd_energy:.10f} Ha")
    print(f"CCSD-parameterized UCJ energy: {ucj_ccsd_energy:.10f} Ha")
    print(f"Variationally optimized UCJ energy: {ucj_optimized_energy:.10f} Ha")
    print(f"Backpropagation (CCSD-parameterized) runtime: {propagate_runtime:.4f} s")
    print(
        f"Variational optimization runtime ({n_timing_trials} trials): "
        f"{optimize_runtime_mean:.4f} +/- {optimize_runtime_sem:.4f} s"
    )

    np.savez(
        results_path,
        natoms=natoms,
        n_qubits=n_qubits,
        num_parameters=num_parameters,
        atomic_distance=atomic_distance,
        hf_energy=hf_energy,
        ccsd_energy=ccsd_energy,
        ccsd_runtime=ccsd_runtime,
        ucj_ccsd_energy=ucj_ccsd_energy,
        ucj_optimized_energy=ucj_optimized_energy,
        propagate_runtime=propagate_runtime,
        optimize_runtimes=optimize_runtimes,
        optimize_runtime_mean=optimize_runtime_mean,
        optimize_runtime_sem=optimize_runtime_sem,
        optimizer_nit=result.nit,
        optimizer_nfev=result.nfev,
        optimizer_njev=result.get("njev", result.nfev),
    )
    print(f"Saved results to {results_path}")
