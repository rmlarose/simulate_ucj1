"""Simulates the UCJ1 ansatz for Hydrogen chains.
"""
import itertools
import time
 
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from tqdm.auto import tqdm
 
import ffsim
import pyscf
import qiskit
import qiskit.providers.fake_provider

ENERGY_MODE = "burton"

# Parameters.
atomic_spacing: float = 0.774  # Angstroms. Distance between H atoms in the chain.
atom: str = "H"

natoms_values = [int(n) for n in np.arange(4, 80 + 1, 4)]
half_layer = False                       # If True, appends a final rotation to the circuit as in [1], but makes the energy worse.
alpha_alpha_indices = lambda norb: None  # Use lambda norb: [(p, p + 1) for p in range(norb - 1)] for an LUCJ circuit as in [1]. Use None to run a UCJ circuit with more gates that improves the energy.
alpha_beta_indices  = lambda norb: None  # Use lambda norb: [(p, p) for p in range(0, norb, 4) if p <= 16] for a (truncated) LUCJ circuit as in [1]. Use None to run a UCJ circuit with more gates that improves the energy.


def generate_linear_geometry(atom: str, natoms: int, atomic_distance: float = atomic_spacing) -> str:
    return "; ".join([f"{atom} 0 0 {i * atomic_distance}" for i in range(natoms)])


def propagate_through_orbital_rotations(h, g, u):
    h_bp = u.conj().T @ h @ u
    g_bp = np.einsum('pi,qj,pqrs,rk,sl->ijkl', u.conj(), u, g, u.conj(), u, optimize=True)
    return h_bp, g_bp


def propagate_through_jastrow(same, diff, norb):
    N = 2 * norb

    def phases(nelec_probe):
        da = int(pyscf.fci.cistring.num_strings(norb, nelec_probe[0]))
        db = int(pyscf.fci.cistring.num_strings(norb, nelec_probe[1]))
        v = np.ones(da * db, dtype=complex)
        w = ffsim.apply_diag_coulomb_evolution(
            v,
            (same, diff, same),
            time=-1.0,
            norb=norb,
            nelec=nelec_probe,
        )
        return np.angle(w).reshape(da, db)

    occ1 = [int(o[0]) for o in pyscf.fci.cistring.gen_occslst(range(norb), 1)]
    occ2 = [(int(o[0]), int(o[1])) for o in pyscf.fci.cistring.gen_occslst(range(norb), 2)]

    L = np.zeros(N)
    A = np.zeros((N, N))
    ph_a = phases((1, 0)).ravel()
    ph_b = phases((0, 1)).ravel()
    for i, p in enumerate(occ1):
        L[p] = ph_a[i]
        L[norb + p] = ph_b[i]

    ph_aa = phases((2, 0)).ravel()
    ph_bb = phases((0, 2)).ravel()
    for i, (p, q) in enumerate(occ2):
        A[p, q] = A[q, p] = (ph_aa[i] - L[p] - L[q]) / 2
        A[norb + p, norb + q] = A[norb + q, norb + p] = (ph_bb[i] - L[norb + p] - L[norb + q]) / 2

    ph_ab = phases((1, 1))
    for i, p in enumerate(occ1):
        for j, q in enumerate(occ1):
            A[p, norb + q] = A[norb + q, p] = (ph_ab[i, j] - L[p] - L[norb + q]) / 2

    return A, L


class _SectorCalc:
    def __init__(self, Q, memo_cap):
        self.Q = Q
        self.Qc = Q.conj()
        self.memo = {}
        self.cap = memo_cap

    def get(self, phi):
        key = phi.round(12).tobytes()
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        d = np.exp(1j * phi)
        S = self.Qc.T @ (d[:, None] * self.Q)
        lu, piv = lu_factor(S)
        det = np.prod(np.diag(lu)) * (-1) ** np.count_nonzero(piv != np.arange(len(piv)))
        val = (d, lu, piv, det)
        if len(self.memo) < self.cap:
            self.memo[key] = val
        return val

    def rho(self, entry, i, j):
        """Returns rho[i, j] = <a+_j a_i> / det."""
        d, lu, piv, _ = entry
        return d[i] * (self.Q[i] @ lu_solve((lu, piv), self.Qc[j]))


def compute_energy(
    Q: np.typing.NDArray,
    ecore: float,
    h_bp: np.typing.NDArray,
    g_bp: np.typing.NDArray,
    A: np.typing.NDArray,
    L: np.typing.NDArray,
    norb: int,
    memory_budget_bytes: float = 2e9,
) -> float:
    """Returns the energy E = ecore + sum h_bp . gamma + 1/2 g_bp . Gamma,
    evaluated exactly on the Slater determinant |Q> using the Löwdin rules for
    matrix elements of a monomial times a diagonal phase.

    Args:
        Q: (norb x n_occ) Occupied-orbital matrix of e^{-K}|HF>, one spin
            sector (alpha == beta for the spin-balanced ansatz).
        ecore: Energy constant.
        h_bp: (norb x norb) One-body integrals back-propagated through the
            trailing orbital rotation u (i.e. u^dag h u).
        g_bp: (norb,)*4 Two-body integrals back-propagated through u.
        A, L: Jastrow phase data. The e^{iJ} conjugation dresses each term with
            a diagonal phase e^{i phi . n}; for an occupation change Delta
            (length 2*norb, alpha sites [0,norb), beta [norb,2norb)),
            phi = -2 A Delta, const = -(Delta^T A Delta + L . Delta).
        norb: Number of orbitals.
        memory_budget_bytes: Approximate cap on the LU cache. Once reached,
            factorizations are recomputed instead of stored.
    """
    N = 2 * norb
    n_occ = Q.shape[1]
    cap = max(1000, int(memory_budget_bytes // (n_occ * n_occ * 16 + norb * 16)))
    sec = _SectorCalc(Q, cap)

    def dress(Delta):
        phi = -2.0 * (A @ Delta)
        const = np.exp(-1j * (Delta @ A @ Delta + L @ Delta))
        return phi[:norb], phi[norb:], const

    E = ecore + 0j

    # Energy from one-body terms: 2 * sum_pq h[p,q] <a+_p a_q D>.
    for p in range(norb):
        for q in range(norb):
            Delta = np.zeros(N)
            Delta[p] += 1
            Delta[q] -= 1
            phi_a, phi_b, c = dress(Delta)
            ea = sec.get(phi_a)
            eb = sec.get(phi_b)
            E += 2 * h_bp[p, q] * c * ea[3] * eb[3] * sec.rho(ea, q, p)

    # Energy from two-body terms.
    n4 = norb**4
    bar = tqdm(
        itertools.product(range(norb), repeat=4),
        total=n4,
        desc="compute_energy",
        unit="term",
        mininterval=0.25,
    )
    for p, q, r, s in bar:
        g = g_bp[p, q, r, s]

        # Same-spin (aa)+(bb) energies.
        Delta = np.zeros(N)
        Delta[[p, r]] += 1
        Delta[[s, q]] -= 1
        phi_a, phi_b, c = dress(Delta)
        ea, eb = sec.get(phi_a), sec.get(phi_b)
        wick = (sec.rho(ea, q, p) * sec.rho(ea, s, r)
                - sec.rho(ea, s, p) * sec.rho(ea, q, r))
        E += g * c * ea[3] * eb[3] * wick

        # Opposite-spin (ab)+(ba) energies.
        Delta = np.zeros(N)
        Delta[p] += 1
        Delta[q] -= 1
        Delta[norb + r] += 1
        Delta[norb + s] -= 1
        phi_a, phi_b, c = dress(Delta)
        ea, eb = sec.get(phi_a), sec.get(phi_b)
        E += g * c * (ea[3] * sec.rho(ea, q, p)) * (eb[3] * sec.rho(eb, s, r))

        bar.set_postfix(E=f"{E.real:.6f}", refresh=False)

    bar.close()

    if abs(E.imag) > 1e-8:
        print(f"warning: Im(E) = {E.imag:.2e}")
    return float(E.real)


def compute_energy_burton(
    Q: np.typing.NDArray,
    ecore: float,
    h_bp: np.typing.NDArray,
    g_bp: np.typing.NDArray,
    A: np.typing.NDArray,
    L: np.typing.NDArray,
    norb: int,
    tol: float = 1e-12,
) -> float:
    """Computes the UCJ1 energy using the generalized nonorthogonal Wick's
    theorem, which remains valid when the overlap matrix S is singular.

    Args:
      Q: (norb x n_occ) Occupied-orbital matrix of e^{-K}|HF>, one spin
          sector (alpha == beta for the spin-balanced ansatz).
      ecore: Energy constant.
      h_bp: (norb x norb) One-body integrals back-propagated through the
          trailing orbital rotation u (i.e. u^dag h u).
      g_bp: (norb,)*4 Two-body integrals back-propagated through u.
      A, L: Scalar and vector phase data from backpropagating the Hamiltonian
          through the Jastrow operation.
      norb: Number of orbitals.
      tol: Singular values of S below this are treated as zero overlaps.
    """
    N = 2 * norb
    Qc = Q.conj()

    # Cache computed terms for speed.
    cache = {}
    cache_cap = max(10_000, int(2e9 // (norb * norb * 16)))   # ~2 GB budget
    def transition(phi):
        """Löwdin pairing of |Q> and D_phi|Q> via the SVD S = U Sigma V^dag.

        Returns (m, Stil, W, Ps) where m is the number of zero-overlap orbital
        pairs, Stil is the reduced overlap (product of nonzero singular values,
        times the phase of det U det V^dag), W is the weighted co-density
        matrix, and Ps holds one unweighted co-density per zero-overlap pair.
        """
        key = phi.round(12).tobytes()
        hit = cache.get(key)
        if hit is not None:
            return hit
        d = np.exp(1j * phi)
        Cw = d[:, None] * Q
        S = Qc.T @ Cw
        U, sigma, Vh = np.linalg.svd(S)
        V = Vh.conj().T
        nonzero = sigma > tol
        m = int((~nonzero).sum())
        phase = np.linalg.det(U) * np.conj(np.linalg.det(V))
        Stil = phase * np.prod(sigma[nonzero]) if nonzero.any() else phase
        Cx_t = Q @ U        # biorthogonalized bra orbitals
        Cw_t = Cw @ V       # biorthogonalized ket orbitals
        W = (Cw_t[:, nonzero] / sigma[nonzero]) @ Cx_t[:, nonzero].conj().T
        Ps = [np.outer(Cw_t[:, k], Cx_t[:, k].conj()) for k in np.flatnonzero(~nonzero)]
        if len(cache) < cache_cap:
            cache[key] = (m, Stil, W, Ps)
        return m, Stil, W, Ps

    def dress(Delta):
        phi = -2.0 * (A @ Delta)
        const = np.exp(-1j * (Delta @ A @ Delta + L @ Delta))
        return phi[:norb], phi[norb:], const

    def one_body(entry, p, q):
        """<Q| a_p^dagger a_q D_phi |Q>. Vanishes for m > 1."""
        m, Stil, W, Ps = entry
        if m == 0:
            return Stil * W[q, p]
        if m == 1:
            return Stil * Ps[0][q, p]
        return 0.0 + 0.0j

    def two_body(entry, p, q, r, s):
        """<Q| a_p^dagger a_r^dagger a_s a_q D_phi |Q>. Vanishes for m > 2."""
        m, Stil, W, Ps = entry
        if m == 0:
            return Stil * (W[q, p] * W[s, r] - W[s, p] * W[q, r])
        if m == 1:
            P = Ps[0]
            return Stil * (P[q, p] * W[s, r] + W[q, p] * P[s, r]
                           - P[s, p] * W[q, r] - W[s, p] * P[q, r])
        if m == 2:
            P1, P2 = Ps
            return Stil * (P1[q, p] * P2[s, r] + P2[q, p] * P1[s, r]
                           - P1[s, p] * P2[q, r] - P2[s, p] * P1[q, r])
        return 0.0 + 0.0j

    E = ecore + 0j
    # Energy from one-body terms: 2 * sum_pq h[p,q] <a_p^\dagger a_q D>.
    for p in range(norb):
        for q in range(norb):
            Delta = np.zeros(N)
            Delta[p] += 1
            Delta[q] -= 1
            phi_a, phi_b, c = dress(Delta)
            entry_a = transition(phi_a)
            entry_b = transition(phi_b)
            E += 2 * h_bp[p, q] * c * one_body(entry_a, p, q) * entry_b[1]

    # Energy from two-body terms.
    n4 = norb ** 4
    bar = tqdm(
        itertools.product(range(norb), repeat=4),
        total=n4,
        desc="compute_energy_burton",
        unit="term",
        mininterval=0.25,
    )
    for (p, q, r, s) in bar:
        g = g_bp[p, q, r, s]

        # Compute same-spin (aa)+(bb) energies.
        Delta = np.zeros(N); Delta[[p, r]] += 1; Delta[[s, q]] -= 1
        phi_a, phi_b, c = dress(Delta)
        entry_a, entry_b = transition(phi_a), transition(phi_b)
        E += g * c * two_body(entry_a, p, q, r, s) * entry_b[1]

        # Compute opposite-spin (ab)+(ba) energies.
        Delta = np.zeros(N); Delta[p] += 1; Delta[q] -= 1
        Delta[norb + r] += 1; Delta[norb + s] -= 1
        phi_a, phi_b, c = dress(Delta)
        entry_a, entry_b = transition(phi_a), transition(phi_b)
        E += g * c * one_body(entry_a, p, q) * one_body(entry_b, r, s)

        # Update progress bar.
        bar.set_postfix(E=f"{E.real:.6f}", refresh=False)

    # Update progress bar with final energy.
    bar.close()

    if abs(E.imag) > 1e-8:
        print(f"warning: Im(E) = {E.imag:.2e}")
    return float(E.real)


if __name__ == "__main__":

    if ENERGY_MODE == "burton":
        compute_energy = compute_energy_burton
    elif ENERGY_MODE == "lowdin":
        pass
    else:
        raise ValueError(f"ENERGY_MODE must be burton or lowdin but was {ENERGY_MODE}")

    all_energies_ucj = []
    all_energies_hf = []
    all_energies_ccsd = []
    for natoms in natoms_values:
        print("Status: natoms =", natoms)
        # Build the molecule.
        mol = pyscf.gto.Mole()
        mol.build(
            atom=generate_linear_geometry(atom, natoms),
            basis="sto-6g",
        )
        n_frozen = 0
        active_space = range(n_frozen, mol.nao_nr())
    
        # Run Hartree-Fock.
        mf_as = pyscf.scf.RHF(mol).run()
    
        num_orb = len(active_space)
        n_electrons = int(sum(mf_as.mo_occ[active_space]))
        n_alpha = (n_electrons + mol.spin) // 2
        n_beta = (n_electrons - mol.spin) // 2
        nelec = (n_alpha, n_beta)
        print(f"norb = {num_orb}")
        print(f"nelec = {nelec}")
    
        # Run CCSD.
        mycc = pyscf.cc.CCSD(
            mf_as, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
        )
        eccsd, *_ = mycc.kernel()
        print("CCSD correlation energy:", eccsd)
        print("CCSD total energy:", mycc.e_tot)
    
        cas = pyscf.mcscf.CASCI(mf_as, num_orb, nelec)
        mo = cas.sort_mo(active_space, base=0)
    
        h1e, constant = cas.get_h1cas(mo)
        h2e = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), num_orb)
    
        ccsd = mycc
        ccsd_energy = mycc.e_tot
        print(f"CCSD energy: {ccsd_energy:.10e}")
    
        nelec = n_electrons

        # Build the UCJ Operation.
        n_reps_base = 2 if half_layer else 1
        base_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=ccsd.t2,
            n_reps=n_reps_base,
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
    
        # Build the quantum circuit.
        nelec = (nelec // 2, nelec // 2)
        qubits = qiskit.QuantumRegister(2 * num_orb, name="q")
        circuit = qiskit.QuantumCircuit(qubits)
        circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orb, nelec), qubits)
        circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
        coupling_map = qiskit.transpiler.CouplingMap.from_full(num_qubits=circuit.num_qubits)
        backend = qiskit.providers.fake_provider.GenericBackendV2(
            coupling_map.size(), coupling_map=coupling_map,
            basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
        )
        compiled = qiskit.transpile(circuit, backend=backend, optimization_level=0)
        print(f"UCJ circuit acts on {compiled.num_qubits} qubit(s).")
        print(f"Operation count:", compiled.count_ops())
    
        # Get the final orbital rotations from the UCJ operator.
        W = ucj_op.orbital_rotations[0]
        Wf = ucj_op.final_orbital_rotation
        u = W if Wf is None else Wf @ W
    
        # Simulate.
        start = time.perf_counter()
        h_bp, g_bp = propagate_through_orbital_rotations(h1e, h2e, u)
        A_J, L_J = propagate_through_jastrow(ucj_op.diag_coulomb_mats[0][0], ucj_op.diag_coulomb_mats[0][1], num_orb)
        Q = W.conj().T[:, :nelec[0]]
        E_ucj = compute_energy(Q, constant, h_bp, g_bp, A_J, L_J, num_orb)
        stop = time.perf_counter()
        print("Runtime (seconds)", stop - start)
    
        # Show/save results.
        print(f"UCJ Energy = {E_ucj:.10f} Ha")
        print(f"Hartree-Fock Energy = {mf_as.e_tot:.10f} Ha")
        print(f"CCSD Energy = {ccsd.e_tot:.10f} Ha")
        all_energies_ucj.append(E_ucj)
        all_energies_hf.append(mf_as.e_tot)
        all_energies_ccsd.append(ccsd.e_tot)
        np.savetxt(f"all_energies_ucj_hchains_{ENERGY_MODE}.txt", all_energies_ucj)
        np.savetxt("all_energies_hf_hchains.txt", all_energies_hf)
        np.savetxt("all_energies_ccsd_hchains.txt", all_energies_ccsd)
