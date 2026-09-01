"""Simulates the UCJ1 ansatz for Hydrogen chains.
"""
import itertools
import time
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from tqdm.auto import tqdm
 
import ffsim
import ffsim.qiskit
import pyscf

ENERGY_MODE = "burton"

# Parameters.
atomic_spacing: float = 0.774  # Angstroms. Distance between H atoms in the chain.
atom: str = "H"
def generate_linear_geometry(atom: str, natoms: int, atomic_distance: float = atomic_spacing) -> str:
    return "; ".join([f"{atom} 0 0 {i * atomic_distance}" for i in range(natoms)])

natoms_values = [int(n) for n in np.arange(4, 80 + 1, 4)]
half_layer = False                       # If True, appends a final rotation to the circuit as in [1], but makes the energy worse.
alpha_alpha_indices = lambda norb: None  # Use lambda norb: [(p, p + 1) for p in range(norb - 1)] for an LUCJ circuit as in [1]. Use None to run a UCJ circuit with more gates that improves the energy.
alpha_beta_indices  = lambda norb: None  # Use lambda norb: [(p, p) for p in range(0, norb, 4) if p <= 16] for a (truncated) LUCJ circuit as in [1]. Use None to run a UCJ circuit with more gates that improves the energy.


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
        When m = 0, Stil = det(S) and W = D_phi Q S^-1 Q^dag, i.e. exactly the
        determinant and transition density of the standard Löwdin formula.
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




# -----------------------------------------------------------------------------
# Appendix H of arXiv:2503.21041
# -----------------------------------------------------------------------------
# This implementation follows Appendix H term-by-term:
#
#   H' = G^\dagger H G = sum_k h_k Sigma_k                         (H4-H5)
#   |alpha_k> = D^\dagger Sigma_k D G^\dagger |x0>,
#   |beta>    = G^\dagger |x0>                                    (H7)
#
# For EACH Pauli term k independently, sample z ~ |beta_z|^2 and estimate
#
#   <beta|alpha_k> = E_z[ <z|alpha_k> / <z|beta> ],
#
# with <z|alpha_k> evaluated using Eq. (H8).  We deliberately do not combine
# the Pauli terms into a local-energy estimator in this reference version.

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PauliTerm:
    """Compiled Pauli tensor product Sigma_k from Eq. (H5)."""

    label: str
    coeff: complex
    flip_mask: int      # X or Y locations
    phase_mask: int     # Y or Z locations
    n_y: int


def compile_pauli_term(label: str, coeff: complex) -> PauliTerm:
    """Compile a Qiskit Pauli label into bit masks.

    Qiskit labels are printed q_{n-1} ... q_0, so reverse(label) maps character
    position directly to the qubit index used by the integer bitstrings.
    """
    flip_mask = 0
    phase_mask = 0
    n_y = 0
    for q, p in enumerate(reversed(label)):
        if p in ("X", "Y"):
            flip_mask |= 1 << q
        if p in ("Y", "Z"):
            phase_mask |= 1 << q
        if p == "Y":
            n_y += 1
    return PauliTerm(
        label=label,
        coeff=complex(coeff),
        flip_mask=flip_mask,
        phase_mask=phase_mask,
        n_y=n_y,
    )


def pauli_phase_on_source(source: int, term: PauliTerm) -> complex:
    r"""Return sigma such that Sigma_k |source> = sigma |source xor flip_mask>.

    Uses
        X|b> = |1-b>,
        Y|b> = i (-1)^b |1-b>,
        Z|b> = (-1)^b |b>.
    """
    parity = (source & term.phase_mask).bit_count() & 1
    return (1j ** term.n_y) * (-1 if parity else 1)


def split_spin_bitstring(z: int, norb: int) -> tuple[int, int]:
    mask = (1 << norb) - 1
    return z & mask, z >> norb


def join_spin_bitstring(z_a: int, z_b: int, norb: int) -> int:
    # ffsim/Qiskit convention: alpha occupies qubits [0, norb), beta the next norb.
    return int(z_a) | (int(z_b) << norb)


def occupation_vector(z: int, norb: int) -> np.ndarray:
    """0/1 occupation vector for an integer bitstring of length norb."""
    return np.fromiter(((z >> p) & 1 for p in range(norb)), dtype=float, count=norb)


def jastrow_phase_angle(
    z_a: int,
    z_b: int,
    J_same: np.ndarray,
    J_diff: np.ndarray,
    norb: int,
) -> float:
    r"""Return theta(z) for D|z> = exp(i theta(z)) |z>.

    ffsim's spin-balanced UCJ convention is

      J = 1/2 sum_{ij,sigma,tau} J^(sigma,tau)_ij n_i,sigma n_j,tau,

    with J^(aa)=J^(bb)=J_same and J^(ab)=J^(ba)=J_diff.  Therefore

      theta(a,b) = 1/2 a^T J_same a
                 +       a^T J_diff b
                 + 1/2 b^T J_same b.
    """
    a = occupation_vector(int(z_a), norb)
    b = occupation_vector(int(z_b), norb)
    return float(
        0.5 * (a @ J_same @ a)
        + (a @ J_diff @ b)
        + 0.5 * (b @ J_same @ b)
    )


def beta_amplitudes(
    bitstrings_a,
    bitstrings_b,
    norb: int,
    nelec: tuple[int, int],
    G: np.ndarray,
) -> np.ndarray:
    r"""Compute <z|beta>, beta = G^dagger |HF>, using ffsim."""
    occupied_orbitals = (range(nelec[0]), range(nelec[1]))
    return ffsim.slater_determinant_amplitudes(
        (
            [int(z) for z in bitstrings_a],
            [int(z) for z in bitstrings_b],
        ),
        norb,
        occupied_orbitals,
        G.conj().T,
    )


def sample_beta(
    norb: int,
    nelec: tuple[int, int],
    G: np.ndarray,
    shots: int,
    rng: np.random.Generator,
):
    r"""Sample |z> ~ |beta_z|^2 for beta = G^dagger|HF>, using ffsim."""
    occupied_orbitals = (range(nelec[0]), range(nelec[1]))
    samples_a, samples_b = ffsim.sample_slater(
        norb,
        occupied_orbitals,
        orbital_rotation=G.conj().T,
        shots=shots,
        concatenate=False,
        bitstring_type=ffsim.BitstringType.INT,
        seed=rng,
    )
    return np.asarray(samples_a), np.asarray(samples_b)


def h8_partial_overlaps(
    samples_a,
    samples_b,
    term: PauliTerm,
    norb: int,
    nelec: tuple[int, int],
    G: np.ndarray,
    J_same: np.ndarray,
    J_diff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Evaluate Eq. (H8) for a batch of sampled bitstrings.

    Returns
    -------
    alpha_z:
        <z|D^dagger Sigma_k D G^dagger|x0>.
    beta_z:
        <z|G^dagger|x0>.

    For a source string z' satisfying Sigma_k|z'> = sigma(z')|z>,

      alpha_z = exp[-i theta(z)] sigma(z') exp[i theta(z')] beta_{z'}.
    """
    samples_a = [int(z) for z in samples_a]
    samples_b = [int(z) for z in samples_b]
    shots = len(samples_a)

    beta_z = beta_amplitudes(samples_a, samples_b, norb, nelec, G)
    alpha_z = np.zeros(shots, dtype=complex)

    # Determine the unique transformed bitstring z' from Eq. (H8).
    zprime_a = np.zeros(shots, dtype=object)
    zprime_b = np.zeros(shots, dtype=object)
    valid = np.zeros(shots, dtype=bool)
    sources = np.zeros(shots, dtype=object)

    for s, (za, zb) in enumerate(zip(samples_a, samples_b)):
        z = join_spin_bitstring(za, zb, norb)
        source = z ^ term.flip_mask  # z' in Eq. (H8)
        zpa, zpb = split_spin_bitstring(source, norb)
        sources[s] = source
        zprime_a[s] = zpa
        zprime_b[s] = zpb
        # beta has fixed N_alpha and N_beta. Outside this sector beta_{z'} = 0.
        valid[s] = (
            int(zpa).bit_count() == nelec[0]
            and int(zpb).bit_count() == nelec[1]
        )

    valid_indices = np.flatnonzero(valid)
    if len(valid_indices):
        amps_zprime = beta_amplitudes(
            [int(zprime_a[s]) for s in valid_indices],
            [int(zprime_b[s]) for s in valid_indices],
            norb,
            nelec,
            G,
        )

        for amp, s in zip(amps_zprime, valid_indices):
            za = samples_a[s]
            zb = samples_b[s]
            zpa = int(zprime_a[s])
            zpb = int(zprime_b[s])
            source = int(sources[s])

            theta_z = jastrow_phase_angle(za, zb, J_same, J_diff, norb)
            theta_zprime = jastrow_phase_angle(zpa, zpb, J_same, J_diff, norb)
            sigma = pauli_phase_on_source(source, term)

            # Eq. (H8): e^{i phi} <z'|G^dagger|x0>
            alpha_z[s] = (
                sigma * np.exp(1j * (theta_zprime - theta_z)) * amp
            )

    return alpha_z, beta_z


def estimate_one_h6_braket(
    term: PauliTerm,
    norb: int,
    nelec: tuple[int, int],
    G: np.ndarray,
    J_same: np.ndarray,
    J_diff: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> tuple[complex, np.ndarray]:
    r"""Estimate one braket on the RHS of Eq. (H6), exactly as H7 + Sec. IIIA.

    A fresh sample set is drawn for this Pauli term, deliberately mirroring the
    term-by-term construction in Appendix H.
    """
    samples_a, samples_b = sample_beta(norb, nelec, G, shots, rng)
    alpha_z, beta_z = h8_partial_overlaps(
        samples_a,
        samples_b,
        term,
        norb,
        nelec,
        G,
        J_same,
        J_diff,
    )

    # Under exact sampling, beta_z=0 is never sampled. Treat a numerical zero as an error.
    if np.any(np.abs(beta_z) < 1e-14):
        raise FloatingPointError(
            "Encountered a sampled beta amplitude numerically equal to zero."
        )

    # Mid-circuit l2 estimator from Sec. III A, applied to Eq. (H7).
    ratios = alpha_z / beta_z
    return np.mean(ratios), ratios


def appendix_h_pauli_decomposition(
    h_bp: np.ndarray,
    g_bp: np.ndarray,
    constant: float,
    norb: int,
    tol: float = 1e-12,
) -> list[PauliTerm]:
    r"""Construct H' = G^dagger H G = sum_k h_k Sigma_k (H4-H5)."""
    hprime = ffsim.MolecularHamiltonian(
        one_body_tensor=h_bp,
        two_body_tensor=g_bp,
        constant=constant,
    )
    fermion_hprime = ffsim.fermion_operator(hprime)
    qubit_hprime = ffsim.qiskit.jordan_wigner(
        fermion_hprime,
        norb=norb,
        tol=tol,
    ).simplify(atol=tol)

    return [compile_pauli_term(label, coeff) for label, coeff in qubit_hprime.to_list()]


def compute_energy_appendix_h(
    pauli_terms: list[PauliTerm],
    norb: int,
    nelec: tuple[int, int],
    G: np.ndarray,
    J_same: np.ndarray,
    J_diff: np.ndarray,
    shots_per_term: int,
    seed: int = 1,
) -> tuple[float, float, complex]:
    r"""Monte Carlo implementation of Appendix H, term-by-term.

    Returns
    -------
    energy:
        Real part of sum_k h_k <beta|D^dagger Sigma_k D|beta>.
    stderr:
        Conventional standard error of the REAL energy estimate, combining the
        independently sampled Pauli-term estimates in quadrature.
    energy_complex:
        Full complex estimate before taking the real part.
    """
    rng = np.random.default_rng(seed)
    energy = 0.0 + 0.0j
    variance_of_real_energy = 0.0

    bar = tqdm(pauli_terms, desc="Appendix H energy", unit="term")
    for term in bar:
        overlap, ratios = estimate_one_h6_braket(
            term,
            norb,
            nelec,
            G,
            J_same,
            J_diff,
            shots_per_term,
            rng,
        )
        energy += term.coeff * overlap

        # The term sample sets are independent, so variances add.
        contributions = np.real(term.coeff * ratios)
        if shots_per_term > 1:
            variance_of_real_energy += np.var(contributions, ddof=1) / shots_per_term

        bar.set_postfix(E=f"{energy.real:.10f}", refresh=False)

    stderr = math.sqrt(variance_of_real_energy)
    return float(energy.real), stderr, energy



def compute_energy_appendix_h_trajectory(
    pauli_terms: list[PauliTerm],
    norb: int,
    nelec: tuple[int, int],
    G: np.ndarray,
    J_same: np.ndarray,
    J_diff: np.ndarray,
    max_shots_per_term: int,
    seed: int = 1,
    runtime_batch_size: int = 5_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Run Appendix H ONCE up to N=max_shots_per_term and keep E(N), SE(N).

    This is the same term-by-term Appendix-H estimator as compute_energy_appendix_h,
    but it retains the full ordered sequence of sampled contributions for each Pauli
    term.  The cumulative first N samples of every term then give the energy and
    conventional standard error for every N = 1, ..., max_shots_per_term.

    No Appendix-H samples are thrown away and no calculation is repeated for smaller N.

    Each Pauli term has its own independent sample stream, exactly as in the reference
    implementation.  Samples are generated in batches only so that actual sampling
    runtime can also be accumulated as a function of N.

    Returns
    -------
    sample_counts:
        1, 2, ..., max_shots_per_term.
    energy_by_n:
        Real Appendix-H energy estimate using the first N samples / Pauli term.
    stderr_by_n:
        Conventional standard error of the real energy estimate at each N.
        stderr_by_n[0] is NaN because the sample variance is undefined for N=1.
    energy_complex_by_n:
        Full complex Appendix-H estimate at each N.
    runtime_sample_counts:
        Sample counts at which actual cumulative Appendix-H runtime was recorded.
    runtime_by_n:
        Actual cumulative time spent generating/evaluating Appendix-H samples up to
        the corresponding runtime_sample_counts value.  This excludes the one-time
        Burton and Jordan-Wigner setup work.
    """
    if max_shots_per_term < 1:
        raise ValueError("max_shots_per_term must be positive.")
    if runtime_batch_size < 1:
        raise ValueError("runtime_batch_size must be positive.")

    rng = np.random.default_rng(seed)
    sample_counts = np.arange(1, max_shots_per_term + 1, dtype=int)
    n_float = sample_counts.astype(float)

    # These accumulate the independent contribution from every Pauli term.
    energy_complex_by_n = np.zeros(max_shots_per_term, dtype=complex)
    variance_of_real_energy_by_n = np.zeros(max_shots_per_term, dtype=float)

    # Actual runtime is measured batch-by-batch.  Summing the j-th batch time over
    # all Pauli terms gives the extra time required to advance the whole Appendix-H
    # calculation from runtime checkpoint j-1 to j.
    batch_starts = list(range(0, max_shots_per_term, runtime_batch_size))
    batch_ends = [min(s + runtime_batch_size, max_shots_per_term) for s in batch_starts]
    batch_runtime = np.zeros(len(batch_starts), dtype=float)

    bar = tqdm(pauli_terms, desc="Appendix H trajectory", unit="term")
    for term_index, term in enumerate(bar, start=1):
        # Store only one Pauli term's N_max sampled energy contributions at a time.
        # Memory is therefore O(N_max), not O(N_max * number_of_Paulis).
        term_contributions = np.empty(max_shots_per_term, dtype=complex)

        for batch_index, (start, stop) in enumerate(zip(batch_starts, batch_ends)):
            shots = stop - start
            t_batch = time.perf_counter()

            _, ratios = estimate_one_h6_braket(
                term,
                norb,
                nelec,
                G,
                J_same,
                J_diff,
                shots,
                rng,
            )
            term_contributions[start:stop] = term.coeff * ratios

            batch_runtime[batch_index] += time.perf_counter() - t_batch

        # Energy from the first N samples of this Pauli term, for every N.
        cumulative_complex = np.cumsum(term_contributions, dtype=complex)
        energy_complex_by_n += cumulative_complex / n_float

        # Conventional sample variance of the REAL contribution for every N:
        #   s_N^2 = [sum x_i^2 - (sum x_i)^2/N] / (N-1)
        # and the variance of the mean is s_N^2 / N.
        x = term_contributions.real
        cumulative_x = np.cumsum(x, dtype=float)
        cumulative_x2 = np.cumsum(x * x, dtype=float)

        if max_shots_per_term > 1:
            numer = cumulative_x2[1:] - cumulative_x[1:] ** 2 / n_float[1:]
            # Guard only against tiny negative values from floating-point roundoff.
            numer = np.maximum(numer, 0.0)
            sample_variance = numer / (n_float[1:] - 1.0)
            variance_of_real_energy_by_n[1:] += sample_variance / n_float[1:]

        bar.set_postfix(
            E=f"{energy_complex_by_n[-1].real:.10f}",
            N=max_shots_per_term,
            refresh=False,
        )

    stderr_by_n = np.full(max_shots_per_term, np.nan, dtype=float)
    if max_shots_per_term > 1:
        stderr_by_n[1:] = np.sqrt(variance_of_real_energy_by_n[1:])

    runtime_sample_counts = np.asarray(batch_ends, dtype=int)
    runtime_by_n = np.cumsum(batch_runtime)

    return (
        sample_counts,
        energy_complex_by_n.real,
        stderr_by_n,
        energy_complex_by_n,
        runtime_sample_counts,
        runtime_by_n,
    )


def fixed_weight_bitstrings(norb: int, nocc: int) -> list[int]:
    out = []
    for occ in itertools.combinations(range(norb), nocc):
        z = 0
        for p in occ:
            z |= 1 << p
        out.append(z)
    return out


def compute_energy_appendix_h_exact_sum(
    pauli_terms: list[PauliTerm],
    norb: int,
    nelec: tuple[int, int],
    G: np.ndarray,
    J_same: np.ndarray,
    J_diff: np.ndarray,
) -> complex:
    r"""Exact finite sum of the Appendix-H estimator expectation, for validation only.

    This is NOT the Monte Carlo algorithm. It enumerates the fixed-particle-number
    support for a tiny test problem so we can test Eq. (H8) and the Pauli decomposition
    against the independent Burton energy to near machine precision.
    """
    strings_a = fixed_weight_bitstrings(norb, nelec[0])
    strings_b = fixed_weight_bitstrings(norb, nelec[1])

    all_a = []
    all_b = []
    for za in strings_a:
        for zb in strings_b:
            all_a.append(za)
            all_b.append(zb)

    beta = beta_amplitudes(all_a, all_b, norb, nelec, G)
    beta_map = {
        (int(za), int(zb)): amp
        for za, zb, amp in zip(all_a, all_b, beta)
    }

    energy = 0.0 + 0.0j
    for term in tqdm(pauli_terms, desc="Exact H8 validation", unit="term"):
        overlap = 0.0 + 0.0j
        for za, zb, beta_z in zip(all_a, all_b, beta):
            z = join_spin_bitstring(za, zb, norb)
            source = z ^ term.flip_mask
            zpa, zpb = split_spin_bitstring(source, norb)
            beta_zprime = beta_map.get((int(zpa), int(zpb)), 0.0 + 0.0j)
            if beta_zprime == 0:
                continue

            theta_z = jastrow_phase_angle(za, zb, J_same, J_diff, norb)
            theta_zprime = jastrow_phase_angle(zpa, zpb, J_same, J_diff, norb)
            sigma = pauli_phase_on_source(source, term)
            alpha_z = sigma * np.exp(1j * (theta_zprime - theta_z)) * beta_zprime
            overlap += np.conj(beta_z) * alpha_z

        energy += term.coeff * overlap

    return energy



if __name__ == "__main__":
    NATOMS = 20
    MAX_SHOTS_PER_TERM = 10_000
    SEED = 1
    RUN_EXACT_H8_CHECK = False

    # Samples are generated once, cumulatively, up to MAX_SHOTS_PER_TERM.
    # This batch size affects only how often actual runtime is recorded.
    RUNTIME_BATCH_SIZE = 5_000

    print(
        f"Appendix H cumulative test: H_{NATOMS}, "
        f"N_max={MAX_SHOTS_PER_TERM} samples / Pauli term"
    )

    # Build the same H-chain problem as simulate_ucj1.py.
    mol = pyscf.gto.Mole()
    mol.build(
        atom=generate_linear_geometry(atom, NATOMS),
        basis="sto-6g",
    )
    n_frozen = 0
    active_space = range(n_frozen, mol.nao_nr())

    mf_as = pyscf.scf.RHF(mol).run()
    num_orb = len(active_space)
    n_electrons = int(sum(mf_as.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)

    print(f"norb = {num_orb}")
    print(f"nelec = {nelec}")

    mycc = pyscf.cc.CCSD(
        mf_as,
        frozen=[i for i in range(mol.nao_nr()) if i not in active_space],
    )
    mycc.kernel()

    cas = pyscf.mcscf.CASCI(mf_as, num_orb, nelec)
    mo = cas.sort_mo(active_space, base=0)
    h1e, constant = cas.get_h1cas(mo)
    h2e = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), num_orb)

    # Exactly one UCJ repetition, with no final orbital rotation: H1 applies verbatim.
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=mycc.t2,
        n_reps=1,
        interaction_pairs=(alpha_alpha_indices(num_orb), alpha_beta_indices(num_orb)),
    )
    if ucj_op.n_reps != 1:
        raise ValueError(f"Appendix H test requires one UCJ layer, got {ucj_op.n_reps}.")
    if ucj_op.final_orbital_rotation is not None:
        raise ValueError("Appendix H H1 test requires no extra final orbital rotation.")

    # H1: |phi> = G D G^dagger |x0>.
    G = ucj_op.orbital_rotations[0]
    J_same = ucj_op.diag_coulomb_mats[0][0]
    J_diff = ucj_op.diag_coulomb_mats[0][1]

    # H4: H' = G^dagger H G.
    h_bp, g_bp = propagate_through_orbital_rotations(h1e, h2e, G)

    # Independent exact Burton energy from the supplied implementation.
    A_J, L_J = propagate_through_jastrow(J_same, J_diff, num_orb)
    Q = G.conj().T[:, :nelec[0]]
    t0 = time.perf_counter()
    E_burton = compute_energy_burton(
        Q,
        constant,
        h_bp,
        g_bp,
        A_J,
        L_J,
        num_orb,
    )
    print(f"Burton energy                  = {E_burton:.12f} Ha")
    print(f"Burton runtime                 = {time.perf_counter() - t0:.3f} s")

    # H5: Pauli decomposition of H'.
    print("\nRunning Appendix H algorithm")
    print("Constructing Pauli Hamiltonian via JW")
    jwt = time.perf_counter()
    pauli_terms = appendix_h_pauli_decomposition(h_bp, g_bp, constant, num_orb)
    print("Elapsed time (seconds)", time.perf_counter() - jwt)
    print(f"Number of Pauli terms in H'    = {len(pauli_terms)}")

    # Deterministic validation of H8 and H5-H6 on a tiny problem.
    if RUN_EXACT_H8_CHECK:
        t0 = time.perf_counter()
        E_h8_exact = compute_energy_appendix_h_exact_sum(
            pauli_terms,
            num_orb,
            nelec,
            G,
            J_same,
            J_diff,
        )
        delta_exact = E_h8_exact.real - E_burton
        print(f"Exact finite-sum H8 energy     = {E_h8_exact.real:.12f} Ha")
        print(f"Exact H8 - Burton              = {delta_exact:+.3e} Ha")
        print(f"Exact H8 imaginary part        = {E_h8_exact.imag:+.3e} Ha")
        print(f"Exact H8 validation runtime    = {time.perf_counter() - t0:.3f} s")
        if not np.isclose(E_h8_exact.real, E_burton, rtol=0.0, atol=1e-8):
            raise AssertionError(
                "Exact evaluation of the Appendix-H H8 construction does not match Burton."
            )

    # Run Appendix H.
    t0 = time.perf_counter()
    (
        sample_counts,
        energy_by_n,
        stderr_by_n,
        energy_complex_by_n,
        runtime_sample_counts,
        runtime_by_n,
    ) = compute_energy_appendix_h_trajectory(
        pauli_terms,
        num_orb,
        nelec,
        G,
        J_same,
        J_diff,
        max_shots_per_term=MAX_SHOTS_PER_TERM,
        seed=SEED,
        runtime_batch_size=RUNTIME_BATCH_SIZE,
    )
    elapsed = time.perf_counter() - t0

    E_appendix_h = energy_by_n[-1]
    stderr = stderr_by_n[-1]
    E_complex = energy_complex_by_n[-1]

    print()
    print(f"Appendix H sampled energy      = {E_appendix_h:.12f} Ha")
    print(f"Appendix H standard error      = {stderr:.10e} Ha")
    print(f"Appendix H - Burton            = {E_appendix_h - E_burton:+.3e} Ha")
    print(f"Appendix H imaginary part      = {E_complex.imag:+.3e} Ha")
    print(f"Appendix H sampling runtime    = {elapsed:.3f} s")
    if stderr > 0:
        print(f"Difference / standard error    = {(E_appendix_h - E_burton) / stderr:+.2f}")

    print(f"Hartree-Fock energy            = {mf_as.e_tot:.12f} Ha")
    print(f"CCSD energy                    = {mycc.e_tot:.12f} Ha")

    # Save all cumulative E(N) and SE(N) values.
    prefix = f"appendix_h_cumulative_H{NATOMS}"

    np.savez(
        f"{prefix}.npz",
        sample_counts=sample_counts,
        energy=energy_by_n,
        stderr=stderr_by_n,
        energy_complex=energy_complex_by_n,
        runtime_sample_counts=runtime_sample_counts,
        runtime_seconds=runtime_by_n,
        E_burton=E_burton,
        E_hf=mf_as.e_tot,
        E_ccsd=mycc.e_tot,
    )

    np.savetxt(
        f"{prefix}_energy_stderr.csv",
        np.column_stack(
            [
                sample_counts,
                energy_by_n,
                stderr_by_n,
                energy_complex_by_n.imag,
            ]
        ),
        delimiter=",",
        header="samples_per_pauli,energy_Ha,stderr_Ha,imaginary_energy_Ha",
        comments="",
    )

    np.savetxt(
        f"{prefix}_runtime.csv",
        np.column_stack([runtime_sample_counts, runtime_by_n]),
        delimiter=",",
        header="samples_per_pauli,runtime_seconds",
        comments="",
    )
