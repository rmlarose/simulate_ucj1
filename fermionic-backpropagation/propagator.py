"""Numerical backpropagator for the UCJ ansatz."""

import os

import numpy as np
import scipy.optimize

import pyscf
import ffsim
import jax
import jax.numpy as jnp

from tqdm import tqdm

from .jax_propagator import energy_and_grad_jax

class UCJBackPropagator: 
    """Numerical backpropagator for the UCJ ansatz.""" 
    
    def __init__(self, 
            ucj: ffsim.UCJOpSpinBalanced, 
            nelec: tuple[int, int],
            num_orb: int,
            h1e: np.typing.NDArray,
            h2e: np.typing.NDArray,
            ecore: float = 0.0
            ): 
        """
        Initialize the backpropagator with a UCJ operator.
        
        Args:
             ucj (ffsim.UCJOpSpinBalanced): The UCJ operator to use for backpropagation.  
             nelec (tuple[int, int]): Number of alpha and beta electrons.
             num_orb (int): Number of orbitals.
             h1e (np.ndarray): One-electron integrals.
             h2e (np.ndarray): Two-electron integrals.
             ecore (float): Core energy constant.
        """
        
        W = ucj.orbital_rotations[0]
        Wf = ucj.final_orbital_rotation

        self.num_orb = num_orb
        self.W = W
        self.u = W if Wf is None else Wf @ W
        self.op = ucj
        self.nelec = nelec
        self.ecore = ecore
        self.h1e = h1e
        self.h2e = h2e

    def propagate(self, show_progress: bool = True) -> float:
        return self._energy(self.op, show_progress=show_progress)

    def _energy(self, ucj: ffsim.UCJOpSpinBalanced, show_progress: bool = True) -> float:
        """Back-propagate and evaluate the energy for a given UCJ operator."""
        W = ucj.orbital_rotations[0]
        Wf = ucj.final_orbital_rotation
        u = W if Wf is None else Wf @ W

        h_bp, g_bp = _propagate_through_orbital_rotations(self.h1e, self.h2e, u)
        A_J, L_J = _propagate_through_jastrow(ucj.diag_coulomb_mats[0][0], ucj.diag_coulomb_mats[0][1], self.num_orb)
        Q = W.conj().T[:, :self.nelec[0]]
        return _compute_energy(Q, self.ecore, h_bp, g_bp, A_J, L_J, self.num_orb, show_progress=show_progress)

    def optimize(
        self,
        x0: np.typing.NDArray | None = None,
        interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None] | None = None,
        show_progress: bool = True,
        **minimize_options,
    ) -> scipy.optimize.OptimizeResult:
        """
        Variationally optimize the UCJ circuit parameters to minimize the
        back-propagated energy with NumPy.

        Args:
            x0: Initial parameters
            interaction_pairs: Pair connectivity for the ansatz
            show_progress: Show a progress bar
            **minimize_options: Additional keyword arguments forwarded to
                `scipy.optimize.minimize` (e.g. `method`, `options`, `tol`,
                `bounds`, `callback`).

        Returns:
            The `scipy.optimize.OptimizeResult` from the minimization.
            Also updates `self.op` and self.W/self.u to the optimized operator.
        """
        n_reps, _, norb, _ = self.op.diag_coulomb_mats.shape
        with_final_orbital_rotation = self.op.final_orbital_rotation is not None

        if x0 is None:
            x0 = self.op.to_parameters(interaction_pairs=interaction_pairs)

        def cost(params: np.typing.NDArray) -> float:
            ucj = ffsim.UCJOpSpinBalanced.from_parameters(
                params,
                norb=norb,
                n_reps=n_reps,
                interaction_pairs=interaction_pairs,
                with_final_orbital_rotation=with_final_orbital_rotation,
            )
            return self._energy(ucj, show_progress=False)

        user_callback = minimize_options.pop("callback", None)
        pbar = None
        if show_progress:
            maxiter = (minimize_options.get("options") or {}).get("maxiter")
            pbar = tqdm(total=maxiter, desc="optimize", unit="step")

            def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
                pbar.update(1)
                pbar.set_postfix(energy=f"{intermediate_result.fun:.8f}")
                if user_callback is not None:
                    user_callback(intermediate_result)

            minimize_options["callback"] = callback
        elif user_callback is not None:
            minimize_options["callback"] = user_callback

        try:
            result = scipy.optimize.minimize(cost, x0, **minimize_options)
        finally:
            if pbar is not None:
                pbar.close()

        self.op = ffsim.UCJOpSpinBalanced.from_parameters(
            result.x,
            norb=norb,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        self.W = self.op.orbital_rotations[0]
        self.u = self.W if self.op.final_orbital_rotation is None else self.op.final_orbital_rotation @ self.W

        return result

    def optimize_jax(
        self,
        x0: np.typing.NDArray | None = None,
        interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None] | None = None,
        chunk_size: int | None = None,
        show_progress: bool = True,
        checkpoint_path: str | os.PathLike | None = None,
        checkpoint_interval: int = 1,
        **minimize_options,
    ) -> scipy.optimize.OptimizeResult:
        """
        Variationally optimize the UCJ circuit parameters to minimize the
        back-propagated energy with JAX.

        Args:
            x0: Initial parameters
            interaction_pairs: Pair connectivity for the ansatz
            show_progress: Show a progress bar
            **minimize_options: Additional keyword arguments forwarded to
                `scipy.optimize.minimize` (e.g. `method`, `options`, `tol`,
                `bounds`, `callback`).
            chunk_size: Optionally bounds memory by processing the two-body
                sum via `jax.lax.map` in chunks of this size
                (must evenly divide `norb**4`)
            checkpoint_path: Periodically save the current parameters 
                and energy to this path as an npz with the interval 
                defined by checkpoint_interval.
            checkpoint_interval: Save this often in iterations.
        Returns:
            The `scipy.optimize.OptimizeResult` from the minimization.
            Also updates `self.op` and self.W/self.u to the optimized operator.
        """
        n_reps, _, norb, _ = self.op.diag_coulomb_mats.shape
        with_final_orbital_rotation = self.op.final_orbital_rotation is not None

        start_iter = 0
        if x0 is None and checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = np.load(checkpoint_path)
            x0 = checkpoint["x"]
            start_iter = int(checkpoint["nit"])
            print(f"optimize_jax: resuming from checkpoint {checkpoint_path} at iteration {start_iter}")
            options = dict(minimize_options.get("options") or {})
            if "maxiter" in options:
                options["maxiter"] = max(options["maxiter"] - start_iter, 0)
                minimize_options["options"] = options
        if x0 is None:
            x0 = self.op.to_parameters(interaction_pairs=interaction_pairs)

        h1e_j = jnp.array(self.h1e)
        h2e_j = jnp.array(self.h2e)

        value_and_grad_fn = jax.jit(
            lambda params: energy_and_grad_jax(
                params, h1e_j, h2e_j, self.ecore, self.nelec, norb, n_reps,
                interaction_pairs, with_final_orbital_rotation, chunk_size,
            )
        )

        def cost(params: np.typing.NDArray):
            energy, grad = value_and_grad_fn(jnp.array(params))
            return float(energy), np.asarray(grad)

        def save_checkpoint(x: np.typing.NDArray, nit: int, energy: float) -> None:
            tmp_path = f"{checkpoint_path}.tmp"
            with open(tmp_path, "wb") as f:
                np.savez(f, x=x, nit=nit, energy=energy)
            os.replace(tmp_path, checkpoint_path)

        user_callback = minimize_options.pop("callback", None)
        pbar = None
        if show_progress:
            maxiter = (minimize_options.get("options") or {}).get("maxiter")
            total = None if maxiter is None else maxiter + start_iter
            pbar = tqdm(total=total, initial=start_iter, desc="optimize_jax", unit="step")

        iteration = start_iter

        def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
            nonlocal iteration
            iteration += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(energy=f"{intermediate_result.fun:.8f}")
            if checkpoint_path is not None and iteration % checkpoint_interval == 0:
                save_checkpoint(intermediate_result.x, iteration, float(intermediate_result.fun))
            if user_callback is not None:
                user_callback(intermediate_result)

        if show_progress or checkpoint_path is not None or user_callback is not None:
            minimize_options["callback"] = callback

        try:
            result = scipy.optimize.minimize(cost, x0, jac=True, **minimize_options)
            if checkpoint_path is not None:
                save_checkpoint(result.x, iteration, float(result.fun))
        finally:
            if pbar is not None:
                pbar.close()

        self.op = ffsim.UCJOpSpinBalanced.from_parameters(
            result.x,
            norb=norb,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        self.W = self.op.orbital_rotations[0]
        self.u = self.W if self.op.final_orbital_rotation is None else self.op.final_orbital_rotation @ self.W

        return result

def _propagate_through_orbital_rotations(h1e, h2e, u):
    """
    Propagate the one and two electron integrals through the orbital rotations 
    defined by the unitary matrix u.

    Args:
        h1e (np.ndarray): The one-electron integrals.
        h2e (np.ndarray): The two-electron integrals.
        u (np.ndarray): The unitary matrix defining the orbital rotations.
    """
    h_bp = u.conj().T @ h1e @ u
    g_bp = np.einsum('pi,qj,pqrs,rk,sl->ijkl', u.conj(), u, h2e, u.conj(), u, optimize=True)
    return h_bp, g_bp

def _propagate_through_jastrow(same, diff, norb): 
    N = 2 * norb
    def phases(nelec_probe):
        da = int(pyscf.fci.cistring.num_strings(norb, nelec_probe[0]))
        db = int(pyscf.fci.cistring.num_strings(norb, nelec_probe[1]))
        v = np.ones(da * db, dtype=complex)
        w = ffsim.apply_diag_coulomb_evolution(
            v, (same, diff, same),
            time=-1.0, norb=norb,
            nelec=nelec_probe
        )
        return np.angle(w).reshape(da, db)

    occ1 = [int(o[0]) for o in pyscf.fci.cistring.gen_occslst(range(norb), 1)]
    occ2 = [(int(o[0]), int(o[1])) for o in pyscf.fci.cistring.gen_occslst(range(norb), 2)]

    L = np.zeros(N)
    A = np.zeros((N, N))
    ph_a = phases((1, 0)).ravel(); ph_b = phases((0, 1)).ravel()
    for i, p in enumerate(occ1):
        L[p] = ph_a[i]
        L[norb + p] = ph_b[i]

    ph_aa = phases((2, 0)).ravel()
    ph_bb = phases((0, 2)).ravel()
    for i, (p, q) in enumerate(occ2):
        A[p, q] = A[q, p] = (ph_aa[i] - L[p] - L[q]) / 2
        A[norb+p, norb+q] = A[norb+q, norb+p] = (ph_bb[i] - L[norb+p] - L[norb+q]) / 2

    ph_ab = phases((1, 1))
    for i, p in enumerate(occ1):
        for j, q in enumerate(occ1):
            A[p, norb+q] = A[norb+q, p] = (ph_ab[i, j] - L[p] - L[norb+q]) / 2

    return A, L

def _dedup_rows(combined: np.typing.NDArray):
    """
    De-duplicate rows via lexsort.
    """
    order = np.lexsort(combined.T)
    sorted_arr = combined[order]
    is_new = np.empty(len(order), dtype=bool)
    is_new[0] = True
    is_new[1:] = np.any(sorted_arr[1:] != sorted_arr[:-1], axis=1)
    group_id_sorted = np.cumsum(is_new) - 1
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = group_id_sorted
    first_idx = order[is_new]
    return first_idx, inverse


def _compute_energy(
    Q: np.typing.NDArray,
    ecore: float,
    h_bp: np.typing.NDArray,
    g_bp: np.typing.NDArray,
    A: np.typing.NDArray,
    L: np.typing.NDArray,
    norb: int,
    tuple_chunk: int = 200_000,
    gid_batch: int = 4_000,
    round_decimals: int = 10,
    show_progress: bool = True,
) -> float:
    """
    Returns the energy with NumPy.

    Args:
    Q: (norb x n_occ) Occupied-orbital matrix of the rotated Slater determinant.
    ecore: Energy constant.
    h_bp: (norb x norb) One-body integrals back-propagated through u.
    g_bp: (norb,)*4 Two-body integrals back-propagated through u.
    A, L: Jastrow phase data.
    norb: Number of orbitals.
    tuple_chunk: Number of (p,q,r,s) tuples processed per pass.
    gid_batch: Number of unique phi vectors batched per call.
    round_decimals: Rounding applied to phi vectors before dedup
    show_progress: Show a progress bar
    """
    N = 2 * norb
    Qc = Q.conj()
    n_occ = Q.shape[1]

    def transition_batch(phi):
        d = np.exp(1j * phi)
        dQ = d[:, :, None] * Q[None, :, :]
        S = np.einsum('pi,bpj->bij', Qc, dQ)
        det = np.linalg.det(S)
        X = np.linalg.solve(S, np.broadcast_to(Qc.T, (phi.shape[0], n_occ, norb)))
        rho = dQ @ X
        return det, rho

    # Energy from one-body terms
    p1, q1 = np.meshgrid(np.arange(norb), np.arange(norb), indexing='ij')
    p1 = p1.ravel(); q1 = q1.ravel()
    Delta1 = np.zeros((norb * norb, N))
    Delta1[np.arange(norb * norb), p1] += 1
    Delta1[np.arange(norb * norb), q1] -= 1
    phi1 = -2.0 * (Delta1 @ A.T)
    const1 = np.exp(-1j * (np.einsum('bi,ij,bj->b', Delta1, A, Delta1) + Delta1 @ L))
    det_a1, rho_a1 = transition_batch(phi1[:, :norb])
    det_b1, _ = transition_batch(phi1[:, norb:])
    idx1 = np.arange(norb * norb)
    E1 = np.sum(2 * h_bp[p1, q1] * const1 * det_a1 * det_b1 * rho_a1[idx1, q1, p1])

    # Energy from two-body terms
    n4 = norb ** 4
    g_flat = g_bp.reshape(-1)
    p_all, q_all, r_all, s_all = np.unravel_index(np.arange(n4), (norb, norb, norb, norb))

    unique_phis = []
    key_to_gid = {}

    inv_a_same = np.empty(n4, dtype=np.int32)
    inv_b_same = np.empty(n4, dtype=np.int32)
    inv_a_opp = np.empty(n4, dtype=np.int32)
    inv_b_opp = np.empty(n4, dtype=np.int32)

    for start in tqdm(range(0, n4, tuple_chunk), desc="compute_energy: dedup", unit="chunk", disable=not show_progress):
        end = min(start + tuple_chunk, n4)
        p, q, r, s = p_all[start:end], q_all[start:end], r_all[start:end], s_all[start:end]

        phi_full_same = -2.0 * (A[:, p] + A[:, r] - A[:, q] - A[:, s]).T
        phi_full_opp = -2.0 * (A[:, p] - A[:, q] + A[:, norb + r] - A[:, norb + s]).T

        combined = np.round(np.concatenate([
            phi_full_same[:, :norb], phi_full_same[:, norb:],
            phi_full_opp[:, :norb], phi_full_opp[:, norb:],
        ], axis=0), round_decimals)

        first_idx, chunk_inverse = _dedup_rows(combined)

        local_to_global = np.empty(first_idx.shape[0], dtype=np.int32)
        for i, row_idx in enumerate(first_idx):
            key = combined[row_idx].tobytes()
            gid = key_to_gid.get(key)
            if gid is None:
                gid = len(unique_phis)
                key_to_gid[key] = gid
                unique_phis.append(combined[row_idx].copy())
            local_to_global[i] = gid

        global_ids = local_to_global[chunk_inverse]
        a_s, b_s, a_o, b_o = np.split(global_ids, 4)
        inv_a_same[start:end] = a_s
        inv_b_same[start:end] = b_s
        inv_a_opp[start:end] = a_o
        inv_b_opp[start:end] = b_o

    unique_phis = np.array(unique_phis)
    n_unique = unique_phis.shape[0]

    const_same_arr = np.empty(n4, dtype=np.complex128)
    const_opp_arr = np.empty(n4, dtype=np.complex128)
    for start in range(0, n4, tuple_chunk):
        end = min(start + tuple_chunk, n4)
        p, q, r, s = p_all[start:end], q_all[start:end], r_all[start:end], s_all[start:end]
        rows = np.arange(end - start)

        phi_full_same = -2.0 * (A[:, p] + A[:, r] - A[:, q] - A[:, s]).T
        const_same_arr[start:end] = np.exp(-1j * (
            -0.5 * (phi_full_same[rows, p] + phi_full_same[rows, r]
                    - phi_full_same[rows, q] - phi_full_same[rows, s])
            + (L[p] + L[r] - L[q] - L[s])
        ))
        phi_full_opp = -2.0 * (A[:, p] - A[:, q] + A[:, norb + r] - A[:, norb + s]).T
        const_opp_arr[start:end] = np.exp(-1j * (
            -0.5 * (phi_full_opp[rows, p] - phi_full_opp[rows, q]
                    + phi_full_opp[rows, norb + r] - phi_full_opp[rows, norb + s])
            + (L[p] - L[q] + L[norb + r] - L[norb + s])
        ))

    det_a_same_arr = np.empty(n4, dtype=np.complex128)
    rho_a_same_qp = np.empty(n4, dtype=np.complex128)
    rho_a_same_sr = np.empty(n4, dtype=np.complex128)
    rho_a_same_sp = np.empty(n4, dtype=np.complex128)
    rho_a_same_qr = np.empty(n4, dtype=np.complex128)
    det_b_same_arr = np.empty(n4, dtype=np.complex128)
    det_a_opp_arr = np.empty(n4, dtype=np.complex128)
    rho_a_opp_qp = np.empty(n4, dtype=np.complex128)
    det_b_opp_arr = np.empty(n4, dtype=np.complex128)
    rho_b_opp_sr = np.empty(n4, dtype=np.complex128)

    for g0 in tqdm(range(0, n_unique, gid_batch), desc="compute_energy: transitions", unit="batch", disable=not show_progress):
        g1 = min(g0 + gid_batch, n_unique)
        det_chunk, rho_chunk = transition_batch(unique_phis[g0:g1])

        mask = (inv_a_same >= g0) & (inv_a_same < g1)
        idxs = np.nonzero(mask)[0]
        if idxs.size:
            loc = inv_a_same[idxs] - g0
            p, q, r, s = p_all[idxs], q_all[idxs], r_all[idxs], s_all[idxs]
            det_a_same_arr[idxs] = det_chunk[loc]
            rho_a_same_qp[idxs] = rho_chunk[loc, q, p]
            rho_a_same_sr[idxs] = rho_chunk[loc, s, r]
            rho_a_same_sp[idxs] = rho_chunk[loc, s, p]
            rho_a_same_qr[idxs] = rho_chunk[loc, q, r]

        mask = (inv_b_same >= g0) & (inv_b_same < g1)
        idxs = np.nonzero(mask)[0]
        if idxs.size:
            loc = inv_b_same[idxs] - g0
            det_b_same_arr[idxs] = det_chunk[loc]

        mask = (inv_a_opp >= g0) & (inv_a_opp < g1)
        idxs = np.nonzero(mask)[0]
        if idxs.size:
            loc = inv_a_opp[idxs] - g0
            p, q = p_all[idxs], q_all[idxs]
            det_a_opp_arr[idxs] = det_chunk[loc]
            rho_a_opp_qp[idxs] = rho_chunk[loc, q, p]

        mask = (inv_b_opp >= g0) & (inv_b_opp < g1)
        idxs = np.nonzero(mask)[0]
        if idxs.size:
            loc = inv_b_opp[idxs] - g0
            r, s = r_all[idxs], s_all[idxs]
            det_b_opp_arr[idxs] = det_chunk[loc]
            rho_b_opp_sr[idxs] = rho_chunk[loc, s, r]

    wick = rho_a_same_qp * rho_a_same_sr - rho_a_same_sp * rho_a_same_qr
    term_same = g_flat * const_same_arr * det_a_same_arr * det_b_same_arr * wick
    term_opp = g_flat * const_opp_arr * (det_a_opp_arr * rho_a_opp_qp) * (det_b_opp_arr * rho_b_opp_sr)
    E2 = np.sum(term_same + term_opp)

    E = ecore + E1 + E2
    if abs(E.imag) > 1e-6:
        print(f"warning: Im(E) = {E.imag:.2e}")
    return float(E.real)
