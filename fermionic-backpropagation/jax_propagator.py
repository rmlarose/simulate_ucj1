"""
JAX-differentiable version of the UCJ backpropagation energy pipeline.
"""
import itertools

import jax
import jax.numpy as jnp

from ffsim.linalg.util import real_symmetrics_from_parameters_jax, unitary_from_parameters_jax

jax.config.update("jax_enable_x64", True)


def _resolve_interaction_pairs(norb, interaction_pairs):
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    if interaction_pairs is None:
        interaction_pairs = (None, None)
    pairs_aa, pairs_ab = interaction_pairs
    if pairs_aa is None:
        pairs_aa = triu_indices
    if pairs_ab is None:
        pairs_ab = triu_indices
    return pairs_aa, pairs_ab


def _ucj_arrays_from_parameters_jax(
    params, norb, n_reps, interaction_pairs=None, with_final_orbital_rotation=False
):
    """
    JAX-differentiable equivalent of `ffsim.UCJOpSpinBalanced.from_parameters`.
    """
    pairs_aa, pairs_ab = _resolve_interaction_pairs(norb, interaction_pairs)

    orbital_rotations = []
    diag_coulomb_mats = []
    index = 0
    for _ in range(n_reps):
        n_rot_params = norb ** 2
        orbital_rotation = unitary_from_parameters_jax(params[index:index + n_rot_params], dim=norb)
        index += n_rot_params

        mats = []
        for pairs in (pairs_aa, pairs_ab):
            n_pair_params = len(pairs)
            mat = real_symmetrics_from_parameters_jax(
                params[index:index + n_pair_params], dim=norb, n_mats=1, triu_indices=pairs
            )[0]
            index += n_pair_params
            mats.append(mat)

        orbital_rotations.append(orbital_rotation)
        diag_coulomb_mats.append(jnp.stack(mats))

    final_orbital_rotation = None
    if with_final_orbital_rotation:
        final_orbital_rotation = unitary_from_parameters_jax(params[index:], dim=norb)

    return jnp.stack(orbital_rotations), jnp.stack(diag_coulomb_mats), final_orbital_rotation


def _propagate_through_jastrow_jax(same, diff, norb):
    """
    Closed-form equivalent of `propagator._propagate_through_jastrow`.
    """
    N = 2 * norb
    same_offdiag = same - jnp.diag(jnp.diag(same))

    A = jnp.zeros((N, N))
    A = A.at[:norb, :norb].set(same_offdiag / 2)
    A = A.at[norb:, norb:].set(same_offdiag / 2)
    A = A.at[:norb, norb:].set(diff / 2)
    A = A.at[norb:, :norb].set(diff.T / 2)

    L = jnp.concatenate([jnp.diag(same) / 2, jnp.diag(same) / 2])
    return A, L


def _propagate_through_orbital_rotations_jax(h1e, h2e, u):
    """jnp port of `propagator._propagate_through_orbital_rotations`."""
    h_bp = u.conj().T @ h1e @ u
    g_bp = jnp.einsum('pi,qj,pqrs,rk,sl->ijkl', u.conj(), u, h2e, u.conj(), u, optimize=True)
    return h_bp, g_bp


def _compute_energy_jax(Q, ecore, h_bp, g_bp, A, L, norb, chunk_size=None):
    """
    Port of `propagator._compute_energy`.
    """
    N = 2 * norb
    Qc = jnp.conj(Q)
    n_occ = Q.shape[1]

    def transition_batch(phi):
        d = jnp.exp(1j * phi)
        dQ = d[:, :, None] * Q[None, :, :]
        S = jnp.einsum('pi,bpj->bij', Qc, dQ)
        det = jnp.linalg.det(S)
        X = jnp.linalg.solve(S, jnp.broadcast_to(Qc.T, (phi.shape[0], n_occ, norb)))
        rho = dQ @ X
        return det, rho

    p1, q1 = jnp.meshgrid(jnp.arange(norb), jnp.arange(norb), indexing='ij')
    p1 = p1.ravel(); q1 = q1.ravel()
    idx1 = jnp.arange(norb * norb)
    Delta1 = (jnp.zeros((norb * norb, N))
              .at[idx1, p1].add(1)
              .at[idx1, q1].add(-1))
    phi1 = -2.0 * (Delta1 @ A.T)
    const1 = jnp.exp(-1j * (jnp.einsum('bi,ij,bj->b', Delta1, A, Delta1) + Delta1 @ L))
    det_a1, rho_a1 = transition_batch(phi1[:, :norb])
    det_b1, _ = transition_batch(phi1[:, norb:])
    E1 = jnp.sum(2 * h_bp[p1, q1] * const1 * det_a1 * det_b1 * rho_a1[idx1, q1, p1])

    n4 = norb ** 4
    g_flat = g_bp.reshape(-1)

    def two_body_chunk(idx):
        p, q, r, s = jnp.unravel_index(idx, (norb, norb, norb, norb))
        g = g_flat[idx]
        rows = jnp.arange(idx.shape[0])

        Delta_same = (jnp.zeros((idx.shape[0], N))
                      .at[rows, p].add(1).at[rows, r].add(1)
                      .at[rows, s].add(-1).at[rows, q].add(-1))
        phi_full_same = -2.0 * (Delta_same @ A.T)
        const_same = jnp.exp(-1j * (jnp.einsum('bi,ij,bj->b', Delta_same, A, Delta_same) + Delta_same @ L))

        Delta_opp = (jnp.zeros((idx.shape[0], N))
                     .at[rows, p].add(1).at[rows, q].add(-1)
                     .at[rows, norb + r].add(1).at[rows, norb + s].add(-1))
        phi_full_opp = -2.0 * (Delta_opp @ A.T)
        const_opp = jnp.exp(-1j * (jnp.einsum('bi,ij,bj->b', Delta_opp, A, Delta_opp) + Delta_opp @ L))

        det_a_same, rho_a_same = transition_batch(phi_full_same[:, :norb])
        det_b_same, _ = transition_batch(phi_full_same[:, norb:])
        wick = (rho_a_same[rows, q, p] * rho_a_same[rows, s, r]
                - rho_a_same[rows, s, p] * rho_a_same[rows, q, r])
        term_same = g * const_same * det_a_same * det_b_same * wick

        det_a_opp, rho_a_opp = transition_batch(phi_full_opp[:, :norb])
        det_b_opp, rho_b_opp = transition_batch(phi_full_opp[:, norb:])
        term_opp = g * const_opp * (det_a_opp * rho_a_opp[rows, q, p]) * (det_b_opp * rho_b_opp[rows, s, r])

        return jnp.sum(term_same + term_opp)

    if chunk_size is None or chunk_size >= n4:
        E2 = two_body_chunk(jnp.arange(n4))
    else:
        if n4 % chunk_size != 0:
            raise ValueError(f"chunk_size ({chunk_size}) must evenly divide norb**4 ({n4}).")
        idx_chunks = jnp.arange(n4).reshape(-1, chunk_size)
        # Checkpointing so that we can fit the computation on a GPU with limited memory
        E2 = jnp.sum(jax.lax.map(jax.checkpoint(two_body_chunk), idx_chunks))

    E = ecore + E1 + E2
    return jnp.real(E)


def energy_and_grad_jax(
    params,
    h1e,
    h2e,
    ecore,
    nelec,
    norb,
    n_reps=1,
    interaction_pairs=None,
    with_final_orbital_rotation=False,
    chunk_size=None,
):
    """
    Energy and gradient (w.r.t. `params`) of the UCJ backpropagation energy.
    """
    def fun(params):
        orbital_rotations, diag_coulomb_mats, final_orbital_rotation = _ucj_arrays_from_parameters_jax(
            params, norb, n_reps, interaction_pairs, with_final_orbital_rotation
        )
        W = orbital_rotations[0]
        Wf = final_orbital_rotation
        u = W if Wf is None else Wf @ W

        h_bp, g_bp = _propagate_through_orbital_rotations_jax(h1e, h2e, u)
        A, L = _propagate_through_jastrow_jax(diag_coulomb_mats[0][0], diag_coulomb_mats[0][1], norb)
        Q = W.conj().T[:, :nelec[0]]
        return _compute_energy_jax(Q, ecore, h_bp, g_bp, A, L, norb, chunk_size=chunk_size)

    return jax.value_and_grad(fun)(params)
