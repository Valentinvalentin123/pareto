"""Toolbox SCF Model A — vertical slice.

4 of 5 designed optimizations (damping deferred):
  1. Core-H initial guess (diagonalize H_core before SCF loop)
  2. while_loop with early-exit on |dE| < tol
  4. k=n_occ in F' diagonalization (only occupied orbitals)
  5. Fock+energy carried inside the loop (no post-loop coulomb_jk)

Skipped in this slice:
  3. Damping (needs ux.select on matrices; verify in follow-up)
  6. DIIS (Model B/C)
"""

from __future__ import annotations

import uniqx as ux
from uniqx.core import types
from uniqx.core.tracing import to_module
from uniqx.domains.chemistry.basis import BasisInfo
from uniqx.domains.chemistry.kernels import coulomb_jk, kinetic, nuclear, overlap


def build_module_A(atoms, info: BasisInfo, max_iter: int = 50, tol: float = 1e-7):
    """Model A traced module returning the SCF total energy (scalar f64)."""
    n = info.n_basis
    n2 = n * n
    n_occ = info.n_electrons // 2
    n_atoms = info.n_atoms
    np_ = info.n_prim
    lmax = info.lmax
    e_nuc = info.nuclear_repulsion
    T = types.tensor

    @to_module
    def _scf(exps_in, coeffs_in, centers_in, ang_in, atom_coords_in, charges_in):
        exps = ux.reshape(exps_in, shape=[n, np_], result_type=T("f64", [n, np_]))
        coeffs = ux.reshape(coeffs_in, shape=[n, np_], result_type=T("f64", [n, np_]))
        centers = ux.reshape(centers_in, shape=[n, 3], result_type=T("f64", [n, 3]))
        ang = ux.reshape(ang_in, shape=[n, 3], result_type=T("f64", [n, 3]))
        atom_coords = ux.reshape(
            atom_coords_in, shape=[n_atoms, 3], result_type=T("f64", [n_atoms, 3])
        )

        # One-electron integrals (same as nmr_full._scf_core)
        s_flat = overlap(exps, coeffs, centers, ang, n_basis=n, n_prim=np_, lmax=lmax)
        t_flat = kinetic(exps, coeffs, centers, ang, n_basis=n, n_prim=np_, lmax=lmax)
        v_flat = nuclear(
            exps, coeffs, centers, ang, atom_coords, charges_in,
            n_basis=n, n_atoms=n_atoms, n_prim=np_, lmax=lmax,
        )
        hcore_flat = t_flat + v_flat
        s_mat = ux.reshape(s_flat, shape=[n, n], result_type=T("f64", [n, n]))
        hcore_mat = ux.reshape(hcore_flat, shape=[n, n], result_type=T("f64", [n, n]))

        # Canonical orthogonalizer X = S^{-1/2}, drop near-zero eigvals
        eigvals, eigvecs = ux.eigs(s_mat, hermitian=True, k=n, which="smallest")
        keep = ux.compare(eigvals, 1e-7, direction=">")
        eigvals_safe = ux.max(eigvals, 1e-12)
        inv_sqrt_raw = ux.div(1.0, ux.sqrt(eigvals_safe))
        zero_vec = inv_sqrt_raw * 0.0
        inv_sqrt = ux.select(keep, inv_sqrt_raw, zero_vec)
        scaled = ux.einsum(eigvecs, inv_sqrt, subscripts="ij,j->ij", result_type=T("f64", [n, n]))
        eigvecs_t = ux.transpose(eigvecs, [1, 0], result_type=T("f64", [n, n]))
        x_mat = ux.matmul(scaled, eigvecs_t)
        x_t = ux.transpose(x_mat, [1, 0], result_type=T("f64", [n, n]))

        # OPTIM 1: Core-H initial guess (diagonalize H_core, build initial density)
        h_prime = ux.matmul(ux.matmul(x_t, hcore_mat), x_mat)
        _eps0, cp0 = ux.eigs(h_prime, hermitian=True, k=n_occ, which="smallest")
        c0 = ux.matmul(x_mat, cp0)
        c0_t = ux.transpose(c0, [1, 0], result_type=T("f64", [n_occ, n]))
        p_init = ux.matmul(c0, c0_t) * 2.0
        p_init_flat = ux.reshape(p_init, shape=[n2], result_type=T("f64", [n2]))

        # Initial Fock = hcore (placeholder; first iter computes real Fock)
        f_init_flat = ux.reshape(hcore_mat, shape=[n2], result_type=T("f64", [n2]))

        # Carry layout: [P (n2) | F (n2) | E (1) | dE (1) | counter (1)]
        z1 = ux.slice(
            p_init_flat, start_indices=[0], limit_indices=[1], result_type=T("f64", [1])
        ) * 0.0
        carry_init = ux.concatenate(
            p_init_flat,           # P
            f_init_flat,           # F (placeholder)
            z1,                    # E_prev
            z1 + 1e10,             # dE (large so we don't exit at iter 0)
            z1 + float(max_iter),  # counter (decrements; 0 means stop)
            axis=0,
            result_type=T("f64", [2 * n2 + 3]),
        )

        OFF_P = 0
        OFF_F = n2
        OFF_E = 2 * n2
        OFF_DE = 2 * n2 + 1
        OFF_CT = 2 * n2 + 2

        # OPTIM 2: while_loop with early-exit on |dE| < tol OR counter == 0
        def scf_cond(carry):
            de = ux.slice(
                carry, start_indices=[OFF_DE], limit_indices=[OFF_DE + 1],
                result_type=T("f64", [1]),
            )
            ct = ux.slice(
                carry, start_indices=[OFF_CT], limit_indices=[OFF_CT + 1],
                result_type=T("f64", [1]),
            )
            de_s = ux.reshape(de, shape=[], result_type=T("f64", []))
            ct_s = ux.reshape(ct, shape=[], result_type=T("f64", []))
            # Continue if ct > 0 AND |dE| > tol. Encode as min(ct, |dE|/tol) > 0.5.
            scaled_de = ux.abs(de_s) / tol
            indicator = ux.min(ct_s, scaled_de)
            return indicator > 0.5  # noqa: PLR2004

        def scf_body(carry):
            pf = ux.slice(
                carry, start_indices=[OFF_P], limit_indices=[OFF_P + n2],
                result_type=T("f64", [n2]),
            )
            e_prev = ux.slice(
                carry, start_indices=[OFF_E], limit_indices=[OFF_E + 1],
                result_type=T("f64", [1]),
            )
            ct = ux.slice(
                carry, start_indices=[OFF_CT], limit_indices=[OFF_CT + 1],
                result_type=T("f64", [1]),
            )

            density = ux.reshape(pf, shape=[n, n], result_type=T("f64", [n, n]))

            # Build Fock
            j_flat, k_flat = coulomb_jk(
                exps, coeffs, centers, ang, density,
                n_basis=n, n_prim=np_, lmax=lmax,
            )
            j_mat = ux.reshape(j_flat, shape=[n, n], result_type=T("f64", [n, n]))
            k_mat = ux.reshape(k_flat, shape=[n, n], result_type=T("f64", [n, n]))
            fock = hcore_mat + j_mat - k_mat * 0.5

            # Diagonalize F' = X^T F X
            fp = ux.matmul(ux.matmul(x_t, fock), x_mat)
            # OPTIM 4: k=n_occ (only occupied orbitals needed for density rebuild)
            _eps, cp = ux.eigs(fp, hermitian=True, k=n_occ, which="smallest")
            c_occ = ux.matmul(x_mat, cp)
            c_occ_t = ux.transpose(c_occ, [1, 0], result_type=T("f64", [n_occ, n]))
            new_p = ux.matmul(c_occ, c_occ_t) * 2.0

            # OPTIM 5: Energy computed inside loop, carried out (no post-loop coulomb_jk)
            ph = ux.matmul(new_p, hcore_mat + fock)
            eel = ux.trace(ph, result_type=T("f64", [])) * 0.5
            et = eel + e_nuc
            et_flat = ux.reshape(et, shape=[1], result_type=T("f64", [1]))
            de_new = et_flat - e_prev
            ct_new = ct - 1.0

            new_p_flat = ux.reshape(new_p, shape=[n2], result_type=T("f64", [n2]))
            fock_flat = ux.reshape(fock, shape=[n2], result_type=T("f64", [n2]))

            return ux.concatenate(
                new_p_flat, fock_flat, et_flat, de_new, ct_new,
                axis=0,
                result_type=T("f64", [2 * n2 + 3]),
            )

        conv = ux.while_loop(scf_cond, scf_body, carry_init)
        e_final = ux.slice(
            conv, start_indices=[OFF_E], limit_indices=[OFF_E + 1],
            result_type=T("f64", [1]),
        )
        return ux.reshape(e_final, shape=[], result_type=T("f64", []))

    return _scf(
        info.exps_flat, info.coeffs_flat, info.centers_flat, info.ang_flat,
        info.atom_coords_flat, info.charges_flat,
    )
