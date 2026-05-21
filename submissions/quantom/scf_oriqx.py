"""
Level 1 ORIQX implementation of the RHF-SCF loop.

The classical Python integral engine (build_basis, build_overlap_kinetic,
build_nuclear, build_eri_tensor, nuclear_repulsion) is kept verbatim from
the baseline. Only the SCF iteration is rewritten as a single traced
ORIQX module — so the entire fixed-point loop compiles into one IR
graph and executes with one gateway submit per geometry.

CRITICAL GOTCHA (learned the hard way):
    Tracer inputs and ops.const MUST be Python list-of-lists. A
    numpy.ndarray is silently interpreted as a scalar (rank-0 tensor),
    which then propagates as scalar through every subsequent op until
    you hit a matmul and get a cryptic "scalar with tensor" error.
    Always call `.tolist()` on numpy arrays before passing them in.

Architecture rationale (see decomposition_map.md for the full table):
- H, X, g, e_nuc are computed once per geometry on CPU (cheap; depends
  on basis-function placement which changes every MD step).
- The 100-iteration SCF loop is the expensive part and the part that
  needs heterogeneous dispatch — that goes into the traced module.
- We pass the four matrices as runtime_inputs so the SAME compiled IR
  is reused across all 18 force-finite-difference submits per MD step
  and across all MD steps.
- D0 (zero density) and constant scaling tensors (half_nn = 0.5*ones,
  two_nn = 2.0*ones) are also runtime inputs to dodge ops.const's
  scalar-only behaviour. They never change between submits.
"""

from uniqx import ir, ops, tracing


def f64_tensor(*shape):
    return ir.Type(tensor=ir.TensorType(dtype="f64", shape=list(shape)))


def f64_scalar():
    return ir.Type(scalar=ir.ScalarType(dtype="f64"))


def make_scf_module(n_basis: int, n_occ: int, max_iter: int = 50):
    """
    Build a traced SCF module specialised for a fixed (n_basis, n_occ).

    Returns:
        A `@tracing.to_module`-decorated function. Call it once with
        properly shaped placeholder list-of-lists inputs to produce the
        IR `Module` object; then `submit(module, runtime_inputs=[...])`
        for each geometry.
    """
    t_nn = f64_tensor(n_basis, n_basis)
    t_no = f64_tensor(n_basis, n_occ)
    t_on = f64_tensor(n_occ, n_basis)
    t_scalar = f64_scalar()

    @tracing.to_module(name=f"rhf_scf_n{n_basis}")
    def rhf_scf(H, X, g, e_nuc, D0, half_nn, two_nn):
        def body(i, D):
            # J, K = Coulomb / Exchange contractions
            J = ops.einsum(g, D, subscripts="pqrs,rs->pq", result_type=t_nn)
            K = ops.einsum(g, D, subscripts="prqs,rs->pq", result_type=t_nn)
            # Fock matrix: F = H + J - 0.5 K
            F = ops.add(H, ops.sub(J, ops.mul(half_nn, K)))
            # Orthogonal Fock: F' = X^T F X
            XT = ops.transpose(X, permutation=(1, 0), result_type=t_nn)
            Fp = ops.matmul(ops.matmul(XT, F), X)
            # Diagonalise
            eps, Cp = ops.eigs(
                Fp, k=n_basis, hermitian=True, which="smallest"
            )
            C = ops.matmul(X, Cp)
            # Density: D = 2 C_occ C_occ^T
            C_occ = ops.slice(
                C,
                start_indices=[0, 0],
                limit_indices=[n_basis, n_occ],
                result_type=t_no,
            )
            CocT = ops.transpose(
                C_occ, permutation=(1, 0), result_type=t_on
            )
            D_new = ops.mul(two_nn, ops.matmul(C_occ, CocT))
            return D_new

        D_final = ops.fori_loop(0, max_iter, body, D0)

        # Final energy with converged D:
        #   E = 0.5 Tr[D (H+F)] + e_nuc
        # We return 2*0.5*Tr = Tr (since we cannot easily multiply by a
        # scalar 0.5 inside the IR), and the caller divides by 2.
        # See `parse_energy()` below.
        J = ops.einsum(g, D_final, subscripts="pqrs,rs->pq", result_type=t_nn)
        K = ops.einsum(g, D_final, subscripts="prqs,rs->pq", result_type=t_nn)
        F = ops.add(H, ops.sub(J, ops.mul(half_nn, K)))
        HF = ops.add(H, F)
        DHF = ops.mul(D_final, HF)
        tr = ops.reduce_sum(DHF, axis=None, result_type=t_scalar)
        # tr_plus_2e_nuc / 2 == 0.5*tr + e_nuc  (applied client-side)
        e_raw = ops.add(tr, ops.add(e_nuc, e_nuc))
        return e_raw

    return rhf_scf


def parse_energy(e_raw: float) -> float:
    """Convert the raw module output back to physical E_total in Hartree.

    The IR returns `tr(D*(H+F)) + 2*e_nuc` because scalar*tensor
    multiplication inside the trace caused shape problems. Divide by 2
    here to recover `0.5*tr(D*(H+F)) + e_nuc` = the standard total.
    """
    return 0.5 * e_raw


def runtime_inputs_for(n_basis, H, X, g, e_nuc):
    """Pack H, X, g, e_nuc plus the fixed constants as runtime_inputs.

    H, X, g are numpy arrays; we convert to list-of-lists because the
    tracer rejects ndarrays.
    """
    import numpy as np
    return [
        H.tolist(),
        X.tolist(),
        g.tolist(),
        float(e_nuc),
        np.zeros((n_basis, n_basis)).tolist(),
        np.full((n_basis, n_basis), 0.5).tolist(),
        np.full((n_basis, n_basis), 2.0).tolist(),
    ]
