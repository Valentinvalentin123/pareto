"""
VQE for H2 (STO-3G, 2-qubit tapered Hamiltonian) on ORIQX.

The Pareto twin to our SCF track: the *same* "find an eigenstate of a
Hermitian operator" problem, but expressed in a Pauli basis that opens
the QPU as an execution option.

Theory
------
After Z2 x Z2 tapering (parity + spin conservation) of the
Jordan-Wigner-mapped H2/STO-3G Hamiltonian, the 4-qubit problem reduces
to a 2-qubit operator on a 4-dimensional Hilbert space:

    H = c0 I  +  c1 Z_0  +  c2 Z_1  +  c3 Z_0 Z_1  +  c4 X_0 X_1

At equilibrium R = 0.735 A, the coefficients are (atomic units,
nuclear repulsion included in c0; source: arXiv:1704.05018 Table I,
matches OpenFermion's tapered Hamiltonian for the same geometry):

    c0 = -1.0523732
    c1 = +0.39793742
    c2 = -0.39793742
    c3 = -0.0112801
    c4 = +0.18093120

The exact ground-state energy is E_0 = -1.13728 Ha (analytical
eigendecomposition of the 4x4 matrix).

Ansatz
------
One-parameter rotation with the simple X_0 X_1 generator:

    |psi(theta)> = exp(-i theta X_0 X_1 / 2) |HF>      with |HF> = |01>
    |psi(theta)> = cos(theta/2) |01> - i sin(theta/2) |10>

KNOWN LIMITATION: this real-symmetric generator produces an imaginary
mixing phase, so the off-diagonal H_{01,10} = 0.181 Ha matrix element
contributes ZERO to <psi(theta)|H|psi(theta)>. The grid-scan minimum
collapses to theta=0 (the Hartree-Fock state, E = -1.117 Ha total).

The correlation energy (~0.020 Ha for H2/STO-3G) is recovered only
by a generator that produces real-valued mixing, e.g. the proper
UCCSD generator G = (X_0 Y_1 - Y_0 X_1)/4 which is purely imaginary
Hermitian. That requires complex-dtype support in uniqx — verified
separately and noted in the submission notebook.

For the Pareto/dispatch story we use this XX ansatz: it is honest
(the FCI gap = the correlation energy, an interpretable quantity),
fast to trace, and still demonstrates the QPU-vs-classical routing
because expv + expect are the relevant primitives.

Trace structure
---------------
A *single* traced module `vqe_h2_energy_module` takes one runtime
input — theta — plus the precomputed H, generator (X0X1/2), and HF
state.  Per gradient step we submit it 1-3 times depending on
optimiser strategy.

For the hackathon submission we grid-scan theta over [-pi, pi] in
n_grid points (one submit per grid point), find the minimum, and
compare to the exact eigenvalue of H (`ops.eigs` reference).

CRITICAL gotcha (same as scf_oriqx): all tensor inputs must be
Python list-of-lists, never numpy.ndarray.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from uniqx import ir, ops, tracing


# ---------------------------------------------------------------------
# Pauli helpers (classical, just for matrix construction)
# ---------------------------------------------------------------------

I2 = np.eye(2)
PX = np.array([[0.0, 1.0], [1.0, 0.0]])
PY = np.array([[0.0, -1.0j], [1.0j, 0.0]])   # only used for sanity, not the H
PZ = np.array([[1.0, 0.0], [0.0, -1.0]])

_PAULI = {"I": I2, "X": PX, "Y": PY, "Z": PZ}


def pauli_string(s: str) -> np.ndarray:
    """Build the tensor product P_{s[0]} (x) P_{s[1]} (x) ... ."""
    mats = [_PAULI[c] for c in s]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


# ---------------------------------------------------------------------
# H2 / STO-3G tapered Hamiltonian and ansatz generator
# ---------------------------------------------------------------------

# Coefficients at R = 0.735 A. c0 includes the nuclear repulsion
# +1/R_bohr = +0.7196 Ha so that comparison with the literature total
# energy is direct. Verified: eigh gives -1.13768 Ha vs literature -1.13728
# (sub-millihartree, well within tapering / coefficient rounding).
H2_COEFFS_EQ = {
    "II": -1.0523732 + 0.7196,   # -0.3328 (total, incl. nuclear repulsion)
    "IZ": +0.39793742,
    "ZI": -0.39793742,
    "ZZ": -0.0112801,
    "XX": +0.18093120,
}


def build_h2_hamiltonian(coeffs: dict[str, float] | None = None) -> np.ndarray:
    """4x4 real symmetric matrix for the tapered H2 Hamiltonian."""
    if coeffs is None:
        coeffs = H2_COEFFS_EQ
    H = np.zeros((4, 4))
    for s, c in coeffs.items():
        H += c * pauli_string(s).real     # H is real symmetric in this basis
    return H


def build_uccsd_generator() -> np.ndarray:
    """Generator G = X_0 X_1 / 2 for the 1-parameter H2 ansatz.

    exp(-i theta G) |HF> = cos(theta/2)|01> - i sin(theta/2)|10>.
    """
    return 0.5 * pauli_string("XX").real


def hf_state() -> list[float]:
    """|HF> = |01> in (|00>, |01>, |10>, |11>) ordering."""
    return [0.0, 1.0, 0.0, 0.0]


def exact_ground_state_energy(H: np.ndarray | None = None) -> float:
    """Classical reference: lowest eigenvalue of H (4x4 dense eigh)."""
    if H is None:
        H = build_h2_hamiltonian()
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


def hf_energy(H: np.ndarray | None = None) -> float:
    """Hartree-Fock expectation value <HF|H|HF> where |HF> = |01>.

    This is what the 1-parameter XX ansatz converges to (the imaginary
    mixing phase cancels the off-diagonal coupling). The gap to
    `exact_ground_state_energy()` is the correlation energy.
    """
    if H is None:
        H = build_h2_hamiltonian()
    hf = np.array(hf_state())
    return float(hf @ H @ hf)


# ---------------------------------------------------------------------
# Traced ORIQX module
# ---------------------------------------------------------------------

def make_vqe_h2_module():
    """Build the @tracing.to_module-decorated function.

    Returns a function that, when called once with placeholder inputs,
    produces an IR `Module` ready to submit. Subsequent submits reuse
    the same compiled module with different `theta` runtime inputs.
    """
    t_44 = ir.Type(tensor=ir.TensorType(dtype="f64", shape=[4, 4]))
    t_4 = ir.Type(tensor=ir.TensorType(dtype="f64", shape=[4]))
    t_scalar = ir.Type(scalar=ir.ScalarType(dtype="f64"))

    @tracing.to_module(name="vqe_h2")
    def vqe_h2_energy(H, G, hf, theta):
        # Prepare the variational state |psi(theta)> = exp(-i theta G) |HF>.
        # ops.expv computes exp(-i A t) v internally — `hermitian=True`
        # tells the planner that A is Hermitian and enables faster
        # quantum / Lanczos lowerings.
        psi = ops.expv(G, hf, theta, hermitian=True, precision=1e-8)
        # <psi|H|psi>. max_shots gates the QPU expectation cost; the
        # planner falls back to exact statevector on CPU/GPU.
        e = ops.expect(H, psi, max_shots=2048, stochastic_ok=True)
        return e

    return vqe_h2_energy


# ---------------------------------------------------------------------
# Driver — grid scan over theta
# ---------------------------------------------------------------------

@dataclass
class VqeResult:
    thetas: np.ndarray
    energies: np.ndarray
    theta_opt: float
    e_opt: float
    e_exact: float

    @property
    def gap(self) -> float:
        return self.e_opt - self.e_exact


def runtime_inputs(theta: float):
    """Pack the four inputs for one submit. H/G/hf are fixed across
    submits — we send them every time because the SDK's runtime_inputs
    interface does not yet support partial-input updates.
    """
    H = build_h2_hamiltonian().tolist()
    G = build_uccsd_generator().tolist()
    hf = hf_state()
    return [H, G, hf, float(theta)]


def grid_scan(
    submit_fn,
    n_grid: int = 41,
    theta_range: tuple[float, float] = (-math.pi, math.pi),
) -> VqeResult:
    """Submit one job per grid point, collect energies, find min.

    `submit_fn(theta) -> float` is the closure that talks to the gateway.
    Tested locally by passing a classical reference function instead.
    """
    thetas = np.linspace(theta_range[0], theta_range[1], n_grid)
    energies = np.array([submit_fn(t) for t in thetas])
    i_min = int(np.argmin(energies))
    e_exact = exact_ground_state_energy()
    return VqeResult(
        thetas=thetas,
        energies=energies,
        theta_opt=float(thetas[i_min]),
        e_opt=float(energies[i_min]),
        e_exact=e_exact,
    )


def classical_energy_reference(theta: float) -> float:
    """Pure NumPy reference: exp(-i theta G) |HF> then <psi|H|psi>.

    Used to validate the ORIQX result and to debug offline before the
    gateway is reachable.
    """
    H = build_h2_hamiltonian()
    G = build_uccsd_generator()
    hf = np.array(hf_state(), dtype=complex)
    # exp(-i theta G) = cos(theta) I - i sin(theta) (G/||G||) ... but
    # easier: G is real symmetric, use full eigendecomposition.
    # For a 4x4 it doesn't matter.
    from scipy.linalg import expm
    U = expm(-1j * theta * G)
    psi = U @ hf
    e = np.real(np.conj(psi) @ H @ psi)
    return float(e)


# ---------------------------------------------------------------------
# Quick sanity check when run as `python vqe_h2_oriqx.py`
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=== H2 / STO-3G tapered Hamiltonian (4x4) ===")
    H = build_h2_hamiltonian()
    print(H)
    print()
    print(f"Exact ground-state energy: {exact_ground_state_energy(H):+.6f} Ha")
    print(f"Literature (R=0.735 A):    -1.13728 Ha")
    print()

    print("=== Classical VQE landscape (no gateway) ===")
    res = grid_scan(classical_energy_reference, n_grid=21)
    for t, e in zip(res.thetas, res.energies):
        marker = "  <-- min" if abs(e - res.e_opt) < 1e-12 else ""
        print(f"  theta = {t:+.4f}   E = {e:+.6f} Ha{marker}")
    print()
    print(f"theta_opt = {res.theta_opt:+.4f}")
    print(f"E_opt     = {res.e_opt:+.6f} Ha")
    print(f"E_exact   = {res.e_exact:+.6f} Ha")
    print(f"gap       = {res.gap:+.2e} Ha")
