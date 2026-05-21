"""
Thermal ensemble of H2 in STO-3G via ORIQX-routed eigendecomposition.

The third leg of the Custom Track submission, complementing RHF-SCF
(`scf_oriqx.py`) and VQE (`vqe_h2_oriqx.py`).

Physics
-------
For a quantum Hamiltonian H with discrete spectrum {E_n}, the canonical
partition function at inverse temperature beta = 1/(k_B T) is

    Z(beta) = Tr[exp(-beta H)] = sum_n exp(-beta E_n).

From Z follows the full thermodynamics of the electronic ensemble:

    U(beta) = <H>_beta    = sum_n E_n exp(-beta E_n) / Z
    F(beta) = -ln(Z) / beta                          (Helmholtz free energy)
    S(beta) = beta (U - F)                           (entropy)
    C_v(beta) = beta^2 (<H^2>_beta - <H>_beta^2)     (heat capacity)

At T -> 0 only the ground state is populated, U -> E_0 = E_FCI = -1.138 Ha,
S -> 0. At T -> infinity all four states are equally populated, U ->
mean(spectrum), S -> ln(4) = 1.386 (max entropy). In between, the
Schottky anomaly in C_v marks where the thermal energy crosses the gap.

For H2/STO-3G the gap is ~0.6 Ha (~19000 K) — far above any temperature
relevant to chemistry, but exactly the regime where the partition
function is non-trivial and our algorithm is exercised meaningfully.

Algorithm
---------
A single ORIQX-traced module performs the only expensive step
(partial diagonalisation of the 4x4 Hamiltonian via `ops.eigs`).
Everything thermodynamic is built classically from the resulting
eigenvalues. The Pareto observation is that `ops.eigs` exposes a
*different* set of QPU-friendly backends than `ops.expv` + `ops.expect`
(used in VQE) — the same molecule routes to different hardware
depending on which operator graph the planner sees.

Algorithm vs the Variational Twin
---------------------------------
VQE searches for a single eigenvalue (the ground state) by minimising
<psi(theta)|H|psi(theta)> over an ansatz family. The thermal protocol
asks for the *entire spectrum* and then weighs each level by the
Boltzmann factor exp(-beta E). The two views are dual:

  - VQE finds the ground state by killing all amplitude on excited
    states through optimisation.
  - The thermal ensemble *uses* the excited-state amplitudes to compute
    finite-T observables.

Both rely on the same Hamiltonian; the algorithmic shape differs in
which ORIQX primitives drive the work.

CRITICAL gotcha
---------------
All tensor inputs must be Python list-of-lists, never numpy.ndarray.
The tracer interprets ndarrays as rank-0 scalars.  See scf_oriqx.py
for the full rant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from uniqx import ir, ops, tracing

from vqe_h2_oriqx import build_h2_hamiltonian, exact_ground_state_energy


# ---------------------------------------------------------------------
# Traced module — just the spectrum
# ---------------------------------------------------------------------

def make_h2_spectrum_module(n_states: int = 4):
    """Build a module that returns the lowest `n_states` eigenvalues of H.

    For H2 in the Z2xZ2-tapered basis, `n_states = 4` is the full
    spectrum (dim = 4). For larger problems, we'd restrict to the lowest
    chemically relevant states.
    """
    t_eigvals = ir.Type(tensor=ir.TensorType(dtype="f64", shape=[n_states]))
    # eigvecs we don't need for thermodynamics — Z, U, F, S depend only
    # on eigenvalues. We ignore the eigenvectors that ops.eigs returns.

    @tracing.to_module(name="h2_thermal_spectrum")
    def h2_spectrum(H):
        eigvals, _eigvecs = ops.eigs(
            H, k=n_states, hermitian=True, which="smallest"
        )
        return eigvals

    return h2_spectrum


# ---------------------------------------------------------------------
# Classical thermodynamic assembly
# ---------------------------------------------------------------------

@dataclass
class ThermoState:
    """Snapshot of thermodynamic state at one temperature."""
    beta: float                # 1/(k_B T) in Hartree^-1
    T: float                   # K, for human consumption
    Z: float                   # partition function
    U: float                   # internal energy <H>
    F: float                   # free energy
    S: float                   # entropy (dimensionless = k_B units)
    Cv: float                  # heat capacity (Hartree / Hartree = dimensionless)
    populations: np.ndarray    # Boltzmann population of each level


# Boltzmann constant in atomic units
K_B_AU = 3.166811e-6           # Hartree / K


def thermo_from_spectrum(eigvals: np.ndarray, beta: float) -> ThermoState:
    """Classical statistical mechanics from a discrete spectrum.

    All quantities are dimensionless / atomic units.
    """
    eigvals = np.asarray(eigvals, dtype=float)
    # Shift for numerical stability (subtract ground state)
    shifted = eigvals - eigvals.min()
    weights = np.exp(-beta * shifted)
    Z_shifted = weights.sum()
    Z = Z_shifted * math.exp(-beta * eigvals.min())     # full Z
    populations = weights / Z_shifted
    U = float(np.sum(eigvals * populations))
    H2 = float(np.sum(eigvals**2 * populations))
    F = float(eigvals.min() - math.log(Z_shifted) / beta)
    S = beta * (U - F)
    Cv = beta * beta * (H2 - U * U)
    return ThermoState(
        beta=float(beta),
        T=float(1.0 / (beta * K_B_AU)),
        Z=float(Z),
        U=U,
        F=F,
        S=float(S),
        Cv=float(Cv),
        populations=populations,
    )


def temperature_sweep(eigvals: np.ndarray,
                      betas: np.ndarray) -> list[ThermoState]:
    """Compute thermodynamic state at each beta in the sweep."""
    return [thermo_from_spectrum(eigvals, b) for b in betas]


# ---------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------

def runtime_inputs_for_spectrum(H: np.ndarray):
    """Pack the Hamiltonian as list-of-lists for one submit."""
    return [H.tolist()]


def classical_reference_spectrum(H: np.ndarray | None = None) -> np.ndarray:
    """NumPy reference: lowest n_states eigenvalues."""
    if H is None:
        H = build_h2_hamiltonian()
    return np.linalg.eigvalsh(H)


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=== H2 / STO-3G thermal ensemble — classical reference ===\n")

    H = build_h2_hamiltonian()
    eigvals = classical_reference_spectrum(H)
    print(f"spectrum (Ha): {eigvals}")
    print(f"E_FCI = {eigvals[0]:.6f} Ha (matches eigh)")
    print(f"gap to first excited: {eigvals[1] - eigvals[0]:.6f} Ha\n")

    # Sweep beta from 0.1 (extreme high T) to 50 (cold)
    betas = np.geomspace(0.1, 50.0, 25)
    sweep = temperature_sweep(eigvals, betas)

    print(f"{'beta (Ha^-1)':>14}  {'T (K)':>12}  {'U (Ha)':>10}  "
          f"{'F (Ha)':>10}  {'S':>8}  {'C_v':>8}  {'p_0':>8}")
    for s in sweep:
        print(f"{s.beta:>14.4f}  {s.T:>12.4e}  "
              f"{s.U:>+10.6f}  {s.F:>+10.6f}  "
              f"{s.S:>8.4f}  {s.Cv:>8.4f}  {s.populations[0]:>8.4f}")

    print("\nSanity checks:")
    high_T = sweep[0]
    print(f"  High T (beta={high_T.beta:.2f}): U={high_T.U:+.4f} should be ≈"
          f" mean(spectrum)={eigvals.mean():+.4f}")
    print(f"                          : S={high_T.S:.4f} should be ≈ "
          f"ln(4)={math.log(4):.4f}")
    low_T = sweep[-1]
    print(f"  Low  T (beta={low_T.beta:.2f}): U={low_T.U:+.6f} should be ≈"
          f" E_0={eigvals[0]:+.6f}")
    print(f"                          : S={low_T.S:.4f} should be ≈ 0")
    print(f"                          : p_0={low_T.populations[0]:.4f} should "
          f"be ≈ 1")
