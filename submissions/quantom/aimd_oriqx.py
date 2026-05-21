"""
ORIQX Custom Track — Born-Oppenheimer AIMD for H2 in STO-3G.

Two algorithms running side-by-side for the same physics question
("ground-state energy on a discrete nuclear-coordinate trajectory"):

  1. **RHF-SCF on ORIQX (this file)** — classical mean-field solver, the
     full 50-iteration SCF compiled into one `ops.fori_loop` IR module
     (see scf_oriqx.py).  One gateway submit per geometry; 2 atoms x 3
     axes x 2 signs + 1 energy = 13 submits per MD step on H2.

  2. **VQE on ORIQX (vqe_h2_oriqx.py)** — 1-parameter variational solver
     using `ops.expv` + `ops.expect` on the Z2xZ2-tapered 2-qubit
     Hamiltonian.  Same observable, quantum-amenable execution path.

Why H2 and not H2O? See decomposition_map.md: the 7^4-element ERI tensor
of H2O exceeded the gateway's submit-payload limit during the hackathon.
H2 keeps the algorithm identical while shrinking the payload by ~150x,
letting us prove end-to-end execution and produce a real Pareto table.
The architecture (1 traced module, N submits per trajectory) carries
over to larger molecules once gateway payload limits relax.

Run:
    UNIQX_GATEWAY=api.oriqx.com:443 UNIQX_API_KEY=uxk_... \
        python aimd_oriqx.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# Make the baseline NumPy reference modules importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pareto", "tracks", "md"))

from aimd import write_xyz                                # NumPy Verlet helpers
from basis import build_basis
from constants import AMU_TO_AU, ANG_TO_BOHR, BOHR_TO_ANG, FS_TO_AUT
from scf import (
    build_eri_tensor,
    build_nuclear,
    build_overlap_kinetic,
    nuclear_repulsion,
    rhf_scf,                                              # NumPy SCF reference
)
from uniqx import connect, get, login, parse_result, preflight, submit

from scf_oriqx import make_scf_module, parse_energy, runtime_inputs_for


def orthogonaliser(S):
    """X = S^(-1/2) via eigendecomposition. CPU, once per geometry."""
    eig, U = np.linalg.eigh(S)
    return U @ np.diag(1.0 / np.sqrt(eig)) @ U.T


def build_inputs(atoms_bohr):
    """Build H, X, g, e_nuc on CPU for one geometry."""
    basis = build_basis(atoms_bohr)
    S, T = build_overlap_kinetic(basis)
    V = build_nuclear(basis, atoms_bohr)
    H = T + V
    g = build_eri_tensor(basis)
    e_nuc = nuclear_repulsion(atoms_bohr)
    X = orthogonaliser(S)
    return H, X, g, e_nuc, len(basis)


class GatewayEnergy:
    """Submit the traced SCF module to ORIQX, return total energy.

    On submit failure (transient h2 errors), retries up to `max_retries`
    times with exponential backoff. If still failing, falls back to the
    NumPy reference so the AIMD loop continues; the failure is logged.
    """

    def __init__(self, n_basis, n_occ, max_iter=50, max_retries=5,
                 force_backend="cpu-only"):
        self.n_basis = n_basis
        self.n_occ = n_occ
        self.max_retries = max_retries
        self.force_backend = force_backend
        self.api_failures = 0
        self.api_successes = 0

        endpoint = os.environ.get("UNIQX_GATEWAY", "api.oriqx.com:443")
        if os.environ.get("UNIQX_API_KEY"):
            login(os.environ["UNIQX_API_KEY"], gateway=endpoint)
        self.client = connect(endpoint)

        # Trace once with placeholder list-of-lists (NOT ndarray!).
        H0 = np.zeros((n_basis, n_basis)).tolist()
        X0 = np.eye(n_basis).tolist()
        g0 = np.zeros((n_basis,) * 4).tolist()
        D0 = np.zeros((n_basis, n_basis)).tolist()
        half = np.full((n_basis, n_basis), 0.5).tolist()
        two = np.full((n_basis, n_basis), 2.0).tolist()
        rhf_scf_fn = make_scf_module(n_basis, n_occ, max_iter)
        self.module = rhf_scf_fn(H0, X0, g0, 0.0, D0, half, two)

        # Preflight; cache the table for `results.json` later.
        self.options = preflight(self.module, client=self.client)
        print(self.options.summary())
        chosen = next(
            (o for o in self.options if o["label"] == force_backend),
            self.options.recommended,
        )
        self.option_idx = chosen["_idx"]
        self.preflight_job_id = self.options.job_id
        print(f"chosen backend: {chosen['label']}  "
              f"(time={chosen['total_time']} tu, "
              f"err={chosen['max_error_rate']*100:.2f}%)")

    def energy(self, atoms_bohr, classical_fallback=True):
        """Submit one geometry. Returns total RHF energy in Hartree."""
        H, X, g, e_nuc, _ = build_inputs(atoms_bohr)
        for attempt in range(self.max_retries):
            try:
                jid = submit(
                    self.module,
                    client=self.client,
                    runtime_inputs=runtime_inputs_for(self.n_basis, H, X, g, e_nuc),
                    preflight_job_id=self.preflight_job_id,
                    option_idx=self.option_idx,
                )
                res = get(jid, client=self.client, timeout=120.0)
                payload = res.get("payload", b"") or b""
                if isinstance(payload, str):
                    payload = payload.encode()
                out = parse_result(payload, ["e_raw"])
                self.api_successes += 1
                return parse_energy(float(out["e_raw"][2][0]))
            except Exception as exc:
                self.api_failures += 1
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    if classical_fallback:
                        e_class = rhf_scf(atoms_bohr,
                                          n_electrons=2 * self.n_occ)
                        print(f"  [api failure -> numpy fallback E={e_class:.6f}]")
                        return e_class
                    raise

    def forces(self, atoms_bohr, n_electrons, fd_step=0.005):
        """Central-difference forces. For each (atom, axis), 2 submits."""
        symbols = [a[0] for a in atoms_bohr]
        positions = np.array([a[1] for a in atoms_bohr], dtype=float)
        forces = np.zeros_like(positions)
        for i in range(len(symbols)):
            for d in range(3):
                pf = positions.copy(); pf[i, d] += fd_step
                pb = positions.copy(); pb[i, d] -= fd_step
                ef = self.energy(list(zip(symbols, pf.tolist())))
                eb = self.energy(list(zip(symbols, pb.tolist())))
                forces[i, d] = -(ef - eb) / (2 * fd_step)
        return forces


def aimd_oriqx(atoms_ang, masses_amu, n_electrons,
               n_steps=5, dt_fs=0.5,
               init_velocities=None, xyz_file=None,
               force_backend="cpu-only"):
    """Born-Oppenheimer AIMD with energies/forces from ORIQX.

    Returns (trajectory, energies, runner) — `runner.api_failures` /
    `runner.api_successes` are useful diagnostics for the submission.
    """
    symbols = [a[0] for a in atoms_ang]
    positions = np.array([a[1] for a in atoms_ang]) * ANG_TO_BOHR
    velocities = (np.array(init_velocities, dtype=float)
                  if init_velocities is not None else np.zeros_like(positions))
    inv_masses = (1.0 / (masses_amu * AMU_TO_AU))[:, None]
    dt = dt_fs * FS_TO_AUT

    # Probe basis size by building once on CPU
    atoms_bohr_init = list(zip(symbols, positions.tolist()))
    _, _, _, _, n_basis = build_inputs(atoms_bohr_init)
    n_occ = n_electrons // 2
    runner = GatewayEnergy(n_basis, n_occ, force_backend=force_backend)

    trajectory, energies = [positions.copy()], []
    atoms = list(zip(symbols, positions.tolist()))

    t0 = time.monotonic()
    forces = runner.forces(atoms, n_electrons)
    e = runner.energy(atoms)
    energies.append(e)
    print(f"\nstep 0 | E = {e:+.6f} Ha | wall = {time.monotonic()-t0:.1f}s")

    for step in range(1, n_steps + 1):
        positions_new = positions + velocities*dt + 0.5*forces*inv_masses*dt**2
        atoms_new = list(zip(symbols, positions_new.tolist()))
        t0 = time.monotonic()
        forces_new = runner.forces(atoms_new, n_electrons)
        velocities = velocities + 0.5*(forces + forces_new)*inv_masses*dt
        positions, forces = positions_new, forces_new
        e = runner.energy(atoms_new)
        energies.append(e)
        trajectory.append(positions.copy())
        # For H2 only one bond; reuse the formatting from baseline.aimd
        bond = np.linalg.norm(positions[1] - positions[0]) * BOHR_TO_ANG
        print(f"step {step} | E = {e:+.6f} Ha | H-H = {bond:.4f} A "
              f"| wall = {time.monotonic()-t0:.1f}s")

    if xyz_file:
        write_xyz(xyz_file, symbols, trajectory, energies)

    print(f"\nAPI stats: {runner.api_successes} successes / "
          f"{runner.api_failures} failures")
    return trajectory, energies, runner


# ---------------------------------------------------------------------
# H2 default geometry
# ---------------------------------------------------------------------

H2_EQUILIBRIUM = [
    ("H", [0.0, 0.0, -0.37]),
    ("H", [0.0, 0.0, +0.37]),
]
H2_MASSES = np.array([1.008, 1.008])
H2_N_ELECTRONS = 2

# Symmetric stretching mode: opposite z-velocities, zero net momentum
H2_VEL_STRETCH = 0.002   # Bohr / atomic-time-unit
H2_INIT_VEL_STRETCH = np.array([[0.0, 0.0, -H2_VEL_STRETCH],
                                 [0.0, 0.0, +H2_VEL_STRETCH]])


if __name__ == "__main__":
    traj, energies, runner = aimd_oriqx(
        H2_EQUILIBRIUM,
        H2_MASSES,
        H2_N_ELECTRONS,
        n_steps=5,
        dt_fs=0.5,
        init_velocities=H2_INIT_VEL_STRETCH,
        xyz_file="aimd_h2_oriqx_trajectory.xyz",
        force_backend="cpu-only",     # most stable during hackathon
    )

    print("\n--- Summary ---")
    print(f"{'Step':>5}  {'E (Ha)':>12}  {'H-H (Å)':>10}")
    for i, (pos, e) in enumerate(zip(traj, energies)):
        r = np.linalg.norm(pos[1] - pos[0]) * BOHR_TO_ANG
        print(f"{i:>5}  {e:>12.6f}  {r:>10.4f}")
