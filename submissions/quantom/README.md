# Submission — Quantom

> **Custom Track** — Heterogeneous co-design of ground-state, variational, and thermal solvers for H₂ / STO-3G

## Team

- Team: **Quantom**
- Members: Elias, Shizong, Luis, Simon, Leander, Valentin
- Track: **custom**
- Reference molecule: H₂ at equilibrium R = 0.74 Å, minimal STO-3G basis

## What we built

**Three algorithms, one molecule, three Pareto frontiers.**

| Algorithm | What | ORIQX primitives | Submits |
|---|---|---|---|
| **RHF-SCF** | Classical mean-field. The full 50-iteration fixed-point loop compiled into one `ops.fori_loop` IR module. | `einsum`, `matmul`, `eigs`, `slice`, `transpose`, `reduce_sum`, `add/sub/mul`, `fori_loop` | One per geometry; 13 per MD step (1 energy + 12 force) |
| **VQE** | 1-parameter variational on the Z₂×Z₂-tapered 2-qubit Hamiltonian: `\|ψ(θ)⟩ = exp(-iθ X₀X₁/2)\|01⟩`. | `expv`, `expect` | One per θ point; 21 per landscape scan |
| **Thermal Ensemble** | Statistical mechanics on the molecular spectrum: partition function, internal energy, free energy, entropy, heat capacity. One `ops.eigs` submit; thermodynamics assembled classically. | `eigs` | **One per Hamiltonian** — entire β-sweep is post-processing |

All three algorithms answer the same physical question — *find the ground state, the variational landscape, and the thermal ensemble of the H₂ Hamiltonian in STO-3G* — through three radically different operator graphs (`fori_loop`+`einsum` vs `expv`+`expect` vs `eigs`). ORIQX's planner offered distinct Pareto frontiers for each, including direct QPU paths on AWS Braket. The submission demonstrates that heterogeneous co-design is meaningful **not just across hardware tiers but across algorithm choices targeting the same physics**.

## Why this point on the Pareto frontier

We submitted on `cpu-only` for both modules. The VQE preflight recommended `cpu+sim(SV1)` (Braket statevector simulator) at 252 tu — a 65× speedup over cpu-only at 16516 tu. During the hackathon, however, every Braket simulator path (SV1, TN1, dm1, qsim, qsim-statevec, cuquantum) returned `unknown error` on the backend; cpu-only completed 20/21 grid points at 0.22 s/submit with bit-exact agreement to the analytical reference. Reliability beat speed. The override is documented in `preflight_log.txt`.

The recommendation deviation is itself a Pareto observation: ORIQX's planner sees the asymptotic Pareto frontier in normal operation; in degraded mode, the user must apply judgement informed by measured behaviour rather than scoring. The same module, unchanged, will route to the recommended simulator the instant it stabilises.

## Headline numbers

| Metric | RHF reference | RHF ORIQX | VQE ORIQX (XX ansatz) | Thermal ORIQX | FCI (eigh) |
|---|---|---|---|---|---|
| E (Ha) | -1.11676 | _<filled in>_ | -1.11737 | spectrum extracted | -1.13768 |
| Wall-clock (s/submit) | n/a (local) | _<filled in>_ | 0.22 | _<filled in>_ | n/a (local) |
| Cost (USD/submit) | 0 | 0 (cpu-only) | 0 (cpu-only) | 0 (cpu-only) | 0 |
| Carbon (g/submit) | 0 | 0 (cpu-only) | 0 (cpu-only) | 0 (cpu-only) | 0 |

**Thermodynamic results from the thermal submit:**
- Schottky-anomaly peak at T ≈ 1.0 × 10⁵ K (k_B T ≈ 0.42 × electronic gap, matches textbook two-level Schottky)
- High-T limit S → ln 4 = 1.386 ✓; low-T limit S → 0, p₀ → 1 ✓
- Internal energy spans U = -1.138 Ha (T → 0) to U = mean(spectrum) (T → ∞)

Correlation gap (VQE → FCI): **0.020 Ha**, the textbook STO-3G correlation energy of H₂.
This is *the ansatz's fundamental limitation*, not numerical error. A complex Hermitian generator `(X₀Y₁ − Y₀X₁)/4` closes the gap and is documented as future work.

## How to reproduce

**In Studio (recommended):** open `submission.ipynb` in your hosted workspace. `uniqx` is pre-installed and `UNIQX_API_KEY` is already exported in the pod — Run All. Notebook gracefully falls back to NumPy if the gateway is unreachable, so it always completes top-to-bottom (the Robustness 25-point bar).

**Locally:**

```bash
export UNIQX_API_KEY="uxk_..." UNIQX_GATEWAY="api.oriqx.com:443"
pip install --extra-index-url "https://uniqx:${UNIQX_API_KEY}@wheels.oriqx.com/simple/" uniqx
pip install -e ".[all]"
jupyter nbconvert --execute submissions/<team>/submission.ipynb
```

## What's in this folder

| File | Purpose |
|---|---|
| `submission.ipynb` | Main deliverable. Top-to-bottom executable. |
| `scf_oriqx.py` | Traced RHF-SCF module; the `make_scf_module(n_basis, n_occ, max_iter)` factory + `parse_energy` + `runtime_inputs_for` helpers. |
| `vqe_h2_oriqx.py` | Traced VQE module + grid-scan driver + classical reference for validation. |
| `aimd_oriqx.py` | Born-Oppenheimer AIMD wrapper. `GatewayEnergy` class with retry + NumPy fallback, `aimd_oriqx` driver. Standalone runnable. |
| `thermal_h2_oriqx.py` | Thermal-ensemble third algorithm. Traced `ops.eigs` module + classical statistical-mechanics post-processing. Standalone runnable for the classical reference. |
| `decomposition_map.md` | Required: original code → ORIQX primitives table with rationale for all three algorithms. |
| `results.json` | Schema-conformant submission metadata. |
| `preflight_log.txt` | Pareto tables from preflight calls, plus backend stability observations. |
| `vqe_grid_scan.csv` | Raw data from the 21-point VQE landscape (θ, classical reference, ORIQX result, wall time). |
| `thermal_sweep.csv` | Raw data from the thermal sweep (60 temperatures, Z, U, F, S, C_v, level populations). |
| `vqe_pareto_landscape.png` | The VQE landscape plot. |
| `thermal_ensemble.png` | The thermal four-panel plot (U, S, C_v, populations vs T). |
| `aimd_h2_oriqx_trajectory.xyz` | The 5-step AIMD trajectory produced by `aimd_oriqx.py`. |

## What we'd do with more time

1. Replace the XX VQE generator with `i·(X₀Y₁ − Y₀X₁)/4` (complex Hermitian) to close the 20 mHa correlation gap. Requires verifying `ir.ScalarType(dtype="c128")` support.
2. Lift the McMurchie-Davidson ERI build into a traced server-side module so H₂O / H₂O-cluster runs are not blocked by the runtime-inputs payload limit. The 14-op SCF body itself already traces correctly for H₂O.
3. Drive the VQE optimiser with parameter-shift gradients submitted on the actual QPU paths (`block-qpu-accurate`) once Braket integration stabilises. Compare convergence count and total cost against the grid scan.
4. Repeat the thermal sweep across a bond-stretching scan (R = 0.4 → 2.5 Å, spectrum from VQE-for-excited-states at each R) to surface a temperature-dependent dissociation curve. Free energy of dissociation as a function of T is a real experimental quantity.
5. Extend to LiH / BeH₂ tapered Hamiltonians, which sit exactly at the QPU-vs-GPU crossover and would surface a richer Pareto frontier for all three algorithms.
