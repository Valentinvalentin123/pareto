"""Quick standalone verification: Toolbox Model A vs PySCF on H2O/STO-3G."""

import os

import uniqx
from uniqx.domains.chemistry.basis import extract_basis

from baseline import rhf_reference
from scf_toolbox import build_module_A


def main():
    gateway = os.environ.get("UNIQX_GATEWAY", "api.oriqx.com:443")
    uniqx.login(os.environ["UNIQX_API_KEY"], gateway=gateway)
    client = uniqx.connect(gateway)

    H2O = [
        ("O", [0.0, 0.0, 0.1173]),
        ("H", [0.0, 0.7572, -0.4692]),
        ("H", [0.0, -0.7572, -0.4692]),
    ]
    info = extract_basis(H2O, "sto-3g")
    print(f"basis fns: {info.n_basis} | electrons: {info.n_electrons}")

    module = build_module_A(H2O, info, max_iter=50, tol=1e-7)
    fn = module.functions[0]
    print(f"Model A IR: {len(module.functions)} fn, {len(fn.ops)} body ops")

    options = uniqx.preflight(module, client=client)
    print(options.summary())

    # Force cpu-only (the recommended cpu+gpu path returns NaN on this gateway)
    choice = options.by_label("cpu-only")
    print(f"Picked: {choice['label']} (time={choice['total_time']:.0f} tu)")

    runtime_inputs = [
        list(info.exps_flat), list(info.coeffs_flat),
        list(info.centers_flat), list(info.ang_flat),
        list(info.atom_coords_flat), list(info.charges_flat),
    ]
    job_id = uniqx.submit(
        module, client=client,
        preflight_job_id=options.job_id, option_idx=choice["_idx"],
        runtime_inputs=runtime_inputs,
    )
    result = uniqx.get(job_id, client=client, timeout=300)
    payload = result.get("payload") or b""
    if isinstance(payload, str):
        payload = payload.encode()
    parsed = uniqx.parse_result(payload, ["e_total"])
    e_toolbox = parsed["e_total"][2][0]

    e_ref = rhf_reference(H2O, "sto-3g")
    rel_err = abs(e_toolbox - e_ref) / abs(e_ref)
    print(f"\nToolbox A  : {e_toolbox:.6f} Ha")
    print(f"PySCF ref  : {e_ref:.6f} Ha")
    print(f"Rel error  : {rel_err:.2e}")


if __name__ == "__main__":
    main()
