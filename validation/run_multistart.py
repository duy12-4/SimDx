"""
validation/run_multistart.py - Gate G6.

Multi-start: >= 4 runs with different seeds and initial values (0.5x - 2x of naive pooled NLS fit)
-> theta agree within 5%, -2LL within 2 units.
"""
import os
import sys
import time
import json
import concurrent.futures
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_io import load_dataset
from pk_core import fit_model


def _run_single_start(s):
    excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset_chính_thức.xlsx"))
    data = load_dataset(excel_path, tinf=1.0)
    init = {
        "mu": [np.log(s["cl"]), np.log(s["v"])],
        "omega": [s["om_cl"], s["om_v"]],
        "err": [s["a"], s["b"]]
    }
    t0 = time.time()
    res = fit_model(data, "1-Compartment IV infusion (multi-dose)", init, {},
                    error_model="combined", n_iter1=300, n_iter2=200, seed=s["seed"], n_is=3000)
    dt = time.time() - t0

    fim = res["fim"]
    log_cl, se_log_cl = fim["CL"]
    log_v, se_log_v = fim["V"]
    est_cl = float(np.exp(log_cl))
    est_v = float(np.exp(log_v))
    m2ll = float(res["ic"]["m2ll"])
    mc_se = float(res["ic"].get("mc_se", 0.0))

    rec = {
        "name": s["name"],
        "seed": s["seed"],
        "init_CL": s["cl"],
        "init_V": s["v"],
        "est_CL": est_cl,
        "est_V": est_v,
        "m2ll": m2ll,
        "mc_se": mc_se,
        "aic": float(res["ic"]["aic"]),
        "bicc": float(res["ic"]["bicc"]),
        "omega_CL": float(res["omega"][0]),
        "omega_V": float(res["omega"][1]),
        "time_s": round(dt, 1)
    }
    print(f"Completed {s['name']}: CL={est_cl:.4f}, V={est_v:.4f}, -2LL={m2ll:.2f} (MC-SE={mc_se:.2f}) in {dt:.1f}s")
    return rec


def run_g6_multistart():
    # Baseline estimate: CL ~ 3.0, V ~ 21.0
    # 4 distinct starts ranging from 0.5x to 2.0x with distinct seeds
    starts = [
        {"name": "0.5x", "cl": 1.5, "v": 11.0, "om_cl": 0.3, "om_v": 0.3, "a": 1.0, "b": 0.20, "seed": 11},
        {"name": "0.8x", "cl": 2.4, "v": 17.0, "om_cl": 0.4, "om_v": 0.3, "a": 1.2, "b": 0.22, "seed": 22},
        {"name": "1.2x", "cl": 3.6, "v": 25.0, "om_cl": 0.5, "om_v": 0.4, "a": 1.5, "b": 0.25, "seed": 33},
        {"name": "2.0x", "cl": 6.0, "v": 42.0, "om_cl": 0.6, "om_v": 0.4, "a": 2.0, "b": 0.30, "seed": 44},
    ]

    print("Starting Gate G6 Multi-Start (4 runs in parallel)...")
    t0_all = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        runs = list(executor.map(_run_single_start, starts))

    cls = [r["est_CL"] for r in runs]
    vs = [r["est_V"] for r in runs]
    m2lls = [r["m2ll"] for r in runs]

    mean_cl = np.mean(cls)
    mean_v = np.mean(vs)

    max_dev_cl_pct = float(np.max(np.abs(cls - mean_cl) / mean_cl) * 100)
    max_dev_v_pct = float(np.max(np.abs(vs - mean_v) / mean_v) * 100)
    delta_m2ll = float(np.max(m2lls) - np.min(m2lls))

    cl_agree = max_dev_cl_pct < 5.0
    v_agree = max_dev_v_pct < 5.0
    m2ll_agree = delta_m2ll <= 2.0
    overall_pass = cl_agree and v_agree and m2ll_agree

    summary = {
        "runs": runs,
        "mean_CL": round(mean_cl, 4),
        "mean_V": round(mean_v, 4),
        "max_dev_CL_pct": round(max_dev_cl_pct, 2),
        "max_dev_V_pct": round(max_dev_v_pct, 2),
        "delta_m2ll": round(delta_m2ll, 2),
        "gates": {
            "CL_agree_5pct": bool(cl_agree),
            "V_agree_5pct": bool(v_agree),
            "m2ll_agree_2units": bool(m2ll_agree),
            "G6_overall_pass": bool(overall_pass)
        },
        "total_time_s": round(time.time() - t0_all, 1)
    }

    out_json = os.path.abspath(os.path.join(os.path.dirname(__file__), "g6_multistart.json"))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== GATE G6 MULTI-START SUMMARY ===")
    print(f"Max CL deviation from mean: {summary['max_dev_CL_pct']}% (Gate < 5%): {'PASS' if cl_agree else 'FAIL'}")
    print(f"Max V deviation from mean: {summary['max_dev_V_pct']}% (Gate < 5%): {'PASS' if v_agree else 'FAIL'}")
    print(f"Max delta -2LL across runs: {summary['delta_m2ll']} (Gate <= 2 units): {'PASS' if m2ll_agree else 'FAIL'}")
    print(f"OVERALL G6: {'PASS' if overall_pass else 'FAIL'}")

    return summary


if __name__ == "__main__":
    run_g6_multistart()
