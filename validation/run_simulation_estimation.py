"""
validation/run_simulation_estimation.py - Gate G5.

Simulation-estimation on the real design (same IDs, times, doses, covariates),
known true parameters, >= 30 replicates with fixed seeds:
- median relative bias theta < 10%
- median relative bias omega < 25%
- mean(SE) / empirical SD of estimates within [0.75, 1.33]
- Report non-convergence rate.
"""
import os
import sys
import time
import json
import concurrent.futures
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_io import load_dataset
from pk_core import fit_model, solve_1cpt_infusion, resid_sd

# True parameters
TRUE_TH = {"CL": 3.0, "V": 25.0}
TRUE_OM = {"CL": 0.30, "V": 0.25}
TRUE_ERR = [1.0, 0.15] # a, b

INIT = {
    "mu": [np.log(4.5), np.log(35.0)], # 1.5x perturbed
    "omega": [0.45, 0.40],
    "err": [1.5, 0.25]
}


def _run_single_rep(args):
    r, rep_seed, data_path = args
    data = load_dataset(data_path, tinf=1.0)
    N = data["N"]
    ids = data["ids"]
    sub_doses = data["sub_doses"]
    t_mat = data["t"]
    mask_mat = data["mask"]

    rng = np.random.default_rng(rep_seed)

    # 1. Simulate individual parameters
    eta_cl = rng.normal(0, TRUE_OM["CL"], N)
    eta_v = rng.normal(0, TRUE_OM["V"], N)
    ind_cl = TRUE_TH["CL"] * np.exp(eta_cl)
    ind_v = TRUE_TH["V"] * np.exp(eta_v)

    # 2. Simulate observations on exact real design
    sim_y = np.zeros_like(t_mat)
    for i, sid in enumerate(ids):
        mk = mask_mat[i]
        if not np.any(mk):
            continue
        to = t_mat[i, mk]
        doses = sub_doses[sid]
        f_true = solve_1cpt_infusion(doses, 1.0, to, ind_cl[i], ind_v[i])
        sd = resid_sd(f_true, TRUE_ERR, error_model="combined")
        eps = rng.standard_normal(len(f_true))
        y_sim = f_true + sd * eps
        y_sim = np.maximum(y_sim, 0.05)
        sim_y[i, mk] = y_sim
        sim_y[i, ~mk] = 0.0

    sim_data = dict(data)
    sim_data["y"] = sim_y

    t_start = time.time()
    try:
        res = fit_model(sim_data, "1-Compartment IV infusion (multi-dose)", INIT, {},
                        error_model="combined", n_iter1=160, n_iter2=80, seed=rep_seed)

        fim = res["fim"]
        est_log_cl, se_log_cl = fim.get("CL", (np.nan, np.nan))
        est_log_v, se_log_v = fim.get("V", (np.nan, np.nan))
        est_om_cl, se_om_cl = fim.get("omega_CL", (np.nan, np.nan))
        est_om_v, se_om_v = fim.get("omega_V", (np.nan, np.nan))

        est_cl = float(np.exp(est_log_cl))
        est_v = float(np.exp(est_log_v))
        se_cl = float(est_cl * se_log_cl) if np.isfinite(se_log_cl) else np.nan
        se_v = float(est_v * se_log_v) if np.isfinite(se_log_v) else np.nan

        is_conv = np.isfinite(se_log_cl) and np.isfinite(se_log_v) and (0.5 * TRUE_TH["CL"] < est_cl < 2.0 * TRUE_TH["CL"])

        rec = {
            "rep": r,
            "seed": rep_seed,
            "conv": bool(is_conv),
            "est_CL": est_cl,
            "se_CL": se_cl,
            "est_V": est_v,
            "se_V": se_v,
            "est_om_CL": float(est_om_cl),
            "se_om_CL": float(se_om_cl),
            "est_om_V": float(est_om_v),
            "se_om_V": float(se_om_v),
            "time_s": round(time.time() - t_start, 1)
        }
        return rec
    except Exception as e:
        return {"rep": r, "seed": rep_seed, "conv": False, "error": str(e), "time_s": round(time.time() - t_start, 1)}


def run_g5_simulation_estimation(n_reps=30, base_seed=100, max_workers=None):
    excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset_chính_thức.xlsx"))

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 6)

    t0_all = time.time()
    print(f"Starting Gate G5 Simulation-Estimation: {n_reps} replicates using {max_workers} processes...")
    print(f"True parameters: CL={TRUE_TH['CL']}, V={TRUE_TH['V']}, omega_CL={TRUE_OM['CL']}, omega_V={TRUE_OM['V']}, a={TRUE_ERR[0]}, b={TRUE_ERR[1]}")

    args_list = [(r, base_seed + r, excel_path) for r in range(n_reps)]

    results = []
    if max_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_single_rep, arg) for arg in args_list]
            for fut in concurrent.futures.as_completed(futures):
                rec = fut.result()
                results.append(rec)
                conv_str = "conv=True" if rec.get("conv") else f"conv=False ({rec.get('error', 'out of range')})"
                cl_val = rec.get('est_CL', np.nan)
                v_val = rec.get('est_V', np.nan)
                om_cl_val = rec.get('est_om_CL', np.nan)
                om_v_val = rec.get('est_om_V', np.nan)
                print(f"[{len(results):02d}/{n_reps}] Rep {rec['rep']+1:02d} ({rec.get('time_s', 0)}s): "
                      f"CL={cl_val:.3f}, V={v_val:.2f}, om_CL={om_cl_val:.3f}, om_V={om_v_val:.3f} [{conv_str}]")
    else:
        for arg in args_list:
            rec = _run_single_rep(arg)
            results.append(rec)
            conv_str = "conv=True" if rec.get("conv") else f"conv=False ({rec.get('error', 'out of range')})"
            cl_val = rec.get('est_CL', np.nan)
            v_val = rec.get('est_V', np.nan)
            om_cl_val = rec.get('est_om_CL', np.nan)
            om_v_val = rec.get('est_om_V', np.nan)
            print(f"[{len(results):02d}/{n_reps}] Rep {rec['rep']+1:02d} ({rec.get('time_s', 0)}s): "
                  f"CL={cl_val:.3f}, V={v_val:.2f}, om_CL={om_cl_val:.3f}, om_V={om_v_val:.3f} [{conv_str}]")

    results.sort(key=lambda x: x["rep"])

    # Summary statistics
    conv_results = [r for r in results if r.get("conv", False)]
    n_conv = len(conv_results)
    non_conv_rate = (n_reps - n_conv) / n_reps

    cls = np.array([r["est_CL"] for r in conv_results])
    vs = np.array([r["est_V"] for r in conv_results])
    om_cls = np.array([r["est_om_CL"] for r in conv_results])
    om_vs = np.array([r["est_om_V"] for r in conv_results])

    se_cls = np.array([r["se_CL"] for r in conv_results])
    se_vs = np.array([r["se_V"] for r in conv_results])

    rel_bias_cl = np.median(np.abs(cls - TRUE_TH["CL"]) / TRUE_TH["CL"])
    rel_bias_v = np.median(np.abs(vs - TRUE_TH["V"]) / TRUE_TH["V"])
    rel_bias_om_cl = np.median(np.abs(om_cls - TRUE_OM["CL"]) / TRUE_OM["CL"])
    rel_bias_om_v = np.median(np.abs(om_vs - TRUE_OM["V"]) / TRUE_OM["V"])

    # SE ratio: mean(SE) / empirical SD
    se_ratio_cl = float(np.mean(se_cls) / np.std(cls, ddof=1)) if np.std(cls, ddof=1) > 0 else np.nan
    se_ratio_v = float(np.mean(se_vs) / np.std(vs, ddof=1)) if np.std(vs, ddof=1) > 0 else np.nan

    g5_pass = (
        rel_bias_cl < 0.10 and rel_bias_v < 0.10 and
        rel_bias_om_cl < 0.25 and rel_bias_om_v < 0.25 and
        0.75 <= se_ratio_cl <= 1.33 and 0.75 <= se_ratio_v <= 1.33
    )

    summary = {
        "n_reps": n_reps,
        "n_conv": n_conv,
        "non_convergence_rate": round(non_conv_rate * 100, 2),
        "total_time_s": round(time.time() - t0_all, 1),
        "metrics": {
            "theta_CL_median_bias_pct": round(rel_bias_cl * 100, 2),
            "theta_V_median_bias_pct": round(rel_bias_v * 100, 2),
            "omega_CL_median_bias_pct": round(rel_bias_om_cl * 100, 2),
            "omega_V_median_bias_pct": round(rel_bias_om_v * 100, 2),
            "se_ratio_CL": round(se_ratio_cl, 3),
            "se_ratio_V": round(se_ratio_v, 3),
        },
        "gates": {
            "theta_bias_gate": bool(rel_bias_cl < 0.10 and rel_bias_v < 0.10),
            "omega_bias_gate": bool(rel_bias_om_cl < 0.25 and rel_bias_om_v < 0.25),
            "se_ratio_gate": bool(0.75 <= se_ratio_cl <= 1.33 and 0.75 <= se_ratio_v <= 1.33),
            "G5_overall_pass": bool(g5_pass),
        },
        "details": results
    }

    out_json = os.path.abspath(os.path.join(os.path.dirname(__file__), "g5_simulation_estimation.json"))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== GATE G5 SUMMARY ===")
    print(f"Non-convergence rate: {summary['non_convergence_rate']}%")
    print(f"theta_CL median bias: {summary['metrics']['theta_CL_median_bias_pct']}% (Gate < 10%): {'PASS' if rel_bias_cl < 0.10 else 'FAIL'}")
    print(f"theta_V median bias: {summary['metrics']['theta_V_median_bias_pct']}% (Gate < 10%): {'PASS' if rel_bias_v < 0.10 else 'FAIL'}")
    print(f"omega_CL median bias: {summary['metrics']['omega_CL_median_bias_pct']}% (Gate < 25%): {'PASS' if rel_bias_om_cl < 0.25 else 'FAIL'}")
    print(f"omega_V median bias: {summary['metrics']['omega_V_median_bias_pct']}% (Gate < 25%): {'PASS' if rel_bias_om_v < 0.25 else 'FAIL'}")
    print(f"SE ratio CL (mean SE / emp SD): {summary['metrics']['se_ratio_CL']} (Gate [0.75, 1.33]): {'PASS' if 0.75 <= se_ratio_cl <= 1.33 else 'FAIL'}")
    print(f"SE ratio V (mean SE / emp SD): {summary['metrics']['se_ratio_V']} (Gate [0.75, 1.33]): {'PASS' if 0.75 <= se_ratio_v <= 1.33 else 'FAIL'}")
    print(f"OVERALL G5: {'PASS' if g5_pass else 'FAIL'}")

    return summary


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_g5_simulation_estimation(n_reps=n)
