"""
validation/run_simulation_estimation_cov.py - Gate F6.

Covariate simulation-estimation check on the real design:
- True model: 1-Compartment IV infusion (multi-dose), CL ~ CLCr_CG (time-varying)
- Known true parameters:
  - theta: CL = 3.0 L/h, V = 25.0 L
  - beta_CL_CLCr_CG = 0.8443
  - omega: omega_CL = 0.30, omega_V = 0.25
  - err: a = 1.0, b = 0.15 (combined)
- >= 15 replicates with fixed seeds:
  - median relative bias of beta_CLCr < 10%
  - median relative bias of theta < 10%
  - mean(SE) / empirical SD in [0.75, 1.33]
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
from pk_core import fit_model, predict, resid_sd, prepare_data

# True parameters
TRUE_TH = {"CL": 3.0, "V": 25.0}
TRUE_BETA = 0.8443
TRUE_OM = {"CL": 0.30, "V": 0.25}
TRUE_ERR = [1.0, 0.15] # a, b

COV_MODEL = {"CL": ["CLCr_CG"]}

INIT = {
    "mu": [np.log(4.5), np.log(35.0)], # 1.5x perturbed
    "omega": [0.45, 0.40],
    "err": [1.5, 0.25]
}


def _run_single_cov_rep(args):
    r, rep_seed, data_path = args
    data = prepare_data(load_dataset(data_path, tinf=1.0), cov_model=COV_MODEL)
    N = data["N"]
    ids = data["ids"]
    t_mat = data["t"]
    mask_mat = data["mask"]
    dose_mat = data["dose"]
    ref_median = data["ref_medians"]["CLCr_CG"]

    rng = np.random.default_rng(rep_seed)

    # 1. Simulate individual random effects
    eta_cl = rng.normal(0, TRUE_OM["CL"], N)
    eta_v = rng.normal(0, TRUE_OM["V"], N)

    # phi matrix includes base shift: phi_i = log(theta) + beta * log(CLCr_base / ref) + eta_i
    # For predicting true concentrations:
    beta_mat = [np.array([np.log(TRUE_TH["CL"]), TRUE_BETA]), np.array([np.log(TRUE_TH["V"])])]
    
    # Calculate individual baseline shifts
    phi_sim = np.zeros((N, 2), dtype=float)
    for i, sid in enumerate(ids):
        events = data["sub_cov_events"][sid]["CLCr_CG"]
        base_val = events[0][1]
        base_shift = TRUE_BETA * np.log(base_val / ref_median)
        phi_sim[i, 0] = np.log(TRUE_TH["CL"]) + base_shift + eta_cl[i]
        phi_sim[i, 1] = np.log(TRUE_TH["V"]) + eta_v[i]

    # 2. Simulate observations on exact real design
    # Full batch predict with explicit subject_ids
    f_true = predict(
        "1-Compartment IV infusion (multi-dose)", t_mat, dose_mat, phi_sim,
        data=data, beta=beta_mat, cov_model=COV_MODEL, cov_mode="time-varying",
        subject_ids=ids
    )

    sim_y = np.zeros_like(t_mat)
    for i in range(N):
        mk = mask_mat[i]
        if not np.any(mk):
            continue
        f_sub = f_true[i, mk]
        sd = resid_sd(f_sub, TRUE_ERR, error_model="combined")
        eps = rng.standard_normal(len(f_sub))
        y_sim = f_sub + sd * eps
        y_sim = np.maximum(y_sim, 0.05)
        sim_y[i, mk] = y_sim
        sim_y[i, ~mk] = 0.0

    sim_data = dict(data)
    sim_data["y"] = sim_y

    t_start = time.time()
    try:
        res = fit_model(sim_data, "1-Compartment IV infusion (multi-dose)", INIT, COV_MODEL,
                        error_model="combined", n_iter1=160, n_iter2=80, seed=rep_seed, n_is=500)

        fim = res["fim"]
        beta_res = res["beta"]
        est_beta = float(beta_res[0][1])
        
        est_log_cl, se_log_cl = fim.get("CL", (np.nan, np.nan))
        est_log_v, se_log_v = fim.get("V", (np.nan, np.nan))
        se_beta = float(fim.get("beta_CL_CLCr_CG", (np.nan, np.nan))[1])
        
        est_om_cl, se_om_cl = fim.get("omega_CL", (np.nan, np.nan))
        est_om_v, se_om_v = fim.get("omega_V", (np.nan, np.nan))

        est_cl = float(np.exp(est_log_cl))
        est_v = float(np.exp(est_log_v))
        se_cl = float(est_cl * se_log_cl) if np.isfinite(se_log_cl) else np.nan
        se_v = float(est_v * se_log_v) if np.isfinite(se_log_v) else np.nan

        is_conv = (
            np.isfinite(se_log_cl) and np.isfinite(se_log_v) and np.isfinite(se_beta) and
            (0.5 * TRUE_TH["CL"] < est_cl < 2.0 * TRUE_TH["CL"]) and
            (0.0 < est_beta < 2.0)
        )

        rec = {
            "rep": r,
            "seed": rep_seed,
            "conv": bool(is_conv),
            "est_CL": est_cl,
            "se_CL": se_cl,
            "est_V": est_v,
            "se_V": se_v,
            "est_beta": est_beta,
            "se_beta": se_beta,
            "est_om_CL": float(est_om_cl),
            "se_om_CL": float(se_om_cl),
            "est_om_V": float(est_om_v),
            "se_om_V": float(se_om_v),
            "time_s": round(time.time() - t_start, 1)
        }
        return rec
    except Exception as e:
        return {"rep": r, "seed": rep_seed, "conv": False, "error": str(e), "time_s": round(time.time() - t_start, 1)}


def run_f6_simulation_estimation_cov(n_reps=15, base_seed=500, max_workers=None):
    excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset_chính_thức.xlsx"))
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 6)

    t0_all = time.time()
    print(f"Starting Gate F6 Covariate Simulation-Estimation: {n_reps} replicates using {max_workers} processes...")
    print(f"True parameters: CL={TRUE_TH['CL']}, V={TRUE_TH['V']}, beta_CLCr={TRUE_BETA}, omega_CL={TRUE_OM['CL']}, omega_V={TRUE_OM['V']}")

    args_list = [(r, base_seed + r, excel_path) for r in range(n_reps)]

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_single_cov_rep, arg) for arg in args_list]
        for fut in concurrent.futures.as_completed(futures):
            rec = fut.result()
            results.append(rec)
            conv_str = "conv=True" if rec.get("conv") else f"conv=False ({rec.get('error', 'out of range')})"
            cl_val = rec.get('est_CL', np.nan)
            v_val = rec.get('est_V', np.nan)
            b_val = rec.get('est_beta', np.nan)
            print(f"[{len(results):02d}/{n_reps}] Rep {rec['rep']+1:02d} ({rec.get('time_s', 0)}s): "
                  f"CL={cl_val:.3f}, V={v_val:.2f}, beta={b_val:.4f} [{conv_str}]")

    results.sort(key=lambda x: x["rep"])

    conv_results = [r for r in results if r.get("conv", False)]
    n_conv = len(conv_results)
    non_conv_rate = (n_reps - n_conv) / n_reps

    cls = np.array([r["est_CL"] for r in conv_results])
    vs = np.array([r["est_V"] for r in conv_results])
    betas = np.array([r["est_beta"] for r in conv_results])
    om_cls = np.array([r["est_om_CL"] for r in conv_results])
    om_vs = np.array([r["est_om_V"] for r in conv_results])

    se_cls = np.array([r["se_CL"] for r in conv_results])
    se_vs = np.array([r["se_V"] for r in conv_results])
    se_betas = np.array([r["se_beta"] for r in conv_results])

    rel_bias_cl = np.median(np.abs(cls - TRUE_TH["CL"]) / TRUE_TH["CL"])
    rel_bias_v = np.median(np.abs(vs - TRUE_TH["V"]) / TRUE_TH["V"])
    rel_bias_beta = np.median(np.abs(betas - TRUE_BETA) / TRUE_BETA)
    rel_bias_om_cl = np.median(np.abs(om_cls - TRUE_OM["CL"]) / TRUE_OM["CL"])
    rel_bias_om_v = np.median(np.abs(om_vs - TRUE_OM["V"]) / TRUE_OM["V"])

    se_ratio_cl = float(np.mean(se_cls) / np.std(cls, ddof=1)) if np.std(cls, ddof=1) > 0 else np.nan
    se_ratio_v = float(np.mean(se_vs) / np.std(vs, ddof=1)) if np.std(vs, ddof=1) > 0 else np.nan
    se_ratio_beta = float(np.mean(se_betas) / np.std(betas, ddof=1)) if np.std(betas, ddof=1) > 0 else np.nan

    f6_pass = (
        rel_bias_beta < 0.10 and rel_bias_cl < 0.10 and rel_bias_v < 0.10 and
        0.75 <= se_ratio_beta <= 1.33 and 0.75 <= se_ratio_cl <= 1.33 and 0.75 <= se_ratio_v <= 1.33
    )

    summary = {
        "n_reps": n_reps,
        "n_conv": n_conv,
        "non_convergence_rate": round(non_conv_rate * 100, 2),
        "total_time_s": round(time.time() - t0_all, 1),
        "metrics": {
            "beta_CLCr_median_bias_pct": round(rel_bias_beta * 100, 2),
            "theta_CL_median_bias_pct": round(rel_bias_cl * 100, 2),
            "theta_V_median_bias_pct": round(rel_bias_v * 100, 2),
            "omega_CL_median_bias_pct": round(rel_bias_om_cl * 100, 2),
            "omega_V_median_bias_pct": round(rel_bias_om_v * 100, 2),
            "se_ratio_beta": round(se_ratio_beta, 3),
            "se_ratio_CL": round(se_ratio_cl, 3),
            "se_ratio_V": round(se_ratio_v, 3),
        },
        "gates": {
            "beta_bias_gate": bool(rel_bias_beta < 0.10),
            "theta_bias_gate": bool(rel_bias_cl < 0.10 and rel_bias_v < 0.10),
            "se_ratio_gate": bool(0.75 <= se_ratio_beta <= 1.33 and 0.75 <= se_ratio_cl <= 1.33 and 0.75 <= se_ratio_v <= 1.33),
            "F6_overall_pass": bool(f6_pass),
        },
        "details": results
    }

    out_json = os.path.abspath(os.path.join(os.path.dirname(__file__), "f6_simulation_estimation_cov.json"))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== GATE F6 COVARIATE SIMULATION-ESTIMATION SUMMARY ===")
    print(f"Non-convergence rate: {summary['non_convergence_rate']}%")
    print(f"beta_CLCr median bias: {summary['metrics']['beta_CLCr_median_bias_pct']}% (Gate < 10%): {'PASS' if rel_bias_beta < 0.10 else 'FAIL'}")
    print(f"theta_CL median bias: {summary['metrics']['theta_CL_median_bias_pct']}% (Gate < 10%): {'PASS' if rel_bias_cl < 0.10 else 'FAIL'}")
    print(f"theta_V median bias: {summary['metrics']['theta_V_median_bias_pct']}% (Gate < 10%): {'PASS' if rel_bias_v < 0.10 else 'FAIL'}")
    print(f"SE ratio beta: {summary['metrics']['se_ratio_beta']} (Gate [0.75, 1.33]): {'PASS' if 0.75 <= se_ratio_beta <= 1.33 else 'FAIL'}")
    print(f"SE ratio CL: {summary['metrics']['se_ratio_CL']} (Gate [0.75, 1.33]): {'PASS' if 0.75 <= se_ratio_cl <= 1.33 else 'FAIL'}")
    print(f"SE ratio V: {summary['metrics']['se_ratio_V']} (Gate [0.75, 1.33]): {'PASS' if 0.75 <= se_ratio_v <= 1.33 else 'FAIL'}")
    print(f"OVERALL F6: {'PASS' if f6_pass else 'FAIL'}")

    return summary


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    run_f6_simulation_estimation_cov(n_reps=n)
