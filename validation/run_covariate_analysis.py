"""
validation/run_covariate_analysis.py - Step 6 Pre-specified Covariate Analysis.

Candidate models:
1. Base model: no covariates
2. CL ~ CLCr_CG (time-varying)
3. V ~ TBW
4. Sepsis on CL and on V
5. Age on CL (never together with CLCr_CG)

Reports: -2LL (OFV), delta OFV, AIC, delta AIC, BICc, delta BICc, Wald p-value.
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_io import load_dataset
from pk_core import fit_model


def run_covariate_analysis():
    excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset_chính_thức.xlsx"))
    data = load_dataset(excel_path, tinf=1.0)

    candidates = [
        {"id": "M1_base", "name": "Base Model (no covariates)", "cov_model": {}, "cov_mode": "time-varying"},
        {"id": "M2_cl_clcr", "name": "CL ~ CLCr_CG (time-varying)", "cov_model": {"CL": ["CLCr_CG"]}, "cov_mode": "time-varying"},
        {"id": "M3_v_tbw", "name": "V ~ TBW", "cov_model": {"V": ["TBW"]}, "cov_mode": "time-varying"},
        {"id": "M4_sepsis", "name": "Sepsis on CL and V", "cov_model": {"CL": ["Sepsis"], "V": ["Sepsis"]}, "cov_mode": "time-varying"},
        {"id": "M5_cl_age", "name": "CL ~ Age (without CLCr_CG)", "cov_model": {"CL": ["Age"]}, "cov_mode": "time-varying"},
    ]

    init_base = {
        "mu": [np.log(3.0), np.log(21.0)],
        "omega": [0.4, 0.3],
        "err": [1.5, 0.25]
    }

    results = []
    base_m2ll = None
    base_aic = None
    base_bicc = None

    print("=== STARTING PRE-SPECIFIED COVARIATE ANALYSIS ===")

    for cand in candidates:
        t0 = time.time()
        print(f"\n--- Fitting {cand['id']}: {cand['name']} ---")
        res = fit_model(data, "1-Compartment IV infusion (multi-dose)", init_base,
                        cand["cov_model"], error_model="combined",
                        n_iter1=250, n_iter2=150, seed=42, cov_mode=cand["cov_mode"])
        dt = time.time() - t0

        fim = res["fim"]
        m2ll = float(res["ic"]["m2ll"])
        aic = float(res["ic"]["aic"])
        bicc = float(res["ic"]["bicc"])

        if base_m2ll is None:
            base_m2ll = m2ll
            base_aic = aic
            base_bicc = bicc

        delta_ofv = m2ll - base_m2ll
        delta_aic = aic - base_aic
        delta_bicc = bicc - base_bicc

        # Parameters and SE
        cl_est = float(np.exp(fim["CL"][0]))
        cl_se = float(cl_est * fim["CL"][1]) if np.isfinite(fim["CL"][1]) else np.nan
        v_est = float(np.exp(fim["V"][0]))
        v_se = float(v_est * fim["V"][1]) if np.isfinite(fim["V"][1]) else np.nan
        om_cl = float(res["omega"][0])
        om_v = float(res["omega"][1])
        err_a = float(res["err"][0])
        err_b = float(res["err"][1])
        eta_shrink = res["diag"]["eta_shrink"].tolist() if "diag" in res and "eta_shrink" in res["diag"] else None
        eps_shrink = float(res["diag"]["eps_shrink"]) if "diag" in res and "eps_shrink" in res["diag"] else None

        # Extract beta effects and Wald p-values
        beta_info = []
        for p in ["CL", "V"]:
            covs = cand["cov_model"].get(p, [])
            for c in covs:
                label = f"beta_{p}_{c}"
                if label in fim:
                    b_est, b_se = fim[label]
                    z = b_est / b_se if np.isfinite(b_se) and b_se > 0 else np.nan
                    p_val = float(2.0 * (1.0 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
                    beta_info.append({
                        "param": label,
                        "est": round(float(b_est), 4),
                        "se": round(float(b_se), 4),
                        "wald_z": round(float(z), 2) if np.isfinite(z) else None,
                        "wald_p": round(float(p_val), 5) if np.isfinite(p_val) else None
                    })

        row = {
            "model_id": cand["id"],
            "model_name": cand["name"],
            "CL": round(cl_est, 4),
            "SE_CL": round(cl_se, 4) if np.isfinite(cl_se) else None,
            "V": round(v_est, 3),
            "SE_V": round(v_se, 3) if np.isfinite(v_se) else None,
            "omega_CL": round(om_cl, 4),
            "omega_V": round(om_v, 4),
            "a": round(err_a, 4),
            "b": round(err_b, 4),
            "-2LL": round(m2ll, 2),
            "delta_OFV": round(delta_ofv, 2),
            "AIC": round(aic, 2),
            "delta_AIC": round(delta_aic, 2),
            "BICc": round(bicc, 2),
            "delta_BICc": round(delta_bicc, 2),
            "covariates": beta_info,
            "eta_shrink": [round(s, 4) for s in eta_shrink] if eta_shrink else None,
            "eps_shrink": round(eps_shrink, 4) if eps_shrink is not None else None,
            "time_s": round(dt, 1)
        }
        results.append(row)
        print(f"Done in {dt:.1f}s: -2LL={m2ll:.2f} (dOFV={delta_ofv:.2f}), AIC={aic:.2f} (dAIC={delta_aic:.2f}), BICc={bicc:.2f}")
        for b in beta_info:
            print(f"  {b['param']}: {b['est']} (SE={b['se']}, Wald p={b['wald_p']})")

    out_json = os.path.abspath(os.path.join(os.path.dirname(__file__), "covariate_analysis.json"))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    df_out = pd.DataFrame([{
        "Model": r["model_name"],
        "-2LL": r["-2LL"],
        "ΔOFV": r["delta_OFV"],
        "AIC": r["AIC"],
        "ΔAIC": r["delta_AIC"],
        "BICc": r["BICc"],
        "ΔBICc": r["delta_BICc"],
        "Covariate Estimates (Wald p)": "; ".join([f"{b['param']}={b['est']} (p={b['wald_p']})" for b in r["covariates"]]) if r["covariates"] else "-"
    } for r in results])

    out_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "covariate_analysis.csv"))
    df_out.to_csv(out_csv, index=False)

    print("\n=== PRE-SPECIFIED COVARIATE ANALYSIS SUMMARY TABLE ===")
    print(df_out.to_string(index=False))

    return results


if __name__ == "__main__":
    run_covariate_analysis()
