"""
validation/run_sensitivity_tinf.py - Gate G8.

Sensitivity table for tinf in {0.5, 1.0, 2.0} h (theta, omega, -2LL, AIC, BICc).
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_io import load_dataset
from pk_core import fit_model


def run_g8_sensitivity():
    excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset_chính_thức.xlsx"))
    tinf_values = [0.5, 1.0, 2.0]
    
    results = []
    print("Starting Gate G8 Sensitivity Analysis for tinf in {0.5, 1.0, 2.0} h...")
    
    init = {
        "mu": [np.log(3.0), np.log(21.0)],
        "omega": [0.4, 0.3],
        "err": [1.5, 0.25]
    }
    
    for ti in tinf_values:
        t0 = time.time()
        print(f"\n--- Fitting with tinf = {ti:.1f} h ---")
        data = load_dataset(excel_path, tinf=ti, enforce_g4=True)
        res = fit_model(data, "1-Compartment IV infusion (multi-dose)", init, {},
                        error_model="combined", n_iter1=250, n_iter2=150, seed=42)
        dt = time.time() - t0
        
        fim = res["fim"]
        cl_est = float(np.exp(fim["CL"][0]))
        cl_se = float(cl_est * fim["CL"][1]) if np.isfinite(fim["CL"][1]) else np.nan
        v_est = float(np.exp(fim["V"][0]))
        v_se = float(v_est * fim["V"][1]) if np.isfinite(fim["V"][1]) else np.nan
        
        om_cl = float(res["omega"][0])
        om_v = float(res["omega"][1])
        err_a = float(res["err"][0])
        err_b = float(res["err"][1])
        m2ll = float(res["ic"]["m2ll"])
        aic = float(res["ic"]["aic"])
        bicc = float(res["ic"]["bicc"])
        
        row = {
            "tinf_h": ti,
            "CL": round(cl_est, 3),
            "CL_SE": round(cl_se, 3),
            "V": round(v_est, 2),
            "V_SE": round(v_se, 2),
            "omega_CL": round(om_cl, 3),
            "omega_V": round(om_v, 3),
            "a": round(err_a, 3),
            "b": round(err_b, 3),
            "m2ll": round(m2ll, 2),
            "AIC": round(aic, 2),
            "BICc": round(bicc, 2),
            "time_s": round(dt, 1)
        }
        results.append(row)
        print(f"tinf={ti}h: CL={cl_est:.3f}, V={v_est:.2f}, om_CL={om_cl:.3f}, om_V={om_v:.3f}, -2LL={m2ll:.2f}")

    df = pd.DataFrame(results)
    out_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "g8_sensitivity_tinf.csv"))
    df.to_csv(out_csv, index=False)
    
    out_json = os.path.abspath(os.path.join(os.path.dirname(__file__), "g8_sensitivity_tinf.json"))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n=== GATE G8 SENSITIVITY TABLE ===")
    print(df.to_string(index=False))
    return results


if __name__ == "__main__":
    run_g8_sensitivity()
