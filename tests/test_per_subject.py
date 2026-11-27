"""
tests/test_per_subject.py - Acceptance Gates F1, F2, F3 for Subject Identification Bug.

Independent Reference Solver:
Uses scipy.integrate.solve_ivp (Radau, rtol=1e-11, atol=1e-13) segment-by-segment
between event times (dose starts, dose ends, covariate changes, observations).
DOES NOT use solve_1cpt_infusion or pk_core predict.
"""
import os
import sys
import unittest
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_io import load_dataset
import pk_core as pk


def independent_segment_solve_ivp(doses, default_tinf, t_obs, cov_events, ref_median, beta_cl, phi_cl, phi_v):
    """
    Independent reference solver for 1-Cpt IV infusion with time-varying CLCr_CG.
    Integrates dA/dt = R(t) - (CL(t)/V)*A segment-by-segment using solve_ivp.
    """
    V = float(np.exp(phi_v))
    base_cl = float(np.exp(phi_cl))
    
    # 1. Expand infusion intervals: (t_start, t_end, rate)
    infusions = []
    for d in doses:
        t_d, amt = d[0], d[1]
        tinf_d = d[2] if len(d) > 2 and d[2] is not None and d[2] > 0 else default_tinf
        infusions.append((t_d, t_d + tinf_d, amt / tinf_d))
        
    # 2. Covariate step function for CL(t)
    if cov_events and len(cov_events) > 0:
        base_val = cov_events[0][1]
        base_shift = beta_cl * np.log(base_val / ref_median)
        cl_steps = []
        for te, ve in cov_events:
            curr_shift = beta_cl * np.log(ve / ref_median)
            cl_val = float(np.exp(phi_cl + (curr_shift - base_shift)))
            cl_steps.append((te, cl_val))
    else:
        cl_steps = [(0.0, base_cl)]
        
    def get_rate(t):
        r = 0.0
        for t_s, t_e, rate in infusions:
            if t_s <= t < t_e:
                r += rate
        return r

    def get_cl(t):
        val = cl_steps[0][1]
        for te, cl_val in cl_steps:
            if t >= te:
                val = cl_val
            else:
                break
        return val

    # 3. Collect all jump points
    jump_pts = set([0.0])
    for t_s, t_e, _ in infusions:
        jump_pts.add(t_s)
        jump_pts.add(t_e)
    for te, _ in cl_steps:
        jump_pts.add(te)
    for to in t_obs:
        jump_pts.add(to)
        
    timeline = sorted(list(jump_pts))
    
    # 4. Integrate segment by segment
    A_curr = 0.0
    t_curr = 0.0
    
    obs_map = {}
    
    for i in range(len(timeline) - 1):
        t0 = timeline[i]
        t1 = timeline[i + 1]
        
        # Check if t0 is an observation time
        if t0 in t_obs and t0 not in obs_map:
            obs_map[t0] = A_curr / V
            
        dt = t1 - t0
        if dt > 1e-12:
            mid = t0 + 0.5 * dt
            R = get_rate(mid)
            CL = get_cl(mid)
            k = CL / V
            
            # solve_ivp on [t0, t1]
            sol = solve_ivp(
                lambda t, y: [R - k * y[0]],
                (t0, t1),
                [A_curr],
                method="Radau",
                rtol=1e-11,
                atol=1e-13
            )
            A_curr = float(sol.y[0, -1])
            t_curr = t1
            
    # Check last point
    last_t = timeline[-1]
    if last_t in t_obs and last_t not in obs_map:
        obs_map[last_t] = A_curr / V
        
    return np.array([obs_map[to] for to in t_obs])


class TestPerSubjectCovariate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset_chính_thức.xlsx"))
        raw_data = load_dataset(excel_path, tinf=1.0)
        cls.cov_model = {"CL": ["CLCr_CG"]}
        cls.data = pk.prepare_data(raw_data, cov_model=cls.cov_model, cov_mode="time-varying")
        cls.model_name = "1-Compartment IV infusion (multi-dose)"
        cls.ref_median = cls.data["ref_medians"]["CLCr_CG"]
        cls.beta_cl = 0.8443
        cls.beta = [np.array([np.log(3.0), cls.beta_cl]), np.array([np.log(20.0)])]
        
        # Identify subjects with time-varying CLCr_CG
        cls.varying_sids = []
        for sid in cls.data["ids"]:
            events = cls.data["sub_cov_events"][sid].get("CLCr_CG", [])
            if len(events) > 1:
                vals = [e[1] for e in events]
                if max(vals) != min(vals):
                    cls.varying_sids.append(sid)
                    
        print(f"\n[TestPerSubjectCovariate] Found {len(cls.varying_sids)} subjects with varying CLCr_CG.")

    def test_F1_single_subject_predict_vs_independent_reference(self):
        """
        Gate F1:
        Single-subject predict vs independent segment-wise solve_ivp for >= 10 subjects with varying CLCr.
        Must satisfy max relative error < 1e-8.
        Also must equal the same subject's row from a full-N batch call.
        """
        test_sids = self.varying_sids[:15]  # 15 subjects with varying CLCr
        self.assertGreaterEqual(len(test_sids), 10)
        
        N = self.data["N"]
        # Generate random phi for all N subjects
        rng = np.random.default_rng(2026)
        phi_all = np.column_stack([
            rng.normal(np.log(3.0), 0.3, N),
            rng.normal(np.log(20.0), 0.2, N)
        ])
        
        # Full-N batch predict call
        f_batch_all = pk.predict(
            self.model_name, self.data["t"], self.data["dose"], phi_all,
            data=self.data, beta=self.beta, cov_model=self.cov_model, cov_mode="time-varying"
        )
        
        errors_vs_ref = []
        errors_vs_batch = []
        
        for sid in test_sids:
            idx = int(np.where(self.data["ids"] == sid)[0][0])
            mk = self.data["mask"][idx]
            t_obs = self.data["t"][idx, mk]
            doses = self.data["sub_doses"][sid]
            cov_events = self.data["sub_cov_events"][sid]["CLCr_CG"]
            phi_i = phi_all[idx:idx+1]
            
            # 1. Independent reference solve_ivp
            c_ref = independent_segment_solve_ivp(
                doses, 1.0, t_obs, cov_events, self.ref_median, self.beta_cl,
                phi_i[0, 0], phi_i[0, 1]
            )
            
            # 2. Full-N batch row
            c_batch = f_batch_all[idx, mk]
            
            # Check batch vs ref (this tests that the full-N logic is mathematically correct)
            rel_err_batch = np.max(np.abs(c_batch - c_ref) / np.maximum(c_ref, 1e-6))
            self.assertLess(rel_err_batch, 1e-8, f"Batch call failed vs ref for subject {sid}: {rel_err_batch}")
            
            # 3. Single-subject predict call with explicit subject_ids
            t_row = self.data["t"][idx:idx+1]
            d_row = self.data["dose"][idx:idx+1]
            
            # Verify that omitting subject_ids raises ValueError
            with self.assertRaises(ValueError):
                pk.predict(
                    self.model_name, t_row, d_row, phi_i,
                    data=self.data, beta=self.beta, cov_model=self.cov_model, cov_mode="time-varying"
                )
            
            c_single_pred = pk.predict(
                self.model_name, t_row, d_row, phi_i,
                data=self.data, beta=self.beta, cov_model=self.cov_model, cov_mode="time-varying",
                subject_ids=[sid]
            )[0, mk]
            
            rel_err_ref = float(np.max(np.abs(c_single_pred - c_ref) / np.maximum(c_ref, 1e-6)))
            rel_err_b = float(np.max(np.abs(c_single_pred - c_batch) / np.maximum(c_batch, 1e-6)))
            
            errors_vs_ref.append((sid, rel_err_ref))
            errors_vs_batch.append((sid, rel_err_b))
            
        max_err_ref = max(e[1] for e in errors_vs_ref)
        max_err_batch = max(e[1] for e in errors_vs_batch)
        print(f"[F1] Max relative error single-subject vs ref: {max_err_ref:.4e}")
        print(f"[F1] Max relative error single-subject vs batch: {max_err_batch:.4e}")
        
        # GATE F1 THRESHOLD: < 1e-8
        self.assertLess(max_err_ref, 1e-8, f"F1 FAIL: single-subject predict vs ref error {max_err_ref:.4e} >= 1e-8")
        self.assertLess(max_err_batch, 1e-8, f"F1 FAIL: single-subject predict vs batch error {max_err_batch:.4e} >= 1e-8")

    def test_F2_ipred_vs_independent_reference_all_83_subjects(self):
        """
        Gate F2:
        After fitting CL~CLCr_CG, for ALL 83 subjects:
        IPRED returned by compute_diagnostics must equal independent reference IPRED
        computed from phi_hat, max relative error < 1e-8.
        """
        init = {
            "mu": [np.log(3.0), np.log(20.0)],
            "omega": [0.4, 0.3],
            "err": [1.5, 0.25]
        }
        # Run fit (with fast iterations for unit test)
        res = pk.fit_model(
            self.data, self.model_name, init, self.cov_model,
            error_model="combined", n_iter1=60, n_iter2=30, seed=42, n_is=100
        )
        
        diag = res["diag"]
        ipred_table = diag["table"]
        phi_hat = res["phi_hat"]
        beta = res["beta"]
        beta_cl = float(beta[0][1])
        
        max_err = 0.0
        err_list = []
        
        for idx, sid in enumerate(self.data["ids"]):
            mk = self.data["mask"][idx]
            if not np.any(mk):
                continue
            t_obs = self.data["t"][idx, mk]
            doses = self.data["sub_doses"][sid]
            cov_events = self.data["sub_cov_events"][sid].get("CLCr_CG", [])
            
            c_ref = independent_segment_solve_ivp(
                doses, 1.0, t_obs, cov_events, self.ref_median, beta_cl,
                phi_hat[idx, 0], phi_hat[idx, 1]
            )
            
            sub_diag = ipred_table[ipred_table["ID"] == sid]
            c_ipred = sub_diag["IPRED"].values
            
            rel_err = float(np.max(np.abs(c_ipred - c_ref) / np.maximum(c_ref, 1e-6)))
            err_list.append((sid, rel_err))
            if rel_err > max_err:
                max_err = rel_err
                
        print(f"[F2] Max IPRED relative error vs ref across 83 subjects: {max_err:.4e}")
        self.assertLess(max_err, 1e-8, f"F2 FAIL: IPRED vs ref error {max_err:.4e} >= 1e-8")

    def test_F3_deterministic_consistency_loglik_rows(self):
        """
        Gate F3:
        At fixed phi, loglik_rows through single-subject path (as used in loglik_is/compute_ebe)
        must equal the full-N batch path for every subject, abs diff < 1e-9.
        """
        N = self.data["N"]
        err = [1.5, 0.25]
        phi_all = np.column_stack([
            np.full(N, np.log(3.0)),
            np.full(N, np.log(20.0))
        ])
        
        # Full-N batch path
        ll_batch = pk.loglik_rows(
            phi_all, self.model_name, self.data["t"], self.data["y"],
            self.data["mask"], self.data["dose"], err, error_model="combined",
            const=True, data=self.data, beta=self.beta, cov_model=self.cov_model,
            cov_mode="time-varying"
        )
        
        max_diff = 0.0
        diffs = []
        
        for i, sid in enumerate(self.data["ids"]):
            t_i = self.data["t"][i:i+1]
            y_i = self.data["y"][i:i+1]
            mk_i = self.data["mask"][i:i+1]
            ds_i = self.data["dose"][i:i+1]
            phi_i = phi_all[i:i+1]
            
            # Single-subject call with explicit subject_ids
            with self.assertRaises(ValueError):
                pk.loglik_rows(
                    phi_i, self.model_name, t_i, y_i, mk_i, ds_i, err,
                    error_model="combined", const=True, data=self.data,
                    beta=self.beta, cov_model=self.cov_model, cov_mode="time-varying"
                )

            ll_single = pk.loglik_rows(
                phi_i, self.model_name, t_i, y_i, mk_i, ds_i, err,
                error_model="combined", const=True, data=self.data,
                beta=self.beta, cov_model=self.cov_model, cov_mode="time-varying",
                subject_ids=[sid]
            )
            
            diff = float(np.abs(ll_single[0] - ll_batch[i]))
            diffs.append((sid, diff))
            if diff > max_diff:
                max_diff = diff
                
        print(f"[F3] Max absolute difference in loglik_rows (single vs full-N): {max_diff:.4e}")
        self.assertLess(max_diff, 1e-9, f"F3 FAIL: loglik_rows difference {max_diff:.4e} >= 1e-9")


if __name__ == "__main__":
    unittest.main()
