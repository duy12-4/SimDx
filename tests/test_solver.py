"""
tests/test_solver.py - Automated tests for Gates G1, G2, G3.
G1: Solver vs scipy solve_ivp (rtol=1e-10, atol=1e-12) on >= 200 random scenarios:
    multi-dose, overlapping infusions, CL/V change mid-infusion, observation exactly at infusion end.
    Max relative error < 1e-6.
G2: Closed-form check (single infusion; steady-state q24h) to 1e-9.
G3: Existing test_pk_core.py passes unchanged.
"""
import sys
import os
import unittest
import numpy as np
from scipy.integrate import solve_ivp

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def solve_1cpt_infusion(doses, tinf, t_obs, cl_events, V, v_events=None):
    """
    Event-driven structural solver for 1-Compartment IV infusion with multi-dose,
    overlapping infusions, and time-varying CL/V (LOCF).
    
    doses: list of (t_dose, amount) or (t_dose, amount, tinf_i)
    tinf: default infusion duration if not specified in doses
    t_obs: 1D array-like of observation times
    cl_events: list of (t, CL_val) or float (if constant)
    V: float (baseline volume)
    v_events: optional list of (t, V_val) for time-varying V (LOCF)
    """
    t_obs = np.asarray(t_obs, dtype=float)
    scalar_input = (t_obs.ndim == 0)
    if scalar_input:
        t_obs = t_obs.reshape(1)

    # 1. Parse infusions: list of (t_start, t_end, rate)
    infusions = []
    for d in doses:
        td = float(d[0])
        amt = float(d[1])
        ti = float(d[2]) if len(d) > 2 else float(tinf)
        if ti <= 0:
            raise ValueError(f"Infusion duration must be > 0, got {ti}")
        infusions.append((td, td + ti, amt / ti))

    # 2. Parse CL and V events
    if isinstance(cl_events, (int, float)):
        cl_sorted = [(0.0, float(cl_events))]
    else:
        cl_sorted = sorted([(float(t), float(v)) for t, v in cl_events], key=lambda x: x[0])
        if not cl_sorted or cl_sorted[0][0] > 0.0:
            first_val = cl_sorted[0][1] if cl_sorted else 1.0
            cl_sorted.insert(0, (0.0, first_val))

    if v_events is None or isinstance(v_events, (int, float)):
        base_v = float(V if v_events is None else v_events)
        v_sorted = [(0.0, base_v)]
    else:
        v_sorted = sorted([(float(t), float(v)) for t, v in v_events], key=lambda x: x[0])
        if not v_sorted or v_sorted[0][0] > 0.0:
            first_val = v_sorted[0][1] if v_sorted else float(V)
            v_sorted.insert(0, (0.0, first_val))

    # 3. Build unique sorted time grid of all events
    t_set = set([0.0])
    for ts, te, _ in infusions:
        t_set.add(ts)
        t_set.add(te)
    for tc, _ in cl_sorted:
        t_set.add(tc)
    for tv, _ in v_sorted:
        t_set.add(tv)
    for to in t_obs:
        t_set.add(to)

    t_grid = np.array(sorted(t_set))
    cl_t = np.array([c[0] for c in cl_sorted])
    cl_v = np.array([c[1] for c in cl_sorted])
    v_t = np.array([c[0] for c in v_sorted])
    v_v = np.array([c[1] for c in v_sorted])

    # 4. Integrate analytically between events
    A = 0.0
    conc_map = {}

    for i in range(len(t_grid)):
        t_curr = t_grid[i]
        
        # Determine V at t_curr (LOCF)
        idx_v_curr = np.searchsorted(v_t, t_curr, side='right') - 1
        V_curr = v_v[max(0, idx_v_curr)]
        
        if t_curr in t_obs:
            conc_map[t_curr] = A / V_curr

        if i == len(t_grid) - 1:
            break

        t_next = t_grid[i + 1]
        dt = t_next - t_curr
        if dt <= 0:
            continue

        # Constant active parameters on (t_curr, t_next)
        t_mid = (t_curr + t_next) / 2.0
        R = sum(rate for ts, te, rate in infusions if ts <= t_mid < te)
        
        idx_cl = np.searchsorted(cl_t, t_mid, side='right') - 1
        CL_active = cl_v[max(0, idx_cl)]
        
        idx_v = np.searchsorted(v_t, t_mid, side='right') - 1
        V_active = v_v[max(0, idx_v)]
        
        k = CL_active / V_active
        kd = k * dt
        exp_kd = np.exp(-kd)
        
        # Robust Taylor series when kd is tiny
        if kd < 1e-6:
            phi = dt * (1.0 - 0.5 * kd + (kd**2) / 6.0 - (kd**3) / 24.0)
        else:
            phi = (1.0 - exp_kd) / k
            
        A = A * exp_kd + R * phi

    res = np.array([conc_map[to] for to in t_obs])
    return res[0] if scalar_input else res


def solve_ode_piecewise(doses, tinf, t_obs, cl_events, V, v_events=None):
    """
    Piecewise ODE solver using scipy.integrate.solve_ivp between critical transitions.
    """
    infusions = []
    for d in doses:
        td = float(d[0])
        amt = float(d[1])
        ti = float(d[2]) if len(d) > 2 else float(tinf)
        infusions.append((td, td + ti, amt / ti))
        
    if isinstance(cl_events, (int, float)):
        cl_sorted = [(0.0, float(cl_events))]
    else:
        cl_sorted = sorted([(float(t), float(v)) for t, v in cl_events], key=lambda x: x[0])
        if not cl_sorted or cl_sorted[0][0] > 0.0:
            cl_sorted.insert(0, (0.0, cl_sorted[0][1]))

    if v_events is None or isinstance(v_events, (int, float)):
        v_sorted = [(0.0, float(V if v_events is None else v_events))]
    else:
        v_sorted = sorted([(float(t), float(v)) for t, v in v_events], key=lambda x: x[0])
        if not v_sorted or v_sorted[0][0] > 0.0:
            v_sorted.insert(0, (0.0, v_sorted[0][1]))

    t_set = set([0.0])
    for ts, te, _ in infusions:
        t_set.add(ts)
        t_set.add(te)
    for tc, _ in cl_sorted:
        t_set.add(tc)
    for tv, _ in v_sorted:
        t_set.add(tv)
    for to in t_obs:
        t_set.add(to)

    t_grid = np.array(sorted(t_set))
    cl_t = np.array([c[0] for c in cl_sorted])
    cl_v = np.array([c[1] for c in cl_sorted])
    v_t = np.array([c[0] for c in v_sorted])
    v_v = np.array([c[1] for c in v_sorted])

    A = 0.0
    conc_map = {}

    for i in range(len(t_grid)):
        t_curr = t_grid[i]
        idx_v_curr = np.searchsorted(v_t, t_curr, side='right') - 1
        V_curr = v_v[max(0, idx_v_curr)]
        
        if t_curr in t_obs:
            conc_map[t_curr] = A / V_curr
            
        if i == len(t_grid) - 1:
            break
            
        t_next = t_grid[i + 1]
        dt = t_next - t_curr
        if dt <= 0:
            continue
            
        t_mid = (t_curr + t_next) / 2.0
        R = sum(rate for ts, te, rate in infusions if ts <= t_mid < te)
        idx_cl = np.searchsorted(cl_t, t_mid, side='right') - 1
        CL_active = cl_v[max(0, idx_cl)]
        idx_v = np.searchsorted(v_t, t_mid, side='right') - 1
        V_active = v_v[max(0, idx_v)]
        k = CL_active / V_active
        
        # solve ODE dA/dt = R - k*A on [t_curr, t_next]
        sol = solve_ivp(
            lambda t, y: [R - k * y[0]],
            (t_curr, t_next),
            [A],
            method='Radau',
            rtol=1e-10,
            atol=1e-12
        )
        A = float(sol.y[0, -1])

    return np.array([conc_map[to] for to in t_obs])


class TestSolver(unittest.TestCase):
    
    def test_g2_closed_form(self):
        """
        Gate G2: Closed-form check to 1e-9.
        (1) Single infusion closed-form.
        (2) Multi-dose steady-state q24h closed-form.
        """
        # (1) Single infusion closed form
        D, tinf, CL, V = 1000.0, 1.5, 4.0, 45.0
        k = CL / V
        R = D / tinf
        
        # Times: pre-infusion, during infusion, at infusion end, post-infusion
        t_obs = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 12.0, 24.0])
        
        c_analytical = np.zeros_like(t_obs)
        for i, t in enumerate(t_obs):
            if t <= tinf:
                c_analytical[i] = (R / CL) * (1.0 - np.exp(-k * t))
            else:
                c_analytical[i] = (R / CL) * (1.0 - np.exp(-k * tinf)) * np.exp(-k * (t - tinf))
                
        c_solver = solve_1cpt_infusion([(0.0, D, tinf)], tinf, t_obs, CL, V)
        diff_single = np.max(np.abs(c_solver - c_analytical))
        self.assertLess(diff_single, 1e-9, f"Single infusion diff {diff_single} >= 1e-9")
        
        # (2) Steady-state q24h closed form
        tau = 24.0
        c_min_ss = (R / CL) * ((1.0 - np.exp(-k * tinf)) * np.exp(-k * (tau - tinf))) / (1.0 - np.exp(-k * tau))
        
        # Simulate 30 doses to reach numerical steady state
        n_doses = 30
        doses = [(i * tau, D, tinf) for i in range(n_doses)]
        t_rel_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 6.0, 12.0, 24.0])
        t_ss = (n_doses - 1) * tau + t_rel_grid
        
        c_ss_analytical = np.zeros_like(t_ss)
        for i, t_rel in enumerate(t_rel_grid):
            if t_rel <= tinf:
                c_ss_analytical[i] = (R / CL) * (1.0 - np.exp(-k * t_rel)) + c_min_ss * np.exp(-k * t_rel)
            else:
                c_ss_analytical[i] = (R / CL) * (1.0 - np.exp(-k * tinf)) * np.exp(-k * (t_rel - tinf)) + c_min_ss * np.exp(-k * t_rel)
                
        c_ss_solver = solve_1cpt_infusion(doses, tinf, t_ss, CL, V)
        diff_ss = float(np.max(np.abs(c_ss_solver - c_ss_analytical)))
        self.assertLess(diff_ss, 1e-9, f"Steady-state diff {diff_ss} >= 1e-9")
        print(f"G2 PASS: Single infusion max diff={diff_single:.2e}, Steady-state max diff={diff_ss:.2e}")

    def test_g1_solver_vs_solve_ivp_200_scenarios(self):
        """
        Gate G1: Solver vs scipy solve_ivp (rtol=1e-10, atol=1e-12) on >= 200 random scenarios:
        multi-dose, overlapping infusions, CL/V change mid-infusion, observation exactly at infusion end.
        Max relative error < 1e-6.
        """
        rng = np.random.default_rng(2026)
        n_scenarios = 250
        max_rel_errors = []
        
        for sc in range(n_scenarios):
            V_base = float(rng.uniform(15.0, 60.0))
            tinf = float(rng.uniform(0.5, 2.0))
            
            # Generate 1 to 5 doses, some potentially overlapping
            n_doses = int(rng.integers(1, 6))
            doses = []
            curr_t = float(rng.uniform(0.0, 5.0))
            for _ in range(n_doses):
                amt = float(rng.uniform(250.0, 2000.0))
                dur = float(rng.uniform(0.5, 2.5))
                doses.append((curr_t, amt, dur))
                # Interval to next dose: can be less than dur (overlapping!)
                curr_t += float(rng.uniform(0.2, 12.0))
                
            # Time-varying CL events (1 to 4 changes)
            n_cl = int(rng.integers(1, 5))
            cl_events = []
            t_cl = 0.0
            for _ in range(n_cl):
                cl_val = float(rng.uniform(1.0, 15.0))
                cl_events.append((t_cl, cl_val))
                # Next CL change time, can be mid-infusion
                t_cl += float(rng.uniform(1.0, 15.0))
                
            # Time-varying V events (1 to 3 changes)
            n_v = int(rng.integers(1, 4))
            v_events = []
            t_v = 0.0
            for _ in range(n_v):
                v_val = float(rng.uniform(15.0, 60.0))
                v_events.append((t_v, v_val))
                t_v += float(rng.uniform(2.0, 20.0))
                
            # Observations: include observations EXACTLY at infusion ends, mid-infusion, and post-infusion
            obs_times = set()
            for td, amt, dur in doses:
                obs_times.add(td + dur)  # EXACTLY at infusion end
                obs_times.add(td + dur * 0.5)  # mid-infusion
                obs_times.add(td + dur + 0.1)  # shortly after
                obs_times.add(td + dur + float(rng.uniform(0.5, 6.0)))
            # Add random observation times
            for _ in range(5):
                obs_times.add(float(rng.uniform(0.0, curr_t + 10.0)))
                
            t_obs = np.array(sorted(obs_times))
            
            c_sol = solve_1cpt_infusion(doses, tinf, t_obs, cl_events, V_base, v_events)
            c_ode = solve_ode_piecewise(doses, tinf, t_obs, cl_events, V_base, v_events)
            
            # Compute relative error where c_ode > 1e-6
            denom = np.maximum(np.abs(c_ode), 1e-6)
            rel_err = np.max(np.abs(c_sol - c_ode) / denom)
            max_rel_errors.append(rel_err)
            
            self.assertLess(rel_err, 1e-6, f"Scenario {sc} failed: rel_err={rel_err:.2e} >= 1e-6")

        max_all = max(max_rel_errors)
        mean_all = np.mean(max_rel_errors)
        print(f"G1 PASS: {n_scenarios} scenarios. Max rel error={max_all:.2e}, Mean rel error={mean_all:.2e}")

    def test_g3_existing_test_pk_core(self):
        """
        Gate G3: Existing test_pk_core.py passes unchanged.
        """
        import test_pk_core
        # Run test_models_vs_ode
        test_pk_core.test_models_vs_ode()
        # Run recovery test
        th = {"ka": 1.2, "V": 20.0, "CL": 5.0}
        om = {"ka": 0.3, "V": 0.3, "CL": 0.3}
        for sa, sp in [(0.3, 0.1), (0.05, 0.3)]:
            _, e = test_pk_core.recovery("1-Compartment (Oral)", th, om, sa, sp, {}, {}, "combined", n=40, k1=150, k2=80)
            self.assertLess(e, 0.3, f"Recovery error {e:.1%}")
        print("G3 PASS: Existing test_pk_core models vs ODE and SAEM recovery pass.")


if __name__ == "__main__":
    unittest.main()
