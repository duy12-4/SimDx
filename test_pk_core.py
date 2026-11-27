"""Test lõi: (1) nghiệm giải tích khớp ODE, (2) SAEM phục hồi tham số từ khởi tạo lệch."""
import sys, time
import numpy as np
from scipy.integrate import solve_ivp
from pk_core import *

def test_models_vs_ode():
    t = np.array([[0.0, 0.25, 1, 3, 8, 24, 48]])
    D = 100.0
    def ode(rhs, y0, idx, V):
        s = solve_ivp(rhs, (0, 48), y0, t_eval=t[0], rtol=1e-10, atol=1e-12)
        return s.y[idx] / V
    def two(ka, V1, CL, Q, V2):
        k10, k12, k21 = CL/V1, Q/V1, Q/V2
        return ode(lambda _, y: [-ka*y[0], ka*y[0]-(k10+k12)*y[1]+k21*y[2], k12*y[1]-k21*y[2]], [D,0,0], 1, V1)
    for ka in (1.2, 0.3, 4.0, 0.2):               # gồm ka sát macro-constant
        a = model_2cpt_oral(t, D, ka, 15.0, 4.0, 8.0, 40.0)[0]
        assert np.allclose(a, two(ka, 15., 4., 8., 40.), rtol=1e-5, atol=1e-8), ka
    k = 5.0/20
    assert np.allclose(model_1cpt_oral(t, D, 1.2, 20., 5.)[0],
                       ode(lambda _, y: [-1.2*y[0], 1.2*y[0]-k*y[1]], [D, 0], 1, 20.), rtol=1e-6, atol=1e-9)
    assert np.allclose(model_1cpt_oral(t, D, k, 20., 5.)[0],    # ka == k (trường hợp suy biến)
                       ode(lambda _, y: [-k*y[0], k*y[0]-k*y[1]], [D, 0], 1, 20.), rtol=1e-6, atol=1e-9)
    for Vm, Km in ((10., 2.), (50., 0.1), (1., 20.)):
        a = model_mm(t, D, 20., Vm, Km)[0]
        b = ode(lambda _, y: [-Vm*(y[0]/20)/(Km+y[0]/20)], [D], 0, 20.)
        assert np.allclose(a, b, rtol=1e-6, atol=1e-9), (Vm, Km)
    print("OK models vs ODE")

def recovery(model, theta, om, sa, sp, cov_eff, cov_model, err_model, n=60, seed=3, k1=250, k2=150):
    names = MODELS[model]["params"]
    df, _ = generate_population_data(model, theta, om, sa, sp, cov_eff, n, 100.0,
                                     [0.25, .5, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24], seed=seed)
    init = dict(mu=np.log([theta[p] * 1.6 for p in names]), omega=[0.5] * len(names), err=[0.5, 0.2])
    t0 = time.time()
    res = fit_model(df, model, init, cov_model, err_model, k1, k2, seed=seed)
    print(f"\n== {model} | err={err_model} | true σ=({sa},{sp}) | {time.time()-t0:.0f}s | -2LL={res['ic']['m2ll']:.1f}")
    truth = {**{p: theta[p] for p in names}, **{f"omega_{p}": om[p] for p in names}, "a": sa, "b": sp,
             **{f"beta_{p}_WT": cov_eff.get(p, 0) for p in names}}
    errs = []
    for lab, (est, se) in res["fim"].items():
        tr = truth.get(lab)
        val = np.exp(est) if lab in names else est
        rse = 100 * se if lab in names else 100 * se / max(abs(val), 1e-9)
        z = (val - tr) / (se * (val if lab in names else 1)) if tr is not None else float("nan")
        print(f"  {lab:12s} true={tr!s:>6} est={val:8.3f} RSE={rse:6.1f}%  z={z:5.1f}")
        if tr is not None and lab in names: errs.append(abs(val/tr - 1))
    print("  η-shrink:", np.round(res["diag"]["eta_shrink"], 2), " ε-shrink:", round(res["diag"]["eps_shrink"], 2))
    print("  drift>5%:", [(l, round(r, 3)) for l, ok, r in convergence_report(res) if not ok])
    return res, max(errs)

if __name__ == "__main__":
    test_models_vs_ode()
    th = {"ka": 1.2, "V": 20.0, "CL": 5.0}; om = {"ka": 0.3, "V": 0.3, "CL": 0.3}
    for sa, sp in [(0.3, 0.1), (0.05, 0.3), (1.0, 0.05)]:
        _, e = recovery("1-Compartment (Oral)", th, om, sa, sp, {}, {}, "combined")
        assert e < 0.3, f"θ lệch {e:.0%}"
    # covariate: allometric CL~WT^0.75, V~WT^1
    _, e = recovery("1-Compartment (Oral)", th, om, 0.1, 0.1, {"CL": .75, "V": 1.0},
                    {"CL": ["WT"], "V": ["WT"]}, "combined")
    assert e < 0.3
    print("\nTất cả test qua.")
