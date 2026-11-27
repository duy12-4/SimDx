"""
pk_core.py - Lõi NLME / SAEM cho Mini Monolix (không phụ thuộc Streamlit).

Mô hình:  y_ij = f(t_ij, psi_i) + g(f) * eps,   g = sqrt(a^2 + b^2 f^2)
          log psi_i = X_i @ beta + eta_i,       eta_i ~ N(0, diag(omega^2))
SAEM:     E-step = MCMC 3 kernel (prior, random-walk từng thành phần, random-walk toàn phần)
          M-step = thống kê đủ (SA) cho beta/omega, tối ưu likelihood cho (a, b).
Sau SAEM: EBE (mode), -2LL (importance sampling), SE/RSE (FIM tuyến tính hoá),
          PRED/IPRED/IWRES/PWRES/NPDE, shrinkage, VPC.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, t as student_t

LOG2PI = np.log(2 * np.pi)
COVARIATES = ["WT", "AGE", "SEX"]


# ───────────────────────── Structural models (closed-form, vector hoá) ─────────────────────────
# t: [M,T]; dose: [M,1] hoặc scalar; tham số: [M,1]  ->  nồng độ [M,T]
def model_1cpt_oral(t, dose, ka, V, CL):
    k = CL / V
    d = ka - k
    x = d * t
    small = np.abs(x) < 1e-6                      # ka ~ k: dùng khai triển thay vì chia ~0
    f = np.where(small, t * np.exp(-k * t) * (1 - x / 2),
                 (np.exp(-k * t) - np.exp(-ka * t)) / np.where(small, 1.0, d))
    return dose * ka / V * f


def model_1cpt_iv(t, dose, V, CL):
    return dose / V * np.exp(-CL / V * t)


def model_2cpt_oral(t, dose, ka, V1, CL, Q, V2):
    k10, k12, k21 = CL / V1, Q / V1, Q / V2
    S, Pr = k10 + k12 + k21, k10 * k21
    disc = np.sqrt(np.maximum(S * S - 4 * Pr, 1e-30))
    al, be = (S + disc) / 2, (S - disc) / 2
    for r in (al, be):                            # ka trùng nghiệm macro-constant thì mẫu số = 0
        ka = np.where(np.abs(ka - r) < 1e-5 * r, ka * (1 + 1e-4), ka)
    A = (k21 - al) / ((ka - al) * (be - al))
    B = (k21 - be) / ((ka - be) * (al - be))
    Cc = (k21 - ka) / ((al - ka) * (be - ka))
    return dose * ka / V1 * (A * np.exp(-al * t) + B * np.exp(-be * t) + Cc * np.exp(-ka * t))


def model_mm(t, dose, V, Vmax, Km):
    """IV bolus, thải trừ Michaelis-Menten, Vmax [mg/h]. Nghiệm Lambert-W: w + ln w = x, w = C/Km."""
    c0 = dose / V
    x = np.log(c0 / Km) + c0 / Km - Vmax * t / (V * Km)
    u = np.where(x > 1, np.log(np.maximum(x, 1e-300)), x)   # bắt đầu bên phải nghiệm -> Newton đơn điệu
    with np.errstate(over="ignore", under="ignore"):
        for _ in range(30):                       # giải e^u + u = x, u = ln w
            eu = np.exp(u)
            u = u - (eu + u - x) / (eu + 1)
        return Km * np.exp(u)


def solve_1cpt_infusion(doses, tinf, t_obs, cl_events, V, v_events=None):
    """
    Event-driven structural solver for 1-Compartment IV infusion with multi-dose,
    overlapping infusions, and time-varying CL/V (LOCF).
    """
    t_obs = np.asarray(t_obs, dtype=float)
    scalar_input = (t_obs.ndim == 0)
    if scalar_input:
        t_obs = t_obs.reshape(1)

    infusions = []
    for d in doses:
        td = float(d[0])
        amt = float(d[1])
        ti = float(d[2]) if len(d) > 2 else float(tinf)
        if ti <= 0:
            raise ValueError(f"Infusion duration must be > 0, got {ti}")
        infusions.append((td, td + ti, amt / ti))

    if isinstance(cl_events, (int, float, np.floating)):
        cl_sorted = [(0.0, float(cl_events))]
    else:
        cl_sorted = sorted([(float(t), float(v)) for t, v in cl_events], key=lambda x: x[0])
        if not cl_sorted or cl_sorted[0][0] > 0.0:
            first_val = cl_sorted[0][1] if cl_sorted else 1.0
            cl_sorted.insert(0, (0.0, first_val))

    if v_events is None or isinstance(v_events, (int, float, np.floating)):
        base_v = float(V if v_events is None else v_events)
        v_sorted = [(0.0, base_v)]
    else:
        v_sorted = sorted([(float(t), float(v)) for t, v in v_events], key=lambda x: x[0])
        if not v_sorted or v_sorted[0][0] > 0.0:
            first_val = v_sorted[0][1] if v_sorted else float(V)
            v_sorted.insert(0, (0.0, first_val))

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
        kd = k * dt
        exp_kd = np.exp(-kd)
        if kd < 1e-6:
            phi_val = dt * (1.0 - 0.5 * kd + (kd**2) / 6.0 - (kd**3) / 24.0)
        else:
            phi_val = (1.0 - exp_kd) / k
        A = A * exp_kd + R * phi_val

    res = np.array([conc_map[to] for to in t_obs])
    return res[0] if scalar_input else res


MODELS = {
    "1-Compartment (Oral)": dict(fn=model_1cpt_oral, params=["ka", "V", "CL"],
                                 defaults=[1.2, 20.0, 5.0], units=["1/h", "L", "L/h"],
                                 ranges=[(0.1, 5.0), (5.0, 100.0), (1.0, 30.0)]),
    "1-Compartment (IV Bolus)": dict(fn=model_1cpt_iv, params=["V", "CL"],
                                     defaults=[20.0, 5.0], units=["L", "L/h"],
                                     ranges=[(5.0, 100.0), (1.0, 30.0)]),
    "2-Compartment (Oral)": dict(fn=model_2cpt_oral, params=["ka", "V1", "CL", "Q", "V2"],
                                 defaults=[1.2, 15.0, 4.0, 8.0, 40.0],
                                 units=["1/h", "L", "L/h", "L/h", "L"],
                                 ranges=[(0.1, 5.0), (5.0, 50.0), (1.0, 20.0), (1.0, 30.0), (10.0, 150.0)]),
    "Michaelis-Menten": dict(fn=model_mm, params=["V", "Vmax", "Km"],
                             defaults=[20.0, 10.0, 2.0], units=["L", "mg/h", "mg/L"],
                             ranges=[(5.0, 50.0), (1.0, 50.0), (0.1, 20.0)]),
    "1-Compartment IV infusion (multi-dose)": dict(fn=None, params=["CL", "V"],
                                                   defaults=[5.0, 40.0], units=["L/h", "L"],
                                                   ranges=[(0.5, 30.0), (5.0, 150.0)]),
}


def predict(model_name, t, dose, phi, data=None, sub_doses=None, sub_cov=None,
            beta=None, cov_model=None, cov_mode="time-varying", tinf=1.0, subject_ids=None):
    """phi [M,P] = log tham số cá thể. Trả về [M,T]; giá trị không hữu hạn -> 0 / 1e30."""
    if model_name == "1-Compartment IV infusion (multi-dose)":
        M = phi.shape[0]
        T = t.shape[1] if hasattr(t, "shape") and t.ndim > 1 else len(t)
        out = np.zeros((M, T), dtype=float)
        
        ids = data.get("ids", None) if data is not None else None
        N = len(ids) if ids is not None else M
        default_tinf = data.get("tinf", tinf) if data is not None else tinf
        ref_medians = data.get("ref_medians", {}) if data is not None else {}
        active_cov_mode = cov_mode if cov_mode is not None else (data.get("cov_mode", "time-varying") if data else "time-varying")
        
        has_time_varying = (
            active_cov_mode == "time-varying" and
            data is not None and
            "sub_cov_events" in data and
            cov_model is not None and
            bool(cov_model.get("CL", []))
        )
        
        if subject_ids is None:
            if M == N:
                subject_ids = ids if ids is not None else np.arange(N)
            elif has_time_varying:
                raise ValueError("subject_ids must be explicitly provided when M != N with time-varying covariates")

        for m in range(M):
            if subject_ids is not None:
                s_val = subject_ids[m]
                if ids is not None and isinstance(s_val, (int, np.integer)) and s_val not in data.get("sub_doses", {}) and 0 <= s_val < len(ids):
                    sid = ids[s_val]
                else:
                    sid = s_val
            else:
                subj_idx = m % N
                sid = ids[subj_idx] if ids is not None else subj_idx
            
            if sub_doses is not None and m < len(sub_doses):
                d_list = sub_doses[m]
            elif data is not None and "sub_doses" in data and sid in data["sub_doses"]:
                d_list = data["sub_doses"][sid]
            else:
                d_val = float(dose[m, 0] if hasattr(dose, 'shape') and dose.ndim > 1 else (dose[m] if hasattr(dose, '__len__') else dose))
                d_list = [(0.0, d_val, default_tinf)]
                
            phi_cl = phi[m, 0]
            phi_v = phi[m, 1]
            V_val = float(np.exp(phi_v))
            
            t_row = t[m] if hasattr(t, "shape") and t.ndim > 1 else t
            
            cl_events = float(np.exp(phi_cl))
            
            if active_cov_mode == "time-varying" and data is not None and "sub_cov_events" in data:
                cl_covs = cov_model.get("CL", []) if cov_model else []
                beta_cl = None
                if beta is not None and len(beta) > 0 and len(beta[0]) > 1:
                    beta_cl = beta[0][1:]
                    
                if "CLCr_CG" in cl_covs:
                    if sid not in data["sub_cov_events"]:
                        raise KeyError(f"Subject {sid} not found in data['sub_cov_events']")
                    if "CLCr_CG" in data["sub_cov_events"][sid]:
                        events = data["sub_cov_events"][sid]["CLCr_CG"]
                        if events:
                            if "CLCr_CG" not in ref_medians:
                                raise KeyError("Missing reference median for 'CLCr_CG' in data['ref_medians']")
                            ref_val = float(ref_medians["CLCr_CG"])
                            
                            if beta_cl is None:
                                raise ValueError("Missing beta coefficient for covariate 'CLCr_CG'")
                            cov_idx = cl_covs.index("CLCr_CG")
                            if cov_idx >= len(beta_cl):
                                raise ValueError(f"Beta coefficient for 'CLCr_CG' at index {cov_idx} missing from beta_cl (length {len(beta_cl)})")
                            b_coef = float(beta_cl[cov_idx])
                            
                            base_val = events[0][1]
                            base_shift = b_coef * np.log(base_val / ref_val)
                            cl_events = []
                            for te, ve in events:
                                curr_shift = b_coef * np.log(ve / ref_val)
                                cl_k = float(np.exp(phi_cl + (curr_shift - base_shift)))
                                cl_events.append((te, cl_k))
                            
            c_pred = solve_1cpt_infusion(d_list, default_tinf, t_row, cl_events, V_val)
            out[m, :] = c_pred
            
        return np.nan_to_num(out, nan=0.0, posinf=1e30, neginf=0.0)

    m = MODELS[model_name]
    psi = np.exp(phi)
    kw = {p: psi[:, [j]] for j, p in enumerate(m["params"])}
    with np.errstate(all="ignore"):
        f = m["fn"](t, dose, **kw)
    return np.nan_to_num(f, nan=0.0, posinf=1e30, neginf=0.0)


def resid_sd(f, err, error_model="combined"):
    """
    Sai số dư:
    - constant: g = a
    - proportional: g = b * f
    - combined1: g = a + b * f
    - combined / combined2: g = sqrt(a^2 + b^2 * f^2)
    """
    a = err[0]
    b = err[1]
    if error_model == "constant":
        return np.maximum(a, 1e-8)
    elif error_model == "proportional":
        return np.maximum(b * np.abs(f), 1e-8)
    elif error_model == "combined1":
        return np.maximum(a + b * np.abs(f), 1e-8)
    else:
        return np.maximum(np.sqrt(a ** 2 + (b * f) ** 2), 1e-8)


def loglik_rows(phi, model_name, t, y, mask, dose, err, error_model="combined", const=False,
                data=None, sub_doses=None, sub_cov=None, beta=None, cov_model=None, cov_mode="time-varying", tinf=1.0,
                subject_ids=None):
    """log p(y_i | phi_i) cho từng hàng phi. t,y,mask,dose broadcast với phi."""
    f = predict(model_name, t, dose, phi, data=data, sub_doses=sub_doses, sub_cov=sub_cov,
                beta=beta, cov_model=cov_model, cov_mode=cov_mode, tinf=tinf, subject_ids=subject_ids)
    sd = resid_sd(f, err, error_model=error_model)
    l = -np.log(sd) - 0.5 * ((y - f) / sd) ** 2 - (0.5 * LOG2PI if const else 0.0)
    l = np.where(mask, l, 0.0)
    return np.where(np.isfinite(l), l, -1e30).sum(axis=1)


# ───────────────────────── Dữ liệu ─────────────────────────
def cov_transform(wt, age, sex):
    """Covariate đã căn giữa: log(WT/70), (AGE-45)/10, SEX=1 nếu nữ. Typical value ứng với 70kg/45t/nam."""
    return {"WT": np.log(np.asarray(wt, float) / 70), "AGE": (np.asarray(age, float) - 45) / 10,
            "SEX": (np.asarray(sex) == "F").astype(float)}


def generate_population_data(model_name, theta, omega, sigma_add, sigma_prop, cov_effects,
                             n_subjects, dose, t_grid, n_obs=None, seed=42):
    """Mô phỏng quần thể. cov_effects: {param: beta của log(WT/70)}. Không cắt DV (cắt sẽ làm lệch ước lượng)."""
    rng = np.random.default_rng(seed)
    names = MODELS[model_name]["params"]
    P, N = len(names), n_subjects
    t_grid = np.asarray(t_grid, float)
    n_obs = len(t_grid) if n_obs is None else min(n_obs, len(t_grid))
    wt = np.clip(rng.normal(70, 15, N), 40, 120).round(1)
    age = rng.integers(20, 70, N)
    sex = rng.choice(["M", "F"], N)
    cv = cov_transform(wt, age, sex)
    om = np.array([omega[p] for p in names])
    mu = np.log([theta[p] for p in names])
    bw = np.array([cov_effects.get(p, 0.0) for p in names])
    phi = mu + cv["WT"][:, None] * bw + om * rng.standard_normal((N, P))
    t = np.array([np.sort(rng.choice(t_grid, n_obs, replace=False)) for _ in range(N)])
    f = predict(model_name, t, np.full((N, 1), float(dose)), phi)
    err = np.array([sigma_add, sigma_prop])
    y = f + resid_sd(f, err) * rng.standard_normal(f.shape)
    df = pd.DataFrame({"ID": np.repeat(np.arange(1, N + 1), n_obs), "TIME": t.ravel(), "DV": y.ravel(),
                       "DOSE": float(dose), "WT": np.repeat(wt, n_obs),
                       "AGE": np.repeat(age, n_obs), "SEX": np.repeat(sex, n_obs)})
    return df, phi


def prepare_data(data_source, cov_model=None, cov_mode="time-varying"):
    """DataFrame dài hoặc dict load_dataset -> ma trận đệm [N,T] + mask."""
    if isinstance(data_source, dict) and "sub_doses" in data_source:
        d = dict(data_source)
        ref_medians = {}
        bdf = d["baseline_cov"]
        for c in ["TBW", "Height", "BMI", "IBW", "AdBW", "Dosing weight", "BSA", "Age", "CLCr_CG", "Creatinine", "eGFR_CKD_EPI", "CLCr_Je"]:
            if c in bdf.columns:
                ref_medians[c] = float(np.nanmedian(bdf[c]))
            elif f"{c}_base" in bdf.columns:
                ref_medians[c] = float(np.nanmedian(bdf[f"{c}_base"]))
        
        cov_dict = {}
        for c in ["TBW", "Age"]:
            if c in bdf.columns:
                ref = ref_medians.get(c, 1.0)
                cov_dict[c] = np.log(np.maximum(bdf[c].values.astype(float), 1e-4) / ref)
        if "CLCr_CG" in bdf.columns:
            if "CLCr_CG" not in ref_medians:
                raise KeyError("Missing reference median for 'CLCr_CG'")
            ref = ref_medians["CLCr_CG"]
            cov_dict["CLCr_CG"] = np.log(np.maximum(bdf["CLCr_CG"].values.astype(float), 1e-4) / ref)
        elif "CLCr_CG_base" in bdf.columns:
            if "CLCr_CG" not in ref_medians:
                raise KeyError("Missing reference median for 'CLCr_CG'")
            ref = ref_medians["CLCr_CG"]
            cov_dict["CLCr_CG"] = np.log(np.maximum(bdf["CLCr_CG_base"].values.astype(float), 1e-4) / ref)
            
        for c in ["Sepsis", "Sex"]:
            if c in bdf.columns:
                cov_dict[c] = bdf[c].values.astype(float)

        d["cov"] = cov_dict
        d["ref_medians"] = ref_medians
        d["cov_mode"] = cov_mode
        d["dose"] = np.zeros((d["N"], 1))
        return d

    df = data_source
    ids = np.sort(df["ID"].unique())
    N = len(ids)
    groups = [df[df["ID"] == i].sort_values("TIME") for i in ids]
    T = max(len(g) for g in groups)
    t, y = np.zeros((N, T)), np.zeros((N, T))
    mask = np.zeros((N, T), bool)
    dose = np.zeros((N, 1))
    for n, g in enumerate(groups):
        k = len(g)
        t[n, :k], y[n, :k], mask[n, :k] = g["TIME"].values, g["DV"].values, True
        t[n, k:] = g["TIME"].values[-1]           # đệm bằng thời điểm hợp lệ
        dose[n, 0] = g["DOSE"].iloc[0]
    first = pd.DataFrame([g.iloc[0] for g in groups])
    return dict(ids=ids, N=N, T=T, t=t, y=y, mask=mask, dose=dose, nobs=int(mask.sum()),
                cov=cov_transform(first["WT"], first["AGE"], first["SEX"]))


def design_matrices(data, names, cov_model):
    """X_p = [1, các covariate của tham số p]. cov_model: {param: ['WT', ...]}."""
    return [np.column_stack([np.ones(data["N"])] + [data["cov"][c] for c in cov_model.get(p, [])])
            for p in names]


def pop_mean_phi(res):
    return np.column_stack([x @ b for x, b in zip(res["X"], res["beta"])])


def err_free(error_model):
    if error_model == "constant":
        return True, False
    elif error_model == "proportional":
        return False, True
    else:  # combined, combined1, combined2
        return True, True


# ───────────────────────── SAEM ─────────────────────────
def update_err(Sr2, Sf2, mask, err, error_model):
    """M-step sai số dư từ thống kê SA của (y-f)^2 và f^2 (không phụ thuộc sigma hiện tại)."""
    r2, f2 = Sr2[mask], Sf2[mask]
    if error_model == "constant":
        return np.array([np.sqrt(max(float(r2.mean()), 1e-8)), 0.0])
    if error_model == "proportional":
        return np.array([0.0, np.sqrt(max(float(np.mean(r2 / np.maximum(f2, 1e-10))), 1e-8))])
    if error_model == "combined1":
        f_vals = np.sqrt(np.maximum(f2, 0.0))
        def obj1(x):
            g = np.exp(x[0]) + np.exp(x[1]) * f_vals
            return np.sum(np.log(np.maximum(g, 1e-8)) + 0.5 * r2 / np.maximum(g ** 2, 1e-16))
        init_x = np.log(np.maximum(err, 1e-3))
        r = minimize(obj1, init_x, method="L-BFGS-B", bounds=[(-7, 3), (-7, 1)])
        return np.exp(r.x)

    def obj(x):
        v = np.exp(2 * x[0]) + np.exp(2 * x[1]) * f2
        return np.sum(0.5 * np.log(v) + 0.5 * r2 / v)
    r = minimize(obj, np.log(np.maximum(err, 1e-3)), method="L-BFGS-B", bounds=[(-7, 3), (-7, 1)])
    return np.exp(r.x)


def run_saem(data, model_name, init, cov_model, error_model="combined",
             n_iter1=250, n_iter2=150, n_chains=1, seed=1, progress=None):
    """init: {'mu': log-typical [P], 'omega': [P], 'err': [a, b]}.
    Pha 1 (gamma=1, simulated annealing cho phương sai) rồi pha 2 (gamma=1/(k-K1))."""
    rng = np.random.default_rng(seed)
    names = MODELS[model_name]["params"]
    P, N, T, C = len(names), data["N"], data["T"], n_chains
    X = design_matrices(data, names, cov_model)
    XtX = [x.T @ x for x in X]
    idx = np.tile(np.arange(N), C)                # hàng m = chain*N + subject
    M = N * C
    tM, yM, mkM, dM = data["t"][idx], data["y"][idx], data["mask"][idx], data["dose"][idx]
    XM = [x[idx] for x in X]
    mask = data["mask"]

    sub_doses_M = [data["sub_doses"][sid] for sid in data["ids"][idx]] if "sub_doses" in data else None
    sub_cov_M = [data["sub_cov_events"][sid] for sid in data["ids"][idx]] if "sub_cov_events" in data else None
    sub_ids_M = data["ids"][idx] if "ids" in data else idx
    cov_mode = data.get("cov_mode", "time-varying")
    tinf = data.get("tinf", 1.0)

    beta = [np.r_[init["mu"][j], np.zeros(X[j].shape[1] - 1)] for j in range(P)]
    omega = np.array(init["omega"], float)
    err = np.array(init["err"], float)
    fa, fb = err_free(error_model)
    err = err * np.array([fa, fb])
    mean = np.column_stack([XM[j] @ beta[j] for j in range(P)])
    phi = mean + omega * rng.standard_normal((M, P))

    def ll(ph, cur_beta=None):
        b = cur_beta if cur_beta is not None else beta
        return loglik_rows(ph, model_name, tM, yM, mkM, dM, err,
                           error_model=error_model, data=data,
                           sub_doses=sub_doses_M, sub_cov=sub_cov_M,
                           beta=b, cov_model=cov_model, cov_mode=cov_mode, tinf=tinf,
                           subject_ids=sub_ids_M)

    cur = ll(phi)

    def accept(prop, lp, dprior):
        a = np.log(rng.random(M)) < lp - cur + dprior
        return a

    c2, c3 = np.ones(P), 0.5                      # scale RW (nhân với omega), thích nghi ở pha 1
    S_xphi = [np.zeros(x.shape[1]) for x in X]
    S_phi2, Sr2, Sf2 = np.zeros(P), np.zeros((N, T)), np.zeros((N, T))
    cm_sum, cm2_sum, n_cm = np.zeros((N, P)), np.zeros((N, P)), 0
    hist = {"theta": [], "beta": [], "omega": [], "err": []}
    K = n_iter1 + n_iter2

    for k in range(1, K + 1):
        phase1 = k <= n_iter1
        gamma = 1.0 if phase1 else 1.0 / (k - n_iter1)

        # ── E-step: kernel 1 (prior) x2 ──
        for _ in range(2):
            prop = mean + omega * rng.standard_normal((M, P))
            lp = ll(prop)
            a = accept(prop, lp, 0.0)
            phi[a], cur[a] = prop[a], lp[a]
        # kernel 2: random-walk từng thành phần x2
        rates = np.zeros(P)
        for _ in range(2):
            for j in range(P):
                prop = phi.copy()
                prop[:, j] += c2[j] * omega[j] * rng.standard_normal(M)
                lp = ll(prop)
                dpr = -0.5 * ((prop[:, j] - mean[:, j]) ** 2 - (phi[:, j] - mean[:, j]) ** 2) / np.maximum(omega[j] ** 2, 1e-12)
                a = accept(prop, lp, dpr)
                phi[a, j], cur[a] = prop[a, j], lp[a]
                rates[j] += a.mean() / 2
        # kernel 3: random-walk toàn phần x2
        r3 = 0.0
        for _ in range(2):
            prop = phi + c3 * omega * rng.standard_normal((M, P))
            lp = ll(prop)
            dpr = -0.5 * (((prop - mean) ** 2 - (phi - mean) ** 2) / np.maximum(omega ** 2, 1e-12)).sum(1)
            a = accept(prop, lp, dpr)
            phi[a], cur[a] = prop[a], lp[a]
            r3 += a.mean() / 2
        if phase1:                                # thích nghi bước về tỷ lệ chấp nhận mục tiêu
            c2 = np.clip(c2 * np.exp(0.3 * (rates - 0.4)), 0.05, 3.0)
            c3 = float(np.clip(c3 * np.exp(0.3 * (r3 - 0.3)), 0.02, 2.0))

        # ── SA: cập nhật thống kê đủ ──
        for j in range(P):
            S_xphi[j] = S_xphi[j] + gamma * (XM[j].T @ phi[:, j] / C - S_xphi[j])
        S_phi2 = S_phi2 + gamma * ((phi ** 2).sum(0) / C - S_phi2)
        f = predict(model_name, tM, dM, phi, data=data, sub_doses=sub_doses_M, sub_cov=sub_cov_M,
                    beta=beta, cov_model=cov_model, cov_mode=cov_mode, tinf=tinf)
        r2 = ((yM - f) ** 2 * mkM).reshape(C, N, T).mean(0)
        f2 = ((f ** 2) * mkM).reshape(C, N, T).mean(0)
        Sr2, Sf2 = Sr2 + gamma * (r2 - Sr2), Sf2 + gamma * (f2 - Sf2)

        # ── M-step ──
        for j in range(P):
            beta[j] = np.linalg.solve(XtX[j], S_xphi[j])                 # hồi quy bình phương tối thiểu
            w2 = max((S_phi2[j] - beta[j] @ S_xphi[j]) / N, 1e-6)        # phương sai phần dư quanh X*beta
            omega[j] = np.sqrt(max(w2, 0.95 * omega[j] ** 2) if phase1 else w2)
        new_err = update_err(Sr2, Sf2, mask, err, error_model)
        err = np.maximum(new_err, np.sqrt(0.95) * err) if phase1 else new_err
        mean = np.column_stack([XM[j] @ beta[j] for j in range(P)])

        if not phase1:                                # tích luỹ phân phối điều kiện cho EBE/IS
            ph = phi.reshape(C, N, P)
            cm_sum += ph.sum(0)
            cm2_sum += (ph ** 2).sum(0)
            n_cm += C
        hist["theta"].append(np.exp([b[0] for b in beta]))
        hist["beta"].append([b.copy() for b in beta])
        hist["omega"].append(omega.copy())
        hist["err"].append(err.copy())
        if progress is not None:
            progress(k / K)

    cm = cm_sum / n_cm
    csd = np.sqrt(np.maximum(cm2_sum / n_cm - cm ** 2, 1e-6))
    return dict(model_name=model_name, names=names, cov_model=cov_model, error_model=error_model,
                data=data, X=X, beta=beta, omega=omega.copy(), err=err.copy(), history=hist,
                cond_mean=cm, cond_sd=csd, n_iter1=n_iter1, n_iter2=n_iter2)


# ───────────────────────── Hậu xử lý ─────────────────────────
def _subj(res, i):
    d = res["data"]
    sub_d = [d["sub_doses"][d["ids"][i]]] if "sub_doses" in d else None
    sub_c = [d["sub_cov_events"][d["ids"][i]]] if "sub_cov_events" in d else None
    return d["t"][i:i + 1], d["y"][i:i + 1], d["mask"][i:i + 1], d["dose"][i:i + 1], sub_d, sub_c


def compute_ebe(res):
    """EBE = mode của p(eta | y) với tham số quần thể đã ước lượng."""
    mean, om, err, P = pop_mean_phi(res), res["omega"], res["err"], len(res["names"])
    out = np.zeros_like(mean)
    d = res["data"]
    for i in range(d["N"]):
        sid = d["ids"][i] if "ids" in d else i
        t, y, mk, ds, sub_d, sub_c = _subj(res, i)
        obj = lambda eta: (-loglik_rows((mean[i] + eta)[None], res["model_name"], t, y, mk, ds, err,
                                        error_model=res["error_model"], data=d,
                                        sub_doses=sub_d, sub_cov=sub_c,
                                        beta=res["beta"], cov_model=res["cov_model"],
                                        cov_mode=d.get("cov_mode", "time-varying"),
                                        tinf=d.get("tinf", 1.0), subject_ids=[sid])[0]
                           + 0.5 * np.sum((eta / np.maximum(om, 1e-8)) ** 2))
        fits = [minimize(obj, x0, method="BFGS") for x0 in (res["cond_mean"][i] - mean[i], np.zeros(P))]
        out[i] = mean[i] + min(fits, key=lambda r: r.fun).x
    return out


def loglik_is(res, n_is=1500, seed=7):
    """log-likelihood biên bằng importance sampling (proposal t5 quanh phân phối điều kiện). Trả (LL, MC-sd -2LL)."""
    rng = np.random.default_rng(seed)
    mean, om, err = pop_mean_phi(res), res["omega"], res["err"]
    P, N = len(res["names"]), res["data"]["N"]
    sd = np.maximum(res["cond_sd"], 0.02)
    li = np.zeros(N)
    var_li = np.zeros(N)
    d = res["data"]
    for i in range(N):
        sid = d["ids"][i] if "ids" in d else i
        sub_ids_nis = np.full(n_is, sid)
        z = student_t.rvs(5, size=(n_is, P), random_state=rng)
        phi = res["cond_mean"][i] + sd[i] * z
        logq = (student_t.logpdf(z, 5) - np.log(sd[i])).sum(1)
        logp = norm.logpdf(phi, mean[i], np.maximum(om, 1e-8)).sum(1)
        t, y, mk, ds, sub_d, sub_c = _subj(res, i)
        sub_d_nis = sub_d * n_is if sub_d is not None else None
        sub_c_nis = sub_c * n_is if sub_c is not None else None
        l = loglik_rows(phi, res["model_name"], np.tile(t, (n_is, 1)), np.tile(y, (n_is, 1)),
                        np.tile(mk, (n_is, 1)), np.tile(ds, (n_is, 1)), err,
                        error_model=res["error_model"], const=True, data=d,
                        sub_doses=sub_d_nis, sub_cov=sub_c_nis,
                        beta=res["beta"], cov_model=res["cov_model"],
                        cov_mode=d.get("cov_mode", "time-varying"),
                        tinf=d.get("tinf", 1.0), subject_ids=sub_ids_nis)
        log_w = l + logp - logq
        max_lw = np.max(log_w)
        w = np.exp(log_w - max_lw)
        mean_w = np.mean(w)
        li[i] = max_lw + np.log(mean_w)
        var_w = np.var(w, ddof=1) if n_is > 1 else 0.0
        var_li[i] = var_w / (n_is * np.maximum(mean_w**2, 1e-30))
    ll_tot = float(li.sum())
    mc_se_m2ll = float(2.0 * np.sqrt(np.sum(var_li)))
    return ll_tot, mc_se_m2ll


def information_criteria(res, ll, mc_se_m2ll=0.0):
    n_fix = sum(len(b) for b in res["beta"]) + len(res["names"])      # beta + omega (tham số có RE)
    n_err = int(sum(err_free(res["error_model"])))
    N, n = res["data"]["N"], res["data"]["nobs"]
    m2 = -2 * ll
    return dict(m2ll=m2, mc_se=mc_se_m2ll, aic=m2 + 2 * (n_fix + n_err), bic=m2 + (n_fix + n_err) * np.log(N),
                bicc=m2 + n_fix * np.log(N) + n_err * np.log(n))      # BICc theo định nghĩa Monolix


def compute_fim(res, phi_hat):
    """SE bằng FIM tuyến tính hoá (FOCE) quanh EBE. Trả dict nhãn -> (ước lượng, SE). SE=nan nếu Hessian không xác định dương."""
    d, names, P = res["data"], res["names"], len(res["names"])
    N, T, mask, y = d["N"], d["T"], d["mask"], d["y"]
    X, mn = res["X"], res["model_name"]
    sub_ids = d["ids"] if "ids" in d else None
    f0 = predict(mn, d["t"], d["dose"], phi_hat, data=d, beta=res["beta"], cov_model=res["cov_model"], subject_ids=sub_ids)
    J = np.zeros((N, T, P))
    for j in range(P):
        up, dn = phi_hat.copy(), phi_hat.copy()
        up[:, j] += 1e-4
        dn[:, j] -= 1e-4
        f_up = predict(mn, d["t"], d["dose"], up, data=d, beta=res["beta"], cov_model=res["cov_model"], subject_ids=sub_ids)
        f_dn = predict(mn, d["t"], d["dose"], dn, data=d, beta=res["beta"], cov_model=res["cov_model"], subject_ids=sub_ids)
        J[:, :, j] = (f_up - f_dn) / 2e-4
    J *= mask[:, :, None]
    fa, fb = err_free(res["error_model"])
    sizes = [len(b) for b in res["beta"]]
    eye = np.eye(T)[None]

    def unpack(th):
        pos, beta = 0, []
        for s in sizes:
            beta.append(th[pos:pos + s])
            pos += s
        om = th[pos:pos + P]
        pos += P
        a = th[pos] if fa else 0.0
        b = th[pos + int(fa)] if fb else 0.0
        return beta, om, a, b

    def m2ll(th):
        beta, om, a, b = unpack(th)
        mean = np.column_stack([X[j] @ beta[j] for j in range(P)])
        E = f0 + np.einsum("ntp,np->nt", J, mean - phi_hat)
        sd_sq = resid_sd(f0, [a, b], error_model=res["error_model"]) ** 2
        R = np.where(mask, np.maximum(sd_sq, 1e-10), 1.0)
        V = np.einsum("ntp,p,nsp->nts", J, om ** 2, J) + R[:, :, None] * eye
        r = np.where(mask, y - E, 0.0)
        _, logdet = np.linalg.slogdet(V)
        quad = np.einsum("nt,nt->n", r, np.linalg.solve(V, r[..., None])[..., 0])
        return float(np.sum(logdet + quad))

    th0 = np.concatenate([np.concatenate(res["beta"]), res["omega"], res["err"][[fa, fb]]])
    n = len(th0)
    h = 1e-3 * np.maximum(np.abs(th0), 0.1)
    f00, H = m2ll(th0), np.zeros((n, n))
    for i in range(n):
        ei = np.eye(n)[i] * h[i]
        H[i, i] = (m2ll(th0 + ei) - 2 * f00 + m2ll(th0 - ei)) / h[i] ** 2
        for j in range(i):
            ej = np.eye(n)[j] * h[j]
            H[i, j] = H[j, i] = (m2ll(th0 + ei + ej) - m2ll(th0 + ei - ej)
                                 - m2ll(th0 - ei + ej) + m2ll(th0 - ei - ej)) / (4 * h[i] * h[j])
    ok = np.all(np.linalg.eigvalsh(H) > 0)
    se = np.sqrt(np.diag(2 * np.linalg.inv(H))) if ok else np.full(n, np.nan)   # cov = 2 H^-1 cho -2LL
    labels = [(f"{names[j]}" if k == 0 else f"beta_{names[j]}_{res['cov_model'].get(names[j], [])[k - 1]}")
              for j in range(P) for k in range(sizes[j])]
    labels += [f"omega_{p}" for p in names] + (["a"] if fa else []) + (["b"] if fb else [])
    return {lab: (th0[i], se[i]) for i, lab in enumerate(labels)}


def compute_diagnostics(res, phi_hat, K=500, seed=11):
    """PRED, IPRED, IWRES, PWRES (điểm), NPDE (có giải tương quan bằng Cholesky) và shrinkage."""
    d, mn, err, om = res["data"], res["model_name"], res["err"], res["omega"]
    mask, N = d["mask"], d["N"]
    mean = pop_mean_phi(res)
    sub_ids = d["ids"] if "ids" in d else None
    PRED = predict(mn, d["t"], d["dose"], mean, data=d, beta=res["beta"], cov_model=res["cov_model"], subject_ids=sub_ids)
    IPRED = predict(mn, d["t"], d["dose"], phi_hat, data=d, beta=res["beta"], cov_model=res["cov_model"], subject_ids=sub_ids)
    IWRES = (d["y"] - IPRED) / resid_sd(IPRED, err, error_model=res["error_model"])
    rng = np.random.default_rng(seed)
    PWRES, NPDE = np.zeros((N, d["T"])), np.zeros((N, d["T"]))
    for i in range(N):
        mk = mask[i]
        sid = d["ids"][i] if "ids" in d else i
        sub_ids_k = np.full(K, sid)
        t, _, _, ds, sub_d, sub_c = _subj(res, i)
        sub_d_k = sub_d * K if sub_d is not None else None
        sub_c_k = sub_c * K if sub_c is not None else None
        phi_sim = mean[i] + om * rng.standard_normal((K, len(om)))
        f = predict(mn, np.tile(t, (K, 1)), np.tile(ds, (K, 1)), phi_sim, data=d,
                    sub_doses=sub_d_k, sub_cov=sub_c_k, beta=res["beta"], cov_model=res["cov_model"],
                    subject_ids=sub_ids_k)
        ys = (f + resid_sd(f, err, error_model=res["error_model"]) * rng.standard_normal(f.shape))[:, mk]
        yi, mu, n = d["y"][i, mk], ys.mean(0), mk.sum()
        PWRES[i, mk] = (yi - mu) / np.maximum(ys.std(0), 1e-12)
        V = np.cov(ys, rowvar=False).reshape(n, n) + 1e-10 * np.eye(n)
        L = np.linalg.cholesky(V)
        zi, zs = np.linalg.solve(L, yi - mu), np.linalg.solve(L, (ys - mu).T).T
        NPDE[i, mk] = norm.ppf(np.clip((zs < zi).mean(0), 0.5 / K, 1 - 0.5 / K))
    eta = phi_hat - mean
    long = pd.DataFrame({"ID": np.repeat(d["ids"][:, None], d["T"], 1)[mask], "TIME": d["t"][mask],
                         "DV": d["y"][mask], "PRED": PRED[mask], "IPRED": IPRED[mask],
                         "IWRES": IWRES[mask], "PWRES": PWRES[mask], "NPDE": NPDE[mask]})
    return dict(table=long, eta=eta, eta_shrink=1 - eta.std(0) / om, eps_shrink=1 - IWRES[mask].std())


def run_vpc(res, n_rep=200, n_bins=10, pcorr=False, pi=(5, 50, 95), ci=90, seed=5):
    """VPC chuẩn: mô phỏng n_rep bản sao dataset theo đúng thiết kế (thời điểm, liều, covariate),
    CI của từng percentile theo bin; tuỳ chọn prediction-corrected."""
    d, mn, err, om = res["data"], res["model_name"], res["err"], res["omega"]
    N, T, mask = d["N"], d["T"], d["mask"]
    rng = np.random.default_rng(seed)
    mean = pop_mean_phi(res)
    sub_ids_nrep = np.tile(d["ids"], n_rep) if "ids" in d else None
    phi = np.tile(mean, (n_rep, 1)) + om * rng.standard_normal((n_rep * N, len(om)))
    f = predict(mn, np.tile(d["t"], (n_rep, 1)), np.tile(d["dose"], (n_rep, 1)), phi,
                data=d, beta=res["beta"], cov_model=res["cov_model"], subject_ids=sub_ids_nrep)
    sim = (f + resid_sd(f, err, error_model=res["error_model"]) * rng.standard_normal(f.shape)).reshape(n_rep, N, T)
    obs = d["y"].copy()
    sub_ids = d["ids"] if "ids" in d else None
    PRED = predict(mn, d["t"], d["dose"], mean, data=d, beta=res["beta"], cov_model=res["cov_model"], subject_ids=sub_ids)
    tt = d["t"][mask]
    u = np.unique(tt)
    edges = (u[:-1] + u[1:]) / 2 if len(u) <= n_bins else \
        np.unique(np.quantile(tt, np.linspace(0, 1, n_bins + 1)[1:-1]))
    bidx = np.digitize(d["t"], edges)
    if pcorr:
        fac = np.ones_like(PRED)
        for b in np.unique(bidx[mask]):
            s = mask & (bidx == b)
            fac[s] = np.median(PRED[s]) / np.maximum(PRED[s], 1e-12)
        obs, sim = obs * fac, sim * fac[None]
    rows = []
    for b in np.unique(bidx[mask]):
        s = mask & (bidx == b)
        qs = np.percentile(sim[:, s], pi, axis=1)                         # [len(pi), n_rep]
        row = {"TIME": np.median(d["t"][s])}
        for k, q in enumerate(pi):
            row[f"obs{q}"] = np.percentile(obs[s], q)
            row[f"med{q}"] = np.percentile(qs[k], 50)
            row[f"lo{q}"] = np.percentile(qs[k], (100 - ci) / 2)
            row[f"hi{q}"] = np.percentile(qs[k], 100 - (100 - ci) / 2)
        rows.append(row)
    return pd.DataFrame(rows)


def convergence_report(res, tol=0.05):
    """So trung bình nửa đầu và nửa sau của pha 2 cho từng tham số; trôi > tol thì cảnh báo."""
    h, k1 = res["history"], res["n_iter1"]
    names, out = res["names"], []
    traces = {f"θ_{p}": np.array([x[j] for x in h["theta"]]) for j, p in enumerate(names)}
    traces.update({f"ω_{p}": np.array([x[j] for x in h["omega"]]) for j, p in enumerate(names)})
    for i, lab in enumerate(["a", "b"]):
        if err_free(res["error_model"])[i]:
            traces[lab] = np.array([x[i] for x in h["err"]])
    for lab, tr in traces.items():
        p2 = tr[k1:]
        a, b = p2[:len(p2) // 2].mean(), p2[len(p2) // 2:].mean()
        out.append((lab, abs(b - a) / max(abs(a), 1e-9) < tol, abs(b - a) / max(abs(a), 1e-9)))
    return out


def fit_model(df, model_name, init, cov_model, error_model="combined", n_iter1=250, n_iter2=150,
              n_chains=1, seed=1, progress=None, n_is=1500, cov_mode="time-varying"):
    """Chạy toàn bộ: SAEM -> EBE -> -2LL -> SE -> chẩn đoán."""
    data = prepare_data(df, cov_model=cov_model, cov_mode=cov_mode)
    res = run_saem(data, model_name, init, cov_model, error_model, n_iter1, n_iter2, n_chains, seed, progress)
    res["phi_hat"] = compute_ebe(res)
    res["ll"], res["mc_se"] = loglik_is(res, n_is)
    res["ic"] = information_criteria(res, res["ll"], res["mc_se"])
    res["fim"] = compute_fim(res, res["phi_hat"])
    res["diag"] = compute_diagnostics(res, res["phi_hat"])
    return res
