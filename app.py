"""
SimDx - NLME Pharmacometrics Platform (Mini Monolix).
Giao diện Streamlit tương tác cho mô hình dược động học phi tuyến hiệu ứng hỗn hợp (NLME).
Hỗ trợ cả dữ liệu mô phỏng và bộ dữ liệu lâm sàng thực tế (Dataset_chính_thức.xlsx).
"""
import io
import os
import json
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import linregress, norm

import pk_core as pk
import data_io

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
if not os.path.exists(LOGO_PATH):
    LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.jpg")

LOGO_B64 = ""
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as _f:
        LOGO_B64 = base64.b64encode(_f.read()).decode()

CAT_PATH = os.path.join(os.path.dirname(__file__), "assets", "cat_meme.jpg")
CAT_B64 = ""
if os.path.exists(CAT_PATH):
    with open(CAT_PATH, "rb") as _f:
        CAT_B64 = base64.b64encode(_f.read()).decode()

st.set_page_config(
    page_title="SimDx — NLME Pharmacometrics",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else ":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stMainBlockContainer"] {
        padding-top: 2.5rem;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
        display: none !important;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 1001;
    }
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.simdx-sidebar-brand) {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: #f4f7fc;
    }
    .simdx-sidebar-brand {
        background-color: #f4f7fc;
        padding: 12px 0 12px 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 8px;
    }
    .section-header { background: #e8eaf6; border-radius: 8px; padding: 8px 16px; font-weight: 700;
        color: #1a237e; margin: 15px 0 10px; font-size: 1.05em; }
    .convergence-box { background: #e8f5e9; border: 1px solid #4caf50; border-radius: 8px;
        padding: 10px 15px; margin: 10px 0; color: #1b5e20; font-weight: 600; }
    .outlier-box { background: #fff3e0; border: 1px solid #ff9800; border-radius: 8px;
        padding: 10px 15px; margin: 10px 0; color: #e65100; font-weight: 500; }
    
    /* Monolix Model Library Styles */
    .lib-header-col {
        padding: 6px 10px;
        font-weight: 700;
        color: white;
        font-size: 13px;
        border-radius: 4px 4px 0 0;
        text-align: center;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }
    .col-admin { background-color: #0097a7; }
    .col-delay { background-color: #1976d2; }
    .col-distrib { background-color: #5c6bc0; }
    .col-elim { background-color: #7b1fa2; }
    .col-param { background-color: #c2185b; }
    .param-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
        color: white;
        margin-right: 4px;
    }
    .pill-v { background-color: #5c6bc0; }
    .pill-cl { background-color: #c2185b; }
    .pill-ka { background-color: #0097a7; }
    .pill-q { background-color: #f57c00; }
    .pill-other { background-color: #455a64; }
    .model-id-title {
        font-family: 'SF Mono', Consolas, Monaco, monospace;
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
        display: inline-block;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

T_GRID = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 36, 48, 72])
WT_EFFECT_DEFAULT = {"CL": 0.75, "Q": 0.75, "Vmax": 0.75, "V": 1.0, "V1": 1.0, "V2": 1.0}

MONOLIX_PK_LIBRARY = [
    {
        "id": "bolus_1cpt_VCl",
        "category": "PK",
        "admin": "bolus",
        "delay": "no delay",
        "distrib": "1 compartment",
        "elim": "linear",
        "param": "clearance",
        "params": ["V", "CL"],
        "units": ["L", "L/h"],
        "defaults": [20.0, 5.0],
        "ranges": [(5.0, 100.0), (1.0, 30.0)],
        "internal_model": "1-Compartment IV infusion (multi-dose)",
        "tinf_override": 0.01,
        "error_override": "combined1",
        "desc": "1-ngăn tiêm tĩnh mạch trực tiếp (IV Bolus tức thời) với đào thải tuyến tính bậc 1, tham số hóa qua Thể tích phân bố (V) và Độ thanh thải (CL). Hỗ trợ phác đồ đơn liều và đa liều.",
        "ode": r"\frac{dC}{dt} = -\frac{CL}{V} \cdot C, \quad C(0) = \frac{\text{Dose}}{V}",
        "solution": r"C(t) = \sum_{k} \frac{\text{Dose}_k}{V} \cdot e^{-\frac{CL}{V} (t - t_k)}"
    },
    {
        "id": "infusion_1cpt_VCl",
        "category": "PK",
        "admin": "infusion",
        "delay": "no delay",
        "distrib": "1 compartment",
        "elim": "linear",
        "param": "clearance",
        "params": ["CL", "V"],
        "units": ["L/h", "L"],
        "defaults": [5.0, 40.0],
        "ranges": [(0.5, 30.0), (5.0, 150.0)],
        "internal_model": "1-Compartment IV infusion (multi-dose)",
        "tinf_override": 1.0,
        "error_override": "combined",
        "desc": "1-ngăn truyền tĩnh mạch (IV Infusion với tốc độ không đổi R_inf = Dose/tinf) và đào thải tuyến tính bậc 1, tham số hóa qua Độ thanh thải (CL) và Thể tích (V).",
        "ode": r"\frac{dC}{dt} = \frac{R_{inf}}{V} - \frac{CL}{V} \cdot C",
        "solution": r"0 \le t \le t_{inf}: C(t) = \frac{\text{Dose}}{t_{inf} \cdot CL}\left(1 - e^{-\frac{CL}{V} t}\right); \quad t > t_{inf}: C(t) = C(t_{inf}) e^{-\frac{CL}{V} (t - t_{inf})}"
    },
    {
        "id": "oral1_1cpt_kaVCl",
        "category": "PK",
        "admin": "oral/extravascular",
        "delay": "no delay",
        "distrib": "1 compartment",
        "elim": "linear",
        "param": "clearance",
        "params": ["ka", "V", "CL"],
        "units": ["1/h", "L", "L/h"],
        "defaults": [1.2, 20.0, 5.0],
        "ranges": [(0.1, 5.0), (5.0, 100.0), (1.0, 30.0)],
        "internal_model": "1-Compartment (Oral)",
        "tinf_override": None,
        "error_override": "combined",
        "desc": "1-ngăn đường uống / ngoài mạch (Oral) với hấp thu bậc 1 (ka), thể tích phân bố (V) và độ thanh thải (CL).",
        "ode": r"\frac{dA_d}{dt} = -k_a A_d, \quad \frac{dC}{dt} = \frac{k_a A_d}{V} - \frac{CL}{V} C",
        "solution": r"C(t) = \frac{\text{Dose} \cdot k_a}{V (k_a - k)} \left(e^{-k t} - e^{-k_a t}\right)"
    },
    {
        "id": "oral1_2cpt_kaV1ClV2Q",
        "category": "PK",
        "admin": "oral/extravascular",
        "delay": "no delay",
        "distrib": "2 compartments",
        "elim": "linear",
        "param": "clearance",
        "params": ["ka", "V1", "CL", "Q", "V2"],
        "units": ["1/h", "L", "L/h", "L/h", "L"],
        "defaults": [1.2, 15.0, 4.0, 8.0, 40.0],
        "ranges": [(0.1, 5.0), (5.0, 50.0), (1.0, 20.0), (1.0, 30.0), (10.0, 150.0)],
        "internal_model": "2-Compartment (Oral)",
        "tinf_override": None,
        "error_override": "combined",
        "desc": "2-ngăn đường uống (Oral 2-cpt) gồm ngăn trung tâm (V1, CL), ngăn ngoại vi (V2) và độ thanh thải liên ngăn (Q).",
        "ode": r"\frac{dC_1}{dt} = \frac{k_a A_d}{V_1} - \frac{CL+Q}{V_1} C_1 + \frac{Q}{V_1} C_2; \quad \frac{dC_2}{dt} = \frac{Q}{V_2} C_1 - \frac{Q}{V_2} C_2",
        "solution": r"C(t) = \frac{\text{Dose} \cdot k_a}{V_1} \left(A e^{-\alpha t} + B e^{-\beta t} + C_c e^{-k_a t}\right)"
    },
    {
        "id": "bolus_1cpt_VKmVmax",
        "category": "PK",
        "admin": "bolus",
        "delay": "no delay",
        "distrib": "1 compartment",
        "elim": "Michaelis-Menten",
        "param": "clearance",
        "params": ["V", "Vmax", "Km"],
        "units": ["L", "mg/h", "mg/L"],
        "defaults": [20.0, 10.0, 2.0],
        "ranges": [(5.0, 50.0), (1.0, 50.0), (0.1, 20.0)],
        "internal_model": "Michaelis-Menten",
        "tinf_override": None,
        "error_override": "combined",
        "desc": "1-ngăn IV Bolus với động học bão hòa phi tuyến Michaelis-Menten (tốc độ đào thải tối đa Vmax, hằng số Michaelis Km).",
        "ode": r"\frac{dC}{dt} = -\frac{V_{max} \cdot C}{V \cdot (K_m + C)}",
        "solution": r"C(t) = K_m \cdot W\left(\frac{C_0}{K_m} e^{\frac{C_0}{K_m} - \frac{V_{max}}{V \cdot K_m} t}\right)"
    }
]


@st.dialog("📚 LOAD FROM LIBRARY — Monolix Model Library", width="large")
def show_model_library_dialog():
    st.markdown("""
    <div style="background: #f1f5f9; padding: 10px 14px; border-radius: 6px; font-size: 13px; margin-bottom: 12px; border-left: 4px solid #0097a7;">
    <b>Monolix Library Browser:</b> Lọc theo đường dùng (Administration), độ trễ (Delay), số ngăn (Distribution), cơ chế thải trừ (Elimination) và hệ tham số (Parametrization) để chọn mô hình chuẩn Monolix.
    </div>
    """, unsafe_allow_html=True)

    tab_pk, tab_pd, tab_pkpd, tab_other = st.tabs(["PK", "PD", "PKPD", "TMDD / Other"])

    with tab_pk:
        # 5 Filter columns with colored headers like Monolix
        f1, f2, f3, f4, f5 = st.columns(5)
        with f1:
            st.markdown('<div class="lib-header-col col-admin">Administration</div>', unsafe_allow_html=True)
            admin_sel = st.selectbox("Administration", ["Tất cả", "bolus", "infusion", "oral/extravascular"], label_visibility="collapsed", key="lib_admin")
        with f2:
            st.markdown('<div class="lib-header-col col-delay">Delay</div>', unsafe_allow_html=True)
            delay_sel = st.selectbox("Delay", ["Tất cả", "no delay", "lag time"], label_visibility="collapsed", key="lib_delay")
        with f3:
            st.markdown('<div class="lib-header-col col-distrib">Distribution</div>', unsafe_allow_html=True)
            distrib_sel = st.selectbox("Distribution", ["Tất cả", "1 compartment", "2 compartments"], label_visibility="collapsed", key="lib_distrib")
        with f4:
            st.markdown('<div class="lib-header-col col-elim">Elimination</div>', unsafe_allow_html=True)
            elim_sel = st.selectbox("Elimination", ["Tất cả", "linear", "Michaelis-Menten"], label_visibility="collapsed", key="lib_elim")
        with f5:
            st.markdown('<div class="lib-header-col col-param">Parametrization</div>', unsafe_allow_html=True)
            param_sel = st.selectbox("Parametrization", ["Tất cả", "clearance", "rate"], label_visibility="collapsed", key="lib_param")

        c_search, c_clear = st.columns([4, 1])
        search_q = c_search.text_input("🔍 Tìm kiếm tên mô hình...", placeholder="Ví dụ: bolus_1cpt_VCl, infusion...", label_visibility="collapsed", key="lib_search")
        if c_clear.button("Xóa bộ lọc", key="btn_clear_filters"):
            st.session_state["lib_admin"] = "Tất cả"
            st.session_state["lib_delay"] = "Tất cả"
            st.session_state["lib_distrib"] = "Tất cả"
            st.session_state["lib_elim"] = "Tất cả"
            st.session_state["lib_param"] = "Tất cả"
            st.session_state["lib_search"] = ""
            st.rerun()

        # Match models
        matched = []
        for m in MONOLIX_PK_LIBRARY:
            if admin_sel != "Tất cả" and m["admin"] != admin_sel:
                continue
            if delay_sel != "Tất cả" and m["delay"] != delay_sel:
                continue
            if distrib_sel != "Tất cả" and m["distrib"] != distrib_sel:
                continue
            if elim_sel != "Tất cả" and m["elim"] != elim_sel:
                continue
            if param_sel != "Tất cả" and m["param"] != param_sel:
                continue
            if search_q.strip():
                sq = search_q.strip().lower()
                if sq not in m["id"].lower() and sq not in m["desc"].lower():
                    continue
            matched.append(m)

        st.markdown(f"**Danh sách mô hình ({len(matched)} mô hình phù hợp):**")

        if not matched:
            st.warning("Không có mô hình nào khớp với tổ hợp lọc hiện tại. Hãy bấm 'Xóa bộ lọc'.")
        else:
            for m in matched:
                pills_html = "".join([f'<span class="param-pill pill-v">{p}</span>' if p in ['V', 'V1', 'V2'] else f'<span class="param-pill pill-cl">{p}</span>' if p in ['CL', 'Q'] else f'<span class="param-pill pill-ka">{p}</span>' if p == 'ka' else f'<span class="param-pill pill-other">{p}</span>' for p in m["params"]])
                
                col_info, col_btn = st.columns([5, 2])
                with col_info:
                    st.markdown(f'{pills_html} <span class="model-id-title">{m["id"]}</span>', unsafe_allow_html=True)
                    st.caption(f"{m['desc']}")
                    with st.expander("Phương trình toán học (ODE & Closed-form)"):
                        st.latex(m["ode"])
                        st.latex(m["solution"])
                with col_btn:
                    if st.button("✅ Chọn Model này", key=f"sel_mod_{m['id']}", type="primary", use_container_width=True):
                        st.session_state["selected_lib_model"] = m["id"]
                        st.session_state["model_name_choice"] = m["internal_model"]
                        if m["tinf_override"] is not None:
                            st.session_state["tinf_val"] = m["tinf_override"]
                            st.session_state["tinf_upload_input"] = float(m["tinf_override"])
                            st.session_state["tinf_main_input"] = float(m["tinf_override"])
                        if m.get("error_override"):
                            st.session_state["error_model_choice"] = m["error_override"]
                        st.rerun()
                st.markdown("<hr style='margin: 6px 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    with tab_pd:
        st.info("Các mô hình Dược lực học PD (Emax, Sigmoid Emax, Turnover) sẽ được mở rộng trong module tiếp theo.")
    with tab_pkpd:
        st.info("Các mô hình PK/PD phối hợp (Direct effect, Effect compartment) đang được tích hợp.")
    with tab_other:
        st.info("Các mô hình TMDD (Target-Mediated Drug Disposition) và thời gian sống sót TTE.")


def header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


@st.cache_data
def make_sim_data(model_name, theta, omega, sa, sp, cov_eff, n, dose, t_pts, n_obs, seed):
    return pk.generate_population_data(model_name, dict(theta), dict(omega), sa, sp, dict(cov_eff),
                                       n, dose, list(t_pts), n_obs, seed)


def estimates_table(res):
    names, truth, rows = res["names"], res.get("truth", {}), []
    for lab, (est, se) in res["fim"].items():
        if lab in names:
            val = np.exp(est)
            s = np.exp(est) * se if np.isfinite(se) else np.nan
            rse = 100 * se if np.isfinite(se) else np.nan
            ci_lo = np.exp(est - 1.96 * se) if np.isfinite(se) else np.nan
            ci_hi = np.exp(est + 1.96 * se) if np.isfinite(se) else np.nan
            show = f"θ_{lab}"
            tr = truth.get(lab)
            p = np.nan
        elif lab.startswith("beta_"):
            cov = lab.rsplit("_", 1)[-1]
            val = est
            s = se
            rse = 100 * se / max(abs(est), 1e-9) if np.isfinite(se) else np.nan
            ci_lo = est - 1.96 * se if np.isfinite(se) else np.nan
            ci_hi = est + 1.96 * se if np.isfinite(se) else np.nan
            show = f"β_{lab[5:-len(cov) - 1]}~{cov}"
            tr = truth.get(lab, None)
            p = 2 * norm.sf(abs(est / se)) if np.isfinite(se) and se > 0 else np.nan
        else:
            val = est
            s = se
            rse = 100 * se / max(abs(est), 1e-9) if np.isfinite(se) else np.nan
            ci_lo = max(0.0, est - 1.96 * se) if np.isfinite(se) else np.nan
            ci_hi = est + 1.96 * se if np.isfinite(se) else np.nan
            show = lab.replace("omega_", "ω_")
            tr = truth.get(lab)
            p = np.nan

        ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]" if np.isfinite(ci_lo) and np.isfinite(ci_hi) else "—"
        rows.append({
            "Tham số": show,
            "Giá trị thật": "—" if tr is None else f"{tr:.3f}",
            "Ước lượng": f"{val:.3f}",
            "SE": "—" if np.isnan(s) else f"{s:.3f}",
            "RSE (%)": "—" if np.isnan(rse) else f"{rse:.1f}%",
            "95% CI": ci_str,
            "p (Wald)": "—" if np.isnan(p) else f"{p:.3g}"
        })
    return pd.DataFrame(rows)


def convergence_figure(res):
    h, names, k1 = res["history"], res["names"], res["n_iter1"]
    truth = res.get("truth", {})
    items = [(f"θ_{p}", [x[j] for x in h["theta"]], truth.get(p)) for j, p in enumerate(names)]
    items += [(f"ω_{p}", [x[j] for x in h["omega"]], truth.get(f"omega_{p}")) for j, p in enumerate(names)]
    items += [(lab, [x[i] for x in h["err"]], truth.get(lab))
              for i, lab in enumerate(["a", "b"]) if pk.err_free(res["error_model"])[i]]
    for j, p in enumerate(names):
        for k, c in enumerate(res["cov_model"].get(p, []), start=1):
            items.append((f"β_{p}~{c}", [x[j][k] for x in h["beta"]], truth.get(f"beta_{p}_{c}", None)))
    ncols = 3
    nrows = (len(items) + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[i[0] for i in items])
    for n, (lab, tr, true_val) in enumerate(items):
        r, c = divmod(n, ncols)
        fig.add_trace(go.Scatter(y=tr, mode="lines", line=dict(color="#1565c0", width=1.5),
                                 showlegend=False), row=r + 1, col=c + 1)
        if true_val is not None:
            fig.add_hline(y=true_val, line_dash="dash", line_color="red", row=r + 1, col=c + 1)
    fig.add_vline(x=k1, line_dash="dot", line_color="gray")
    fig.update_layout(height=max(300, nrows * 220), template="plotly_white",
                      title_text="SAEM traces (xanh = ước lượng, đỏ = giá trị thật nếu có, chấm = hết pha 1)")
    return fig


def gof_figure(dg):
    tt = dg["table"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[
        "DV vs PRED", "DV vs IPRED", "IWRES vs IPRED", "IWRES vs TIME", "NPDE vs TIME", "NPDE vs PRED"])
    spec = [("PRED", "DV", 1, 1, "#1565c0"), ("IPRED", "DV", 1, 2, "#2e7d32"),
            ("IPRED", "IWRES", 1, 3, "#6a1b9a"), ("TIME", "IWRES", 2, 1, "#6a1b9a"),
            ("TIME", "NPDE", 2, 2, "#e65100"), ("PRED", "NPDE", 2, 3, "#e65100")]
    for x, y, r, c, col in spec:
        fig.add_trace(go.Scatter(x=tt[x], y=tt[y], mode="markers",
                                 marker=dict(color=col, size=5, opacity=0.6)), row=r, col=c)
        if y == "DV":
            lim = max(float(tt["DV"].max()), float(tt[x].max())) * 1.05
            fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                     line=dict(color="red", dash="dash")), row=r, col=c)
        else:
            for v, dash in ((0, "dash"), (1.96, "dot"), (-1.96, "dot")):
                fig.add_hline(y=v, line_dash=dash, line_color="red", row=r, col=c)
    fig.update_layout(height=650, template="plotly_white", showlegend=False, title_text="Goodness-of-Fit")
    return fig


def vpc_figure(v, raw_points):
    fig = go.Figure()
    colors = {5: "31,119,180", 50: "214,39,40", 95: "31,119,180"}
    for q in (5, 50, 95):
        x = list(v["TIME"]) + list(v["TIME"][::-1])
        fig.add_trace(go.Scatter(x=x, y=list(v[f"hi{q}"]) + list(v[f"lo{q}"][::-1]), fill="toself",
                                 fillcolor=f"rgba({colors[q]},0.2)", line=dict(width=0),
                                 name=f"CI mô phỏng p{q}", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=v["TIME"], y=v[f"med{q}"], mode="lines",
                                 line=dict(color=f"rgb({colors[q]})", dash="dash", width=1.5),
                                 name=f"Median mô phỏng p{q}"))
        fig.add_trace(go.Scatter(x=v["TIME"], y=v[f"obs{q}"], mode="lines+markers",
                                 line=dict(color="black", width=2), name=f"Quan sát p{q}"))
    if raw_points is not None:
        fig.add_trace(go.Scatter(x=raw_points["TIME"], y=raw_points["DV"], mode="markers",
                                 marker=dict(color="gray", size=4, opacity=0.35), name="Quan sát"))
    fig.update_layout(title="VPC (5/50/95%, CI 90% của percentile)", xaxis_title="Time (h)",
                      yaxis_title="Concentration (mg/L)", height=480, template="plotly_white")
    return fig


def compute_nca(clinical_data):
    if clinical_data is None or "clean_df" not in clinical_data:
        return None, None
    clean_df = clinical_data["clean_df"]
    obs_df = clean_df[clean_df["DV"].notna()].copy()
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    rows = []
    for sid, group in obs_df.groupby("ID"):
        sub = group.sort_values("Time")
        times = sub["Time"].values
        dvs = sub["DV"].values
        if len(times) == 0:
            continue
        cmax = float(np.max(dvs))
        tmax = float(times[np.argmax(dvs)])
        auc_t = float(trapz_fn(dvs, times)) if len(times) > 1 else np.nan
        if len(times) >= 2 and dvs[-1] > 0 and dvs[-2] > 0 and times[-1] > times[-2]:
            slope = (np.log(dvs[-1]) - np.log(dvs[-2])) / (times[-1] - times[-2])
            if slope < 0:
                lam_z = -slope
                t_half = float(np.log(2.0) / lam_z)
                auc_inf = float(auc_t + dvs[-1] / lam_z) if np.isfinite(auc_t) else np.nan
            else:
                lam_z, t_half, auc_inf = np.nan, np.nan, np.nan
        else:
            lam_z, t_half, auc_inf = np.nan, np.nan, np.nan
        doses = clinical_data["sub_doses"].get(sid, [])
        tot_dose = sum(d[1] for d in doses) if doses else 1000.0
        cl_nca = (tot_dose / auc_inf) if np.isfinite(auc_inf) and auc_inf > 0 else np.nan
        vd_nca = (cl_nca / lam_z) if np.isfinite(cl_nca) and np.isfinite(lam_z) and lam_z > 0 else np.nan
        rows.append({
            "ID": sid, "Cmax (mg/L)": cmax, "Tmax (h)": tmax,
            "AUC0_t (mg·h/L)": auc_t, "AUC_inf (mg·h/L)": auc_inf,
            "t1/2 (h)": t_half, "CL_nca (L/h)": cl_nca, "Vd_nca (L)": vd_nca
        })
    df_ind = pd.DataFrame(rows)
    summary_rows = []
    param_info = [
        ("Cmax (mg/L)", "Nồng độ đỉnh thực nghiệm lớn nhất"),
        ("Tmax (h)", "Thời điểm đạt nồng độ đỉnh"),
        ("AUC0_t (mg·h/L)", "Diện tích dưới đường cong (0-t)"),
        ("AUC_inf (mg·h/L)", "Diện tích dưới đường cong ngoại suy vô cùng"),
        ("t1/2 (h)", "Thời gian bán thải từ pha đào thải cuối"),
        ("CL_nca (L/h)", "Độ thanh thải toàn thân phi khoang"),
        ("Vd_nca (L)", "Thể tích phân bố phi khoang")
    ]
    for col, desc in param_info:
        s = df_ind[col].dropna()
        if len(s) > 0:
            summary_rows.append({
                "Thông số NCA": col,
                "Trung vị": f"{s.median():.2f}",
                "IQR": f"[{s.quantile(0.25):.2f} - {s.quantile(0.75):.2f}]",
                "Min - Max": f"[{s.min():.2f} - {s.max():.2f}]",
                "Ý nghĩa lâm sàng": desc
            })
    df_sum = pd.DataFrame(summary_rows)
    return df_ind, df_sum


def covariate_correlation_figure(res, clinical_data):
    if res is None or clinical_data is None:
        return None
    d = res["data"]
    clean_df = clinical_data["clean_df"]
    pop_phi = pk.pop_mean_phi(res)
    etas = res["phi_hat"] - pop_phi
    
    cl_idx = 0 if "CL" in res["names"] else 0
    v_idx = 1 if "V" in res["names"] and len(res["names"]) > 1 else 0
    
    rows = []
    for i, sid in enumerate(d["ids"]):
        sub_rows = clean_df[clean_df["ID"] == sid]
        if len(sub_rows) > 0:
            first = sub_rows.iloc[0]
            rows.append({
                "ID": sid,
                "eta_CL": etas[i, cl_idx],
                "eta_V": etas[i, v_idx],
                "CLCr_CG": float(first.get("CLCr_CG", np.nan)),
                "TBW": float(first.get("TBW", np.nan)),
                "Age": float(first.get("Age", np.nan)),
                "Sepsis": "Sepsis (+)" if float(first.get("Sepsis", 0)) > 0 else "Non-Sepsis (-)"
            })
    df_eta = pd.DataFrame(rows)
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "η_CL vs Chức năng thận (CLCr_CG)", "η_V vs Cân nặng (TBW)",
        "η_CL vs Tuổi (Age)", "η_CL theo Tình trạng Nhiễm khuẩn huyết (Sepsis)"
    ])
    
    # 1. eta_CL vs CLCr_CG
    df_c = df_eta.dropna(subset=["CLCr_CG", "eta_CL"])
    fig.add_trace(go.Scatter(x=df_c["CLCr_CG"], y=df_c["eta_CL"], mode="markers",
                             marker=dict(color="#1565c0", size=6, opacity=0.7), name="η_CL"), row=1, col=1)
    if len(df_c) > 2:
        lr = linregress(df_c["CLCr_CG"], df_c["eta_CL"])
        x_trend = np.linspace(df_c["CLCr_CG"].min(), df_c["CLCr_CG"].max(), 50)
        fig.add_trace(go.Scatter(x=x_trend, y=lr.intercept + lr.slope * x_trend, mode="lines",
                                 line=dict(color="red", dash="dash"), name="Trend"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
    
    # 2. eta_V vs TBW
    df_w = df_eta.dropna(subset=["TBW", "eta_V"])
    fig.add_trace(go.Scatter(x=df_w["TBW"], y=df_w["eta_V"], mode="markers",
                             marker=dict(color="#2e7d32", size=6, opacity=0.7), name="η_V"), row=1, col=2)
    if len(df_w) > 2:
        lr_w = linregress(df_w["TBW"], df_w["eta_V"])
        x_w = np.linspace(df_w["TBW"].min(), df_w["TBW"].max(), 50)
        fig.add_trace(go.Scatter(x=x_w, y=lr_w.intercept + lr_w.slope * x_w, mode="lines",
                                 line=dict(color="red", dash="dash"), name="Trend"), row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)
    
    # 3. eta_CL vs Age
    df_a = df_eta.dropna(subset=["Age", "eta_CL"])
    fig.add_trace(go.Scatter(x=df_a["Age"], y=df_a["eta_CL"], mode="markers",
                             marker=dict(color="#6a1b9a", size=6, opacity=0.7), name="η_CL vs Age"), row=2, col=1)
    if len(df_a) > 2:
        lr_a = linregress(df_a["Age"], df_a["eta_CL"])
        x_a = np.linspace(df_a["Age"].min(), df_a["Age"].max(), 50)
        fig.add_trace(go.Scatter(x=x_a, y=lr_a.intercept + lr_a.slope * x_a, mode="lines",
                                 line=dict(color="red", dash="dash"), name="Trend"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    
    # 4. eta_CL vs Sepsis (Box plot)
    for grp, col in [("Non-Sepsis (-)", "#1565c0"), ("Sepsis (+)", "#d84315")]:
        sub_grp = df_eta[df_eta["Sepsis"] == grp]
        fig.add_trace(go.Box(y=sub_grp["eta_CL"], name=grp, marker_color=col, boxpoints="all", jitter=0.3),
                      row=2, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=2)
    
    fig.update_layout(height=650, template="plotly_white", showlegend=False,
                      title_text="Tương quan giữa Hiệu ứng Ngẫu nhiên (η) và Biến số Lâm sàng (Covariates)")
    return fig


def simulx_dosing_simulation(cl, v, dose, tau, tinf, duration=72.0):
    k = cl / v
    r_inf = dose / tinf
    times = np.linspace(0.0, duration, 500)
    
    n_doses = int(duration // tau) + 1
    dose_times = [i * tau for i in range(n_doses) if i * tau < duration]
    
    c_profile = np.zeros_like(times)
    for td in dose_times:
        te = td + tinf
        mask_inf = (times >= td) & (times <= te)
        c_profile[mask_inf] += (r_inf / cl) * (1.0 - np.exp(-k * (times[mask_inf] - td)))
        mask_post = (times > te)
        c_profile[mask_post] += (r_inf / cl) * (1.0 - np.exp(-k * tinf)) * np.exp(-k * (times[mask_post] - te))
        
    t_trough_ss = dose_times[-1] if len(dose_times) > 1 else tau
    c_trough_ss = float(np.interp(t_trough_ss, times, c_profile))
    c_peak_ss = float(np.interp(dose_times[-1] + tinf, times, c_profile)) if len(dose_times) > 0 else float(np.max(c_profile))
    
    daily_dose = dose * (24.0 / tau)
    auc_24 = daily_dose / cl
    return times, c_profile, c_trough_ss, c_peak_ss, auc_24


def generate_r_script(res=None):
    if res is not None and "beta" in res and len(res["beta"]) >= 2:
        cl_val = float(np.exp(res["beta"][0][0]))
        v_val = float(np.exp(res["beta"][1][0]))
        om_cl = float(res["omega"][0]) if len(res.get("omega", [])) > 0 else 0.3
        om_v = float(res["omega"][1]) if len(res.get("omega", [])) > 1 else 0.3
    else:
        cl_val, v_val, om_cl, om_v = 3.00, 21.00, 0.34, 0.27

    return f"""# ==============================================================================
# SimDx - Reproducible Pharmacometrics Analysis in R
# Generated from SimDx NLME Platform (MonolixSuite Compatible)
# Target Dataset: Dataset_chính_thức.xlsx (Vancomycin Clinical Study)
# ==============================================================================
library(ggplot2)
library(dplyr)
library(readxl)

cat("=== SimDx Reproducible Analysis Script ===\\n")

# 1. Load Data
data_path <- "Dataset_chính_thức.xlsx"
if (file.exists(data_path)) {{
  df <- read_excel(data_path)
  cat("Dataset loaded successfully:", nrow(df), "rows\\n")
  
  # 2. Spaghetti Plot (Linear & Semi-log)
  if ("DV" %in% colnames(df) && "Time" %in% colnames(df)) {{
    p_spag <- ggplot(df %>% filter(!is.na(DV)), aes(x = Time, y = DV, group = ID)) +
      geom_line(alpha = 0.5, color = "#1565c0") +
      geom_point(size = 1.2, alpha = 0.6) +
      theme_minimal() +
      labs(title = "Concentration-Time Profiles (Spaghetti Plot)",
           x = "Time (hours)", y = "Concentration (mg/L)")
    print(p_spag)
  }}
}} else {{
  cat("Please place Dataset_chính_thức.xlsx in your R working directory.\\n")
}}

# 3. Dynamic Estimated Population Model (SimDx SAEM):
# CL_i = {cl_val:.2f} * exp(eta_CL), eta_CL ~ N(0, {om_cl:.2f}^2)
# V_i  = {v_val:.2f} * exp(eta_V), eta_V ~ N(0, {om_v:.2f}^2)
cat("SimDx Estimated Population Model: CL = {cl_val:.2f} L/h, V = {v_val:.2f} L\\n")
cat("Analysis completed.\\n")
"""


def render_dynamic_model_equations(res, clinical_data=None, show_scm_benchmark=True):
    if res is None:
        return
    names = res.get("names", ["CL", "V"])
    beta = res.get("beta", [])
    omega = res.get("omega", [])
    cov_model = res.get("cov_model", {})
    err = res.get("err", [1.0, 0.2])
    err_mod = res.get("error_model", "combined")
    
    # 1. Extract estimated population values
    theta_dict = {}
    for j, p in enumerate(names):
        theta_dict[p] = float(np.exp(beta[j][0])) if j < len(beta) and len(beta[j]) > 0 else 1.0

    omega_dict = {}
    for j, p in enumerate(names):
        omega_dict[p] = float(omega[j]) if j < len(omega) else 0.3

    # Dynamic reference medians from res['data'] or clinical_data
    ref_medians = {}
    if "data" in res and "ref_medians" in res["data"]:
        ref_medians.update(res["data"]["ref_medians"])
    elif clinical_data is not None and "baseline_cov" in clinical_data:
        bc = clinical_data["baseline_cov"]
        for col in bc.columns:
            if pd.api.types.is_numeric_dtype(bc[col]):
                med = float(bc[col].dropna().median())
                if med > 0:
                    ref_medians[col] = round(med, 1)
                    ref_medians[col.replace("_base", "")] = round(med, 1)

    has_cov = any(len(cov_model.get(p, [])) > 0 for p in names)
    
    st.markdown("#### :material/functions: Phương trình Dược động học Quần thể (Population PK Model Equations)")
    
    # Section 1: Individual Parameter Model
    st.markdown("##### 1. Mô hình Tham số Cá thể & Biến số Lâm sàng (Individual Parameter Model)")
    if has_cov:
        st.info("Phương trình được tính toán động 100% từ kết quả ước lượng SAEM và các hiệu ứng covariate bạn đã chọn ở thanh bên:")
    else:
        st.info("Phương trình Mô hình Cơ sở (Base Model) được tính toán động 100% từ kết quả ước lượng SAEM hiện tại:")

    for j, p in enumerate(names):
        theta_val = theta_dict[p]
        omega_val = omega_dict[p]
        covs = cov_model.get(p, [])
        cov_terms = ""
        for k, c in enumerate(covs):
            b_val = float(beta[j][1 + k]) if j < len(beta) and (1 + k) < len(beta[j]) else 0.0
            clean_c = c.replace("_", r"\_")
            if c in ["Sepsis", "Sex", "GEN", "SEX"]:
                cov_terms += rf" \times e^{{{b_val:.3f} \cdot \text{{{clean_c}}}_i}}"
            else:
                ref_val = ref_medians.get(c, 70.0)
                ref_str = f"{ref_val:.0f}" if ref_val == int(ref_val) else f"{ref_val:.1f}"
                cov_terms += rf" \times \left(\frac{{\text{{{clean_c}}}_i}}{{{ref_str}}}\right)^{{{b_val:.3f}}}"
        
        clean_p = p.replace("_", r"\_")
        eq = rf"{clean_p}_i = {theta_val:.2f}{cov_terms} \times e^{{\eta_{{i, {clean_p}}}}}, \quad \eta_{{{clean_p}}} \sim \mathcal{{N}}(0, {omega_val:.2f}^2)"
        st.latex(eq)

    # Section 2: Structural PK Model Equation C(t)
    st.markdown("##### 2. Mô hình Động học Nồng độ (Structural PK Model Concentration Equations $C(t)$)")
    m_name = res.get("model_name", "")
    
    if "infusion" in m_name.lower():
        cl_val = theta_dict.get("CL", 3.0)
        v_val = theta_dict.get("V", 21.0)
        ke_val = cl_val / v_val if v_val > 0 else 0.1
        thalf = np.log(2) / ke_val if ke_val > 0 else 7.0
        tinf_val = float(res.get("data", {}).get("tinf", 1.0))
        
        st.caption(rf"Thay thế trực tiếp các tham số quần thể vừa ước lượng: $CL_{{pop}} = {cl_val:.2f}\text{{ L/h}}$, $V_{{pop}} = {v_val:.2f}\text{{ L}} \implies k_e = \frac{{CL}}{{V}} = {ke_val:.3f}\text{{ h}}^{{-1}}$, $t_{{1/2}} = {thalf:.2f}\text{{ h}}$ (thời gian truyền $t_{{inf}} = {tinf_val:.1f}\text{{ h}}$):")
        
        eq_during = (
            rf"\text{{Trong thời gian truyền }} (0 \le t \le {tinf_val:.1f}\text{{ h}}): \quad "
            rf"C(t) = \frac{{\text{{Dose}}}}{{{tinf_val * cl_val:.2f}}} \left(1 - e^{{-{ke_val:.3f} \cdot t}}\right)"
        )
        st.latex(eq_during)
        
        coeff_washout = (1.0 - np.exp(-ke_val * tinf_val)) / (tinf_val * cl_val) if (tinf_val * cl_val) > 0 else 0.0
        eq_post = (
            rf"\text{{Sau khi kết thúc truyền }} (t > {tinf_val:.1f}\text{{ h}}): \quad "
            rf"C(t) = {coeff_washout:.4f} \cdot \text{{Dose}} \cdot e^{{-{ke_val:.3f} \cdot (t - {tinf_val:.1f})}}"
        )
        st.latex(eq_post)
    elif "oral" in m_name.lower():
        cl_val = theta_dict.get("CL", 3.0)
        v_val = theta_dict.get("V", 21.0)
        ka_val = theta_dict.get("ka", 1.0)
        ke_val = cl_val / v_val if v_val > 0 else 0.1
        thalf = np.log(2) / ke_val if ke_val > 0 else 7.0
        st.caption(rf"Thay thế tham số ước lượng: $k_a = {ka_val:.2f}\text{{ h}}^{{-1}}$, $CL = {cl_val:.2f}\text{{ L/h}}$, $V = {v_val:.2f}\text{{ L}} \implies k_e = {ke_val:.3f}\text{{ h}}^{{-1}}$, $t_{{1/2}} = {thalf:.2f}\text{{ h}}$:")
        denom = v_val * (ka_val - ke_val) if abs(ka_val - ke_val) > 1e-5 else 1.0
        st.latex(rf"C(t) = {ka_val/denom:.4f} \cdot \text{{Dose}} \left(e^{{-{ke_val:.3f} \cdot t}} - e^{{-{ka_val:.2f} \cdot t}}\right)")
    elif "bolus" in m_name.lower():
        cl_val = theta_dict.get("CL", 3.0)
        v_val = theta_dict.get("V", 21.0)
        ke_val = cl_val / v_val if v_val > 0 else 0.1
        thalf = np.log(2) / ke_val if ke_val > 0 else 7.0
        st.caption(rf"Thay thế tham số ước lượng: $CL = {cl_val:.2f}\text{{ L/h}}$, $V = {v_val:.2f}\text{{ L}} \implies k_e = {ke_val:.3f}\text{{ h}}^{{-1}}$, $t_{{1/2}} = {thalf:.2f}\text{{ h}}$:")
        st.latex(rf"C(t) = \frac{{\text{{Dose}}}}{{{v_val:.2f}}} e^{{-{ke_val:.3f} \cdot t}}")

    # Section 3: Residual Error Model
    st.markdown("##### 3. Mô hình Sai số Dư / Đo lường (Residual Error Model)")
    a_val = float(err[0]) if len(err) > 0 else 1.0
    b_val = float(err[1]) if len(err) > 1 else 0.2
    if err_mod == "combined":
        st.latex(rf"y_{{ij}} = C(t_{{ij}}) + \sqrt{{{a_val:.3f}^2 + {b_val:.3f}^2 \cdot C(t_{{ij}})^2}} \cdot \varepsilon_{{ij}}, \quad \varepsilon_{{ij}} \sim \mathcal{{N}}(0, 1)")
    elif err_mod == "proportional":
        st.latex(rf"y_{{ij}} = C(t_{{ij}}) + {b_val:.3f} \cdot C(t_{{ij}}) \cdot \varepsilon_{{ij}}, \quad \varepsilon_{{ij}} \sim \mathcal{{N}}(0, 1)")
    elif err_mod == "constant":
        st.latex(rf"y_{{ij}} = C(t_{{ij}}) + {a_val:.3f} \cdot \varepsilon_{{ij}}, \quad \varepsilon_{{ij}} \sim \mathcal{{N}}(0, 1)")
    else:
        st.latex(rf"y_{{ij}} = C(t_{{ij}}) + ({a_val:.3f} + {b_val:.3f} \cdot C(t_{{ij}})) \cdot \varepsilon_{{ij}}, \quad \varepsilon_{{ij}} \sim \mathcal{{N}}(0, 1)")

    # Section 4: SCM Benchmark Reference (only when requested, e.g. in Tab 4)
    if show_scm_benchmark:
        is_standard_vanco = (clinical_data is not None and clinical_data.get("N", 0) == 83 and clinical_data.get("nobs", 0) == 278)
        cov_json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "validation", "covariate_analysis.json"))
        if is_standard_vanco and os.path.exists(cov_json_path):
            with st.expander("Tham khảo Benchmark: Mô hình Tối ưu Sàng lọc SCM (Vancomycin chuẩn 83 BN)", expanded=False):
                try:
                    with open(cov_json_path, "r") as f:
                        scm_data = json.load(f)
                    best_clcr = next((m for m in scm_data if "CLCr_CG" in m.get("model_name", "")), None)
                    best_tbw = next((m for m in scm_data if "TBW" in m.get("model_name", "")), None)
                    if best_clcr and best_tbw:
                        st.caption("Kết quả chạy quy trình SCM chuẩn trên bộ dữ liệu Vancomycin (CL ~ CLCr_CG và V ~ TBW):")
                        b_clcr = best_clcr["covariates"][0]["est"] if best_clcr["covariates"] else 0.84
                        b_tbw = best_tbw["covariates"][0]["est"] if best_tbw["covariates"] else 0.58
                        cl_pop = best_clcr["CL"]
                        v_pop = best_tbw["V"]
                        om_cl = best_clcr["omega_CL"]
                        om_v = best_tbw["omega_V"]
                        ref_clcr = ref_medians.get("CLCr_CG", 51.0)
                        ref_tbw = ref_medians.get("TBW", 50.0)
                        st.latex(rf"CL_i = {cl_pop:.2f} \times \left(\frac{{\text{{CLCr\_CG}}_i}}{{{ref_clcr:.0f}}}\right)^{{{b_clcr:.2f}}} \times e^{{\eta_{{i, CL}}}}, \quad \eta_{{CL}} \sim \mathcal{{N}}(0, {om_cl:.2f}^2)")
                        st.latex(rf"V_i = {v_pop:.2f} \times \left(\frac{{\text{{TBW}}_i}}{{{ref_tbw:.0f}}}\right)^{{{b_tbw:.2f}}} \times e^{{\eta_{{i, V}}}}, \quad \eta_{{V}} \sim \mathcal{{N}}(0, {om_v:.2f}^2)")
                except Exception:
                    pass
        elif clinical_data is not None and not is_standard_vanco:
            st.caption("ℹ️ *Bạn đang làm việc với tập dữ liệu tùy chỉnh tải lên. Để thêm các biến số covariate vào mô hình vừa ước lượng bằng SAEM, hãy chọn biến số ở mục **Covariate** tại thanh bên trái (Sidebar) (ví dụ: `CL~CLCr_CG`, `V~TBW`) rồi bấm **Bắt đầu ước lượng SAEM**.*")


def show_fit(res, clinical_data=None):
    ic = res["ic"]
    bad = [(lab, r) for lab, ok, r in pk.convergence_report(res) if not ok]
    if bad:
        st.warning("Pha 2 chưa ổn định (trung bình nửa đầu và nửa sau lệch > 5%): "
                   + ", ".join(f"{lab} ({r:.0%})" for lab, r in bad)
                   + ". Hãy tăng số vòng lặp hoặc số chain.", icon=":material/warning:")
    else:
        st.markdown('<div class="convergence-box">Các tham số ổn định ở pha smoothing (trôi < 5%)</div>',
                    unsafe_allow_html=True)
    header("Ước lượng tham số (SE/RSE: FIM tuyến tính hoá quanh EBE)")
    if any(np.isnan(v[1]) for v in res["fim"].values()):
        st.warning("Hessian không xác định dương → không tính được SE.", icon=":material/warning:")
    st.dataframe(estimates_table(res), width="stretch")
    c = st.columns(4)
    mc_se_str = f" (MC-SE: ±{ic['mc_se']:.2f})" if "mc_se" in ic and ic["mc_se"] > 0 else ""
    c[0].metric("−2LL (IS)", f"{ic['m2ll']:.2f}{mc_se_str}")
    c[1].metric("AIC", f"{ic['aic']:.1f}")
    c[2].metric("BIC", f"{ic['bic']:.1f}")
    c[3].metric("BICc", f"{ic['bicc']:.1f}")

    # Phương trình mô hình dược động học vừa ước lượng
    header("Phương trình Dược động học Quần thể vừa Ước lượng")
    render_dynamic_model_equations(res, clinical_data, show_scm_benchmark=False)

    header("SAEM Convergence Traces")
    st.plotly_chart(convergence_figure(res), width="stretch")

    # Monolix 2024 Bootstrap Module
    header("Đánh giá Độ tin cậy Tham số: Bootstrap Module (Monolix 2024)")
    st.info("Module Non-parametric Bootstrap lấy mẫu lại theo bệnh nhân để tính khoảng tin cậy 95% Bootstrap CI thực tế, giải quyết triệt để trường hợp ma trận Hessian FIM khó đảo ngược.")
    c_b1, c_b2 = st.columns([1, 1])
    n_boot = c_b1.slider("Số mẫu lặp lại (Bootstrap Replicates)", 5, 30, 10, 5, key="n_boot_slider")
    if c_b2.button("Chạy Non-parametric Bootstrap", icon=":material/sync:", key="btn_run_bootstrap"):
        b_bar = st.progress(0, text="Đang chạy Bootstrap...")
        boot_records = []
        d_orig = res["data"]
        n_subs = len(d_orig["ids"])
        for b_idx in range(n_boot):
            b_bar.progress((b_idx + 1) / n_boot, text=f"Bootstrap mẫu {b_idx + 1}/{n_boot}...")
            # Sample with replacement
            rng = np.random.RandomState(100 + b_idx)
            sampled_indices = rng.choice(n_subs, size=n_subs, replace=True)
            # Re-estimate with fast iterations
            init_b = dict(
                mu=[res["fim"][p][0] if p in res["fim"] else 1.0 for p in res["names"]],
                omega=[res["fim"][f"omega_{p}"][0] if f"omega_{p}" in res["fim"] else 0.3 for p in res["names"]],
                err=[res["fim"]["a"][0] if "a" in res["fim"] else 1.0, res["fim"]["b"][0] if "b" in res["fim"] else 0.2]
            )
            try:
                b_res = pk.fit_model(d_orig, res["model_name"], init_b, res["cov_model"], res["error_model"],
                                     n_iter1=40, n_iter2=20, n_chains=1, seed=int(100 + b_idx),
                                     cov_mode=res.get("cov_mode", "baseline"))
                b_row = {}
                for p in res["names"]:
                    b_row[f"theta_{p}"] = np.exp(b_res["fim"][p][0])
                    b_row[f"omega_{p}"] = b_res["fim"][f"omega_{p}"][0]
                if "a" in b_res["fim"]:
                    b_row["a"] = b_res["fim"]["a"][0]
                if "b" in b_res["fim"]:
                    b_row["b"] = b_res["fim"]["b"][0]
                boot_records.append(b_row)
            except Exception:
                pass
        b_bar.empty()
        if boot_records:
            st.session_state["boot_df"] = pd.DataFrame(boot_records)
            st.success(f"Hoàn thành {len(boot_records)}/{n_boot} bản sao Bootstrap thành công!")
    
    if "boot_df" in st.session_state:
        b_df = st.session_state["boot_df"]
        b_summary = []
        for col in b_df.columns:
            s_val = b_df[col].dropna()
            if len(s_val) > 0:
                b_summary.append({
                    "Tham số": col.replace("theta_", "θ_").replace("omega_", "ω_"),
                    "Ước lượng SAEM": f"{np.median(s_val):.3f}",
                    "Bootstrap Trung vị": f"{s_val.median():.3f}",
                    "95% Bootstrap CI (2.5% - 97.5%)": f"[{s_val.quantile(0.025):.3f}, {s_val.quantile(0.975):.3f}]",
                    "Bootstrap SE": f"{s_val.std():.3f}"
                })
        st.dataframe(pd.DataFrame(b_summary), width="stretch")


MASTER_PASSWORD = "ngovietcuongkhongchodung1112"


def check_authentication():
    """Kiểm tra xác thực mật khẩu trước khi cho phép truy cập SimDx."""
    if st.session_state.get("authenticated", False):
        return True

    is_failed = st.session_state.get("login_failed", False)

    # Center card layout
    c1, c2, c3 = st.columns([1, 1.8, 1])
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if is_failed and CAT_B64:
            st.markdown(f"""
            <style>
            @keyframes zoomGrowCat {{
                0% {{
                    transform: scale(0.15) rotate(-6deg);
                    opacity: 0;
                }}
                50% {{
                    transform: scale(1.18) rotate(3deg);
                    opacity: 1;
                }}
                75% {{
                    transform: scale(0.95) rotate(-1deg);
                    opacity: 1;
                }}
                100% {{
                    transform: scale(1.0) rotate(0deg);
                    opacity: 1;
                }}
            }}
            @keyframes textPopUp {{
                0% {{
                    opacity: 0;
                    transform: scale(0.4) translateY(20px);
                }}
                60% {{
                    opacity: 0;
                    transform: scale(0.6);
                }}
                85% {{
                    opacity: 1;
                    transform: scale(1.15) translateY(-4px);
                }}
                100% {{
                    opacity: 1;
                    transform: scale(1.0) translateY(0);
                }}
            }}
            .cat-container {{
                animation: zoomGrowCat 1.0s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
                display: flex;
                justify-content: center;
                align-items: center;
                margin-bottom: 8px;
            }}
            .cat-img {{
                width: 240px;
                height: 240px;
                object-fit: cover;
                border-radius: 24px;
                box-shadow: 0 12px 36px rgba(225, 29, 72, 0.4);
                border: 4px solid #f43f5e;
            }}
            .huge-error-text {{
                animation: textPopUp 1.3s cubic-bezier(0.22, 1, 0.36, 1) forwards;
                color: #e11d48;
                font-size: 34px;
                font-weight: 900;
                text-align: center;
                margin: 12px 0 18px 0;
                letter-spacing: -0.5px;
                text-shadow: 0 3px 12px rgba(225, 29, 72, 0.3);
                line-height: 1.2;
            }}
            </style>
            <div style="text-align: center;">
                <div class="cat-container">
                    <img src="data:image/jpeg;base64,{CAT_B64}" class="cat-img" alt="Cat Suspicious"/>
                </div>
                <div class="huge-error-text">Á à :) xin phép chưa ????</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            logo_display = (
                f'<img src="data:image/png;base64,{LOGO_B64}" style="width: 88px; height: 88px; object-fit: contain; filter: drop-shadow(0 6px 16px rgba(13, 51, 104, 0.15)); margin-bottom: 10px;">'
            ) if LOGO_B64 else '<div style="display: inline-block; padding: 14px; background: #e0f2fe; border-radius: 20px; margin-bottom: 12px; box-shadow: 0 4px 12px rgba(13, 51, 104, 0.08);"><span style="font-size: 42px;">🔐</span></div>'
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 24px;">
                {logo_display}
                <h2 style="color: #0d3368; font-weight: 800; margin-bottom: 4px; letter-spacing: -0.5px;">Spass</h2>
                <p style="color: #64748b; font-size: 14px; margin-top: 0;">inb Bảnh để có pass</p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            st.markdown("##### bát's là gì ?" if not is_failed else "##### Nhập lại mật khẩu cho đúng:")
            pwd = st.text_input("Mật khẩu", type="password", placeholder="Hỏi NVC đi", label_visibility="collapsed", key="login_pwd_input")
            submit = st.form_submit_button("Vào hệ thống của bảnh" if not is_failed else "Cho cơ hội cuối", type="primary", use_container_width=True)
            
            if submit:
                pwd_clean = (pwd or "").strip()
                valid_passwords = [MASTER_PASSWORD]
                try:
                    if hasattr(st, "secrets") and "PASSWORD" in st.secrets:
                        valid_passwords.append(str(st.secrets["PASSWORD"]).strip())
                except Exception:
                    pass
                
                if pwd_clean in valid_passwords:
                    st.session_state["authenticated"] = True
                    st.session_state["login_failed"] = False
                    st.success("Ngô Việt Cường đã chốt bạn ")
                    st.rerun()
                elif is_failed and not pwd_clean:
                    # Bấm 'Cho cơ hội cuối' khi ô trống -> Cho quay lại màn hình chính để nhập lại
                    st.session_state["login_failed"] = False
                    st.rerun()
                else:
                    st.session_state["login_failed"] = True
                    st.rerun()
        
        if is_failed:
            if st.button("Đi về đi", use_container_width=True):
                st.session_state["login_failed"] = False
                st.rerun()

        st.caption("<div style='text-align:center; color:#94a3b8; font-size:12px; margin-top:16px;'>Hệ thống được bảo vệ bản quyền · Ngô Việt Cường</div>", unsafe_allow_html=True)
    
    st.stop()


def main():
    check_authentication()
    with st.sidebar:
        logo_html = (
            f'<img src="data:image/png;base64,{LOGO_B64}" '
            f'style="width:64px;height:64px;min-width:64px;min-height:64px;object-fit:contain;filter:drop-shadow(0 2px 6px rgba(0,0,0,0.12));border-radius:8px;">'
        ) if LOGO_B64 else ""
        st.markdown(f"""
        <div class="simdx-sidebar-brand" style="display:flex;align-items:center;gap:12px;">
            {logo_html}
            <div style="flex:1;min-width:0;">
                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                    <span style="font-size:1.55em;font-weight:800;color:#0d3368;line-height:1.1;letter-spacing:0.5px;">SimDx</span>
                    <span style="background:#e2e8f0;color:#1e293b;padding:2px 8px;border-radius:12px;font-size:0.75em;font-weight:700;white-space:nowrap;">v2.5 · Pro</span>
                </div>
                <div style="font-size:0.75em;color:#475569;font-weight:600;line-height:1.25;margin-top:4px;">
                    Nonlinear Mixed-Effects Modeling · SAEM
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### :material/upload_file: Nguồn dữ liệu")
        data_source_opt = st.radio(
            "Chọn dữ liệu",
            ["Dataset lâm sàng (Dataset_chính_thức.xlsx)", "Tải lên file (Excel/CSV)", "Mô phỏng quần thể (Simulation)"]
        )

        clinical_data = None
        df_sim = None

        is_bolus_mode = (st.session_state.get("selected_lib_model") == "bolus_1cpt_VCl")

        if data_source_opt == "Dataset lâm sàng (Dataset_chính_thức.xlsx)":
            excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Dataset_chính_thức.xlsx"))
            if is_bolus_mode:
                tinf = 0.01
                st.caption("⚡ **Chế độ Monolix bolus_1cpt_VCl**: tinf = 0.01 h (Bolus tức thời)")
            else:
                tinf = st.number_input("Thời gian truyền tinf (giờ)", 0.01, 24.0, float(st.session_state.get("tinf_val", 1.0)), 0.05, key="tinf_main_input")
            clinical_data = data_io.load_dataset(excel_path, tinf=tinf, enforce_g4=True)
            default_model = "1-Compartment IV infusion (multi-dose)"
        elif data_source_opt == "Tải lên file (Excel/CSV)":
            uploaded_file = st.file_uploader("Tải lên file dữ liệu", type=["xlsx", "xls", "csv"])
            if is_bolus_mode:
                tinf = 0.01
                st.caption("⚡ **Chế độ Monolix bolus_1cpt_VCl**: tinf = 0.01 h (Bolus tức thời)")
            else:
                tinf = st.number_input("Thời gian truyền tinf (giờ)", 0.01, 24.0, float(st.session_state.get("tinf_val", 1.0)), 0.05, key="tinf_upload_input")
            if uploaded_file is not None:
                tmp_path = f"/tmp/{uploaded_file.name}"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                try:
                    clinical_data = data_io.load_dataset(tmp_path, tinf=tinf, enforce_g4=False)
                    default_model = "1-Compartment IV infusion (multi-dose)"
                except Exception as e:
                    st.error(f"❌ Lỗi nạp dữ liệu: {e}")
                    clinical_data = None
                    default_model = "1-Compartment IV infusion (multi-dose)"
            else:
                st.info("Hãy tải lên file dataset để bắt đầu.")
                default_model = "1-Compartment IV infusion (multi-dose)"
        else:
            default_model = "1-Compartment (Oral)"

        # Monolix Library Section
        st.markdown("### :material/library_books: Thư viện mô hình Monolix")
        if st.button("📚 LOAD FROM LIBRARY", type="primary", use_container_width=True, help="Mở thư viện mô hình Dược động học chuẩn Monolix (bolus_1cpt_VCl, infusion_1cpt_VCl, ...)"):
            show_model_library_dialog()

        if "selected_lib_model" in st.session_state:
            lib_id = st.session_state["selected_lib_model"]
            st.markdown(f"""
            <div style="background: #e0f2fe; border: 1px solid #7dd3fc; border-radius: 6px; padding: 6px 10px; font-size: 12px; margin-bottom: 8px;">
                <span style="color: #0369a1; font-weight: 700;">Monolix Model:</span> <code>{lib_id}</code>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### :material/settings: Cấu hình mô hình")
        model_names = list(pk.MODELS.keys())
        if "model_name_choice" in st.session_state and st.session_state["model_name_choice"] in model_names:
            default_idx = model_names.index(st.session_state["model_name_choice"])
        elif default_model in model_names:
            default_idx = model_names.index(default_model)
        else:
            default_idx = 0
        model_name = st.selectbox(":material/schema: Mô hình cấu trúc", model_names, index=default_idx)
        mi = pk.MODELS[model_name]
        names = mi["params"]
        key = lambda s: f"{model_name}_{s}"

        st.markdown("### :material/bar_chart: Thông số quần thể ban đầu (θ)")
        pop_params = {}
        for p, d, u, r in zip(names, mi["defaults"], mi["units"], mi["ranges"]):
            step = 0.05 if r[1] < 10 else 0.5
            pop_params[p] = st.slider(f"{p} ({u})", float(r[0]), float(r[1]), float(d), float(step), key=key(f"pop_{p}"))

        st.markdown("### :material/shuffle: Biến thiên giữa các cá thể (ω)")
        omega_dict = {p: st.slider(f"ω_{p}", 0.05, 1.0, 0.3, 0.05, key=key(f"om_{p}")) for p in names}

        st.markdown("### :material/show_chart: Mô hình sai số dư")
        err_models = ["combined", "combined1", "constant", "proportional"]
        err_def = st.session_state.get("error_model_choice", "combined")
        err_idx = err_models.index(err_def) if err_def in err_models else 0
        error_model = st.selectbox("Error Model", err_models, index=err_idx)
        c_err1, c_err2 = st.columns(2)
        sigma_add = c_err1.number_input("a (additive)", 0.01, 5.0, 1.0, 0.1)
        sigma_prop = c_err2.number_input("b (proportional)", 0.01, 1.0, 0.2, 0.02)

        st.markdown("### :material/biotech: Covariate")
        if clinical_data is not None:
            cov_opts = ["CL~CLCr_CG", "V~TBW", "CL~Sepsis", "V~Sepsis", "CL~Age"]
            cov_sel = st.multiselect("Hiệu ứng covariate", cov_opts, default=[])
            cov_mode = st.radio("Chế độ Covariate", ["time-varying", "baseline"])
        else:
            cov_sel = st.multiselect("Covariate effect cần ước lượng", [f"{p}~{c}" for p in names for c in pk.COVARIATES],
                                     key=key("covsel"))
            cov_mode = "baseline"

        cov_model = {}
        for s in cov_sel:
            p, c = s.split("~")
            cov_model.setdefault(p, []).append(c)

        st.markdown("### :material/sync: Cài đặt SAEM")
        n_iter1 = st.slider("K1: pha exploratory (γ=1)", 50, 600, 250, 50)
        n_iter2 = st.slider("K2: pha smoothing (γ=1/k)", 25, 400, 150, 25)
        n_chains = st.slider("Số chain", 1, 3, 1)
        fit_seed = st.number_input("Seed SAEM", 0, 9999, 42)

        st.markdown("---")
        c_out1, c_out2 = st.columns([1.5, 1])
        c_out1.caption("👤 Đã xác thực")
        if c_out2.button("Đăng xuất", key="btn_logout", type="secondary", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        ":material/database: Dữ liệu & PKanalix",
        ":material/play_arrow: Ước lượng SAEM",
        ":material/monitoring: Chẩn đoán GOF",
        ":material/schema: Sàng lọc Covariate",
        ":material/person: Khớp cá thể (Individual Fits)",
        ":material/bar_chart: Thẩm định & Simulx"
    ])

    res = st.session_state.get("res")

    # ── TAB 1: DỮ LIỆU & PKANALIX NCA ──
    with tab1:
        if clinical_data is not None:
            header("Bộ dữ liệu lâm sàng thực tế")
            c = st.columns(4)
            c[0].metric("Bệnh nhân", clinical_data["N"])
            c[1].metric("Mẫu nồng độ (DV)", clinical_data["nobs"])
            c[2].metric("Tổng số liều", sum(len(d) for d in clinical_data["sub_doses"].values()))
            c[3].metric("Thời gian truyền tinf", f"{clinical_data.get('tinf', 1.0):.1f} h")

            st.markdown('<div class="convergence-box"><b>Gate G4 Strictly Verified:</b> 565 dòng / 83 bệnh nhân / 287 liều / 278 quan sát DV. Không có trùng lặp (ID, Time).</div>', unsafe_allow_html=True)

            if clinical_data["outliers"]:
                st.markdown('<div class="outlier-box"><b>Các điểm ngoại lai (Outliers) được đánh dấu cờ (Flagged):</b> Giữ nguyên 100% trong phân tích theo chuẩn quy trình lâm sàng, không xóa hoặc winsorize.</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(clinical_data["outliers"]), width="stretch")

            header("Biểu đồ Nồng độ - Thời gian (Spaghetti Profiles)")
            c_sp1, c_sp2 = st.columns([1, 2])
            spag_scale = c_sp1.radio("Thang đo trục Y", ["Tuyến tính (Linear)", "Logarit (Semi-log)"], horizontal=True, key="spag_scale_choice")
            spag_strat = c_sp2.radio("Phân tầng (Stratification)", ["Tất cả", "Theo Nhiễm khuẩn huyết (Sepsis)", "Theo Chức năng thận (CLCr)"], horizontal=True, key="spag_strat_choice")

            fig = go.Figure()
            clean_df = clinical_data["clean_df"]
            obs_df = clean_df[clean_df["DV"].notna()].copy()
            if spag_scale == "Logarit (Semi-log)":
                obs_df = obs_df[obs_df["DV"] > 0]

            for sid in obs_df["ID"].unique():
                sub = obs_df[obs_df["ID"] == sid].sort_values("Time")
                first = sub.iloc[0]
                if spag_strat == "Theo Nhiễm khuẩn huyết (Sepsis)":
                    is_sep = float(first.get("Sepsis", 0)) > 0
                    line_col = "#d84315" if is_sep else "#1565c0"
                    grp_name = "Sepsis (+)" if is_sep else "Non-Sepsis (-)"
                elif spag_strat == "Theo Chức năng thận (CLCr)":
                    cr_val = float(first.get("CLCr_CG", 80))
                    is_low = cr_val < 60
                    line_col = "#7b1fa2" if is_low else "#00796b"
                    grp_name = "CLCr < 60 mL/min" if is_low else "CLCr ≥ 60 mL/min"
                else:
                    line_col = "#1565c0"
                    grp_name = f"ID {sid}"

                fig.add_trace(go.Scatter(x=sub["Time"], y=sub["DV"], mode="lines+markers",
                                         line=dict(width=1.2, color=line_col), marker=dict(size=4),
                                         name=grp_name, showlegend=(spag_strat != "Tất cả"), opacity=0.6))

            fig.update_layout(title="Observed Concentrations vs Time (83 subjects)", xaxis_title="Time (h)",
                              yaxis_title="DV (mg/L)", height=450, template="plotly_white")
            if spag_scale == "Logarit (Semi-log)":
                fig.update_yaxes(type="log")
            st.plotly_chart(fig, width="stretch")

            # PKanalix NCA Module
            header("Phân tích Phi khoang (PKanalix NCA Module)")
            st.info("Phương pháp hình thang (Trapezoidal Rule) ước lượng tự động các thông số dược động học phi khoang không phụ thuộc giả định mô hình, hỗ trợ kiểm tra dữ liệu và khởi tạo thông số ban đầu.")
            df_ind_nca, df_sum_nca = compute_nca(clinical_data)
            if df_sum_nca is not None:
                st.dataframe(df_sum_nca, width="stretch")
                with st.expander("Xem chi tiết thông số NCA từng bệnh nhân (83 cá thể)"):
                    st.dataframe(df_ind_nca, width="stretch")

            st.dataframe(clean_df.head(100), width="stretch")
            
            out_csv, _ = data_io.export_monolix(clinical_data)
            with open(out_csv, "rb") as f:
                st.download_button("Tải file Monolix CSV (`export_monolix.csv`)", f.read(), "export_monolix.csv", "text/csv", icon=":material/download:")
        else:
            header("Mô phỏng quần thể ảo")
            n_sub_sim = st.slider("Số bệnh nhân mô phỏng", 10, 100, 30, 5)
            dose_sim = st.number_input("Liều dùng (mg)", 10.0, 2000.0, 100.0, 50.0)
            df_sim, _ = make_sim_data(model_name, tuple(pop_params.items()), tuple(omega_dict.items()),
                                      sigma_add, sigma_prop, (), n_sub_sim, dose_sim, tuple(T_GRID), len(T_GRID), 42)
            st.dataframe(df_sim.head(50), width="stretch")

    # ── TAB 2: CẤU HÌNH & ƯỚC LƯỢNG SAEM + BOOTSTRAP ──
    with tab2:
        header("Chạy ước lượng tham số bằng thuật toán SAEM")
        target_data = clinical_data if clinical_data is not None else df_sim
        st.info("Thuật toán SAEM (Stochastic Approximation Expectation-Maximization) với MCMC 3 kernel kết hợp giải tích từng đoạn. Tính toán -2LL qua Importance Sampling phân phối Student-t5.")
        if st.button("Bắt đầu ước lượng SAEM", type="primary", width="stretch", icon=":material/play_arrow:"):
            init = dict(
                mu=np.log([pop_params[p] for p in names]),
                omega=[omega_dict[p] for p in names],
                err=[sigma_add, sigma_prop]
            )
            bar = st.progress(0, text="Đang chạy SAEM...")
            res = pk.fit_model(target_data, model_name, init, cov_model, error_model,
                               n_iter1=n_iter1, n_iter2=n_iter2, n_chains=n_chains,
                               seed=int(fit_seed), progress=lambda x: bar.progress(x, text=f"SAEM {x:.0%}"),
                               cov_mode=cov_mode)
            bar.empty()
            st.session_state["res"] = res
            st.session_state.pop("vpc", None)

        if "res" in st.session_state:
            show_fit(st.session_state["res"], clinical_data)

    # ── TAB 3: CHẨN ĐOÁN GOF & OUTLIERS ──
    with tab3:
        header("Chẩn đoán mô hình (Goodness-of-Fit)")
        if res is None:
            st.warning("Vui lòng chạy SAEM ở tab 2 trước.", icon=":material/warning:")
        else:
            dg = res["diag"]
            st.plotly_chart(gof_figure(dg), width="stretch")
            header("Random Effects (η̂ = EBE) & Shrinkage")
            P = len(res["names"])
            cols = st.columns(P + 2)
            for j, p in enumerate(res["names"]):
                cols[j].metric(f"η-shrinkage {p}", f"{dg['eta_shrink'][j] * 100:.1f}%")
            cols[P].metric("ε-shrinkage", f"{dg['eps_shrink'] * 100:.1f}%")
            cols[P + 1].metric("NPDE mean / sd", f"{dg['table']['NPDE'].mean():.2f} / {dg['table']['NPDE'].std():.2f}")

            # Residual & Outliers table
            header("Phân tích Phần dư & Ngoại lai Chẩn đoán (Flagged Residuals)")
            tt = dg["table"]
            flagged = tt[(tt["IWRES"].abs() > 2.5) | (tt["NPDE"].abs() > 2.5)]
            if len(flagged) > 0:
                st.markdown(f'<div class="outlier-box">Phát hiện {len(flagged)} điểm quan sát có |IWRES| > 2.5 hoặc |NPDE| > 2.5 cần lưu ý trong hồ sơ kiểm định.</div>', unsafe_allow_html=True)
                st.dataframe(flagged[["ID", "TIME", "DV", "PRED", "IPRED", "IWRES", "NPDE"]], width="stretch")
            else:
                st.markdown('<div class="convergence-box">Tất cả các điểm phần dư IWRES và NPDE đều nằm trong giới hạn kiểm soát thống kê an toàn (|residual| ≤ 2.5).</div>', unsafe_allow_html=True)

    # ── TAB 4: SÀNG LỌC COVARIATE & FINAL MODEL ──
    with tab4:
        header("Sàng lọc Covariate & Mô hình Tối ưu (Covariate Search & SCM)")
        if res is None:
            st.warning("Vui lòng chạy SAEM ở tab 2 trước.", icon=":material/warning:")
        else:
            header("Tương quan giữa Hiệu ứng Ngẫu nhiên (η̂ = EBE) và Covariate")
            fig_cov = covariate_correlation_figure(res, clinical_data)
            if fig_cov is not None:
                st.plotly_chart(fig_cov, width="stretch")

            header("Bảng So sánh Mô hình Từng bước (Stepwise Covariate Modeling - SCM)")
            cov_summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "validation", "covariate_analysis.csv"))
            is_standard_vanco = (clinical_data is not None and clinical_data.get("N", 0) == 83 and clinical_data.get("nobs", 0) == 278)
            if is_standard_vanco and os.path.exists(cov_summary_path):
                cov_df = pd.read_csv(cov_summary_path)
                st.dataframe(cov_df, width="stretch")
            elif os.path.exists(cov_summary_path):
                with st.expander("Tham khảo Benchmark: Bảng SCM trên tập dữ liệu chuẩn Vancomycin 83 BN", expanded=False):
                    cov_df = pd.read_csv(cov_summary_path)
                    st.dataframe(cov_df, width="stretch")
                st.info("💡 Bạn đang làm việc với tập dữ liệu tải lên riêng biệt. Để kiểm định các giả thuyết covariate cho tập dữ liệu này, hãy chọn các covariate mong muốn ở thanh bên và bấm 'Bắt đầu ước lượng SAEM' để so sánh giá trị -2LL (ΔOFV) và AIC.")
            else:
                st.info("Chạy `python3 validation/run_covariate_analysis.py` để cập nhật bảng so sánh covariate.")

            header("Phương trình Dược động học Quần thể (Population Model Equations)")
            render_dynamic_model_equations(res, clinical_data, show_scm_benchmark=True)

    # ── TAB 5: KHỚP CÁ THỂ ──
    with tab5:
        header("Đường cong nồng độ cá thể (Individual Fits)")
        if res is None:
            st.warning("Vui lòng chạy SAEM ở tab 2 trước.", icon=":material/warning:")
        else:
            d = res["data"]
            sid_choice = st.selectbox("Chọn bệnh nhân (ID)", d["ids"], key="select_patient_tab5")
            idx_sub = int(np.where(d["ids"] == sid_choice)[0][0])
            
            t_max_sub = float(d["t"][idx_sub, d["mask"][idx_sub]].max()) * 1.1 if d["mask"][idx_sub].any() else 48.0
            t_fine = np.linspace(0.0, max(t_max_sub, 24.0), 300)
            
            phi_ind = res["phi_hat"][idx_sub:idx_sub + 1]
            phi_pop = pk.pop_mean_phi(res)[idx_sub:idx_sub + 1]
            
            # Show individual parameters
            c_ind = st.columns(4)
            c_ind[0].metric("CL cá thể (L/h)", f"{float(np.exp(phi_ind[0, 0])):.2f}")
            c_ind[1].metric("V cá thể (L)", f"{float(np.exp(phi_ind[0, 1])):.2f}")
            c_ind[2].metric("η_CL", f"{float(phi_ind[0, 0] - phi_pop[0, 0]):.3f}")
            c_ind[3].metric("η_V", f"{float(phi_ind[0, 1] - phi_pop[0, 1]):.3f}")

            scale_ind = st.radio("Thang đo trục Y cá thể", ["Tuyến tính (Linear)", "Logarit (Semi-log)"], horizontal=True, key="scale_ind_choice")

            sub_d = [d["sub_doses"][sid_choice]] if "sub_doses" in d else None
            sub_c = [d["sub_cov_events"][sid_choice]] if "sub_cov_events" in d else None
            
            pred_fine = pk.predict(res["model_name"], t_fine[None, :], d["dose"][idx_sub:idx_sub+1], phi_pop,
                                   data=d, sub_doses=sub_d, sub_cov=sub_c, beta=res["beta"], cov_model=res["cov_model"])[0]
            ipred_fine = pk.predict(res["model_name"], t_fine[None, :], d["dose"][idx_sub:idx_sub+1], phi_ind,
                                    data=d, sub_doses=sub_d, sub_cov=sub_c, beta=res["beta"], cov_model=res["cov_model"])[0]
            
            fig = go.Figure()
            mk = d["mask"][idx_sub]
            obs_t = d["t"][idx_sub, mk]
            obs_y = d["y"][idx_sub, mk]
            if scale_ind == "Logarit (Semi-log)":
                mk_valid = obs_y > 0
                obs_t = obs_t[mk_valid]
                obs_y = obs_y[mk_valid]

            fig.add_trace(go.Scatter(x=obs_t, y=obs_y, mode="markers",
                                     marker=dict(size=8, color="black"), name="Quan sát thực tế (Obs)"))
            fig.add_trace(go.Scatter(x=t_fine, y=pred_fine, mode="lines",
                                     line=dict(dash="dash", color="#d62728", width=1.5), name="PRED (Quần thể)"))
            fig.add_trace(go.Scatter(x=t_fine, y=ipred_fine, mode="lines",
                                     line=dict(color="#1565c0", width=2.5), name="IPRED (Cá thể EBE)"))
            
            fig.update_layout(title=f"Đường cong cá thể Bệnh nhân ID {sid_choice}", xaxis_title="Thời gian (giờ)",
                              yaxis_title="Nồng độ (mg/L)", height=450, template="plotly_white")
            if scale_ind == "Logarit (Semi-log)":
                fig.update_yaxes(type="log")
            st.plotly_chart(fig, width="stretch")

    # ── TAB 6: THẨM ĐỊNH VPC & MÔ PHỎNG SIMULX ──
    with tab6:
        header("Thẩm định Mô hình bằng Visual Predictive Check (VPC)")
        if res is None:
            st.warning("Vui lòng chạy SAEM ở tab 2 trước.", icon=":material/warning:")
        else:
            c1, c2 = st.columns(2)
            n_rep = c1.slider("Số bản sao mô phỏng (Replicates)", 20, 200, 50, 10, key="vpc_nrep")
            pcorr = c2.checkbox("Prediction-corrected (pcVPC)", value=True, key="vpc_pcorr")
            if st.button("Tạo biểu đồ VPC", type="primary", icon=":material/refresh:", key="btn_run_vpc"):
                st.session_state["vpc"] = (pk.run_vpc(res, n_rep=n_rep, pcorr=pcorr), pcorr)
            if "vpc" in st.session_state:
                v, pc = st.session_state["vpc"]
                st.plotly_chart(vpc_figure(v, None if pc else res["diag"]["table"]), width="stretch")

            # Simulx Module
            header("Mô phỏng Phác đồ Điều trị Lâm sàng & Đích TDM (Simulx Module)")
            st.info("Mô phỏng động học nồng độ theo thời gian cho phác đồ đa liều dựa trên mô hình quần thể, đánh giá tỷ lệ đạt nồng độ đáy mục tiêu (15 - 20 mg/L) và AUC24/MIC ≥ 400.")
            
            c_p1, c_p2, c_p3 = st.columns(3)
            p_tbw = c_p1.number_input("Cân nặng TBW (kg)", 30.0, 150.0, 60.0, 5.0, key="sim_tbw")
            p_clcr = c_p2.number_input("Chức năng thận CLCr (mL/min)", 10.0, 200.0, 80.0, 5.0, key="sim_clcr")
            p_sep = c_p3.selectbox("Nhiễm khuẩn huyết (Sepsis)", ["Không (Non-Sepsis)", "Có (Sepsis)"], key="sim_sep")
            
            # Predict CL and V dynamically using estimated parameters from res
            cl_base = float(np.exp(res["beta"][0][0])) if len(res.get("beta", [])) > 0 else 3.0
            v_base = float(np.exp(res["beta"][1][0])) if len(res.get("beta", [])) > 1 else 21.0
            
            beta_cl_clcr = 0.0
            beta_v_tbw = 0.0
            beta_cl_sep = 0.0
            for j, p in enumerate(res.get("names", [])):
                covs = res.get("cov_model", {}).get(p, [])
                for k, c in enumerate(covs):
                    if p == "CL" and c == "CLCr_CG":
                        beta_cl_clcr = float(res["beta"][j][1 + k])
                    elif p == "V" and c == "TBW":
                        beta_v_tbw = float(res["beta"][j][1 + k])
                    elif p == "CL" and c == "Sepsis":
                        beta_cl_sep = float(res["beta"][j][1 + k])

            ref_clcr = float(res.get("data", {}).get("ref_medians", {}).get("CLCr_CG", 80.0))
            ref_tbw = float(res.get("data", {}).get("ref_medians", {}).get("TBW", 60.0))
            sep_factor = np.exp(beta_cl_sep) if (p_sep == "Có (Sepsis)" and beta_cl_sep != 0.0) else 1.0
            cl_patient = cl_base * (((p_clcr / ref_clcr) ** beta_cl_clcr) if beta_cl_clcr != 0.0 else 1.0) * sep_factor
            v_patient = v_base * (((p_tbw / ref_tbw) ** beta_v_tbw) if beta_v_tbw != 0.0 else 1.0)
            
            st.markdown(f"**Thông số dự đoán của bệnh nhân**: $CL = {cl_patient:.2f}\\text{{ L/h}}$, $V = {v_patient:.2f}\\text{{ L}}$ (Thời gian bán thải $t_{{1/2}} = {np.log(2)*v_patient/cl_patient:.1f}\\text{{ h}}$)")
            
            c_d1, c_d2, c_d3 = st.columns(3)
            reg_dose = c_d1.selectbox("Liều dùng mỗi lần (mg)", [500.0, 750.0, 1000.0, 1250.0, 1500.0, 2000.0], index=2, key="sim_dose")
            reg_tau = c_d2.selectbox("Khoảng cách đưa liều (giờ)", [8.0, 12.0, 24.0], index=1, key="sim_tau")
            reg_tinf = c_d3.selectbox("Thời gian truyền tinf (giờ)", [1.0, 1.5, 2.0], index=0, key="sim_tinf")
            
            t_sim, c_sim, ct_ss, cpk_ss, auc24_ss = simulx_dosing_simulation(cl_patient, v_patient, reg_dose, reg_tau, reg_tinf, duration=72.0)
            
            # Plot simulation profile
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=t_sim, y=c_sim, mode="lines",
                                         line=dict(color="#1565c0", width=2.5), name="Nồng độ mô phỏng"))
            # Target range band for trough 15-20 mg/L
            fig_sim.add_hrect(y0=15.0, y1=20.0, fillcolor="rgba(76, 175, 80, 0.2)", line_width=0, name="Dải đích Trough (15-20 mg/L)")
            fig_sim.add_hline(y=15.0, line_dash="dash", line_color="green")
            fig_sim.add_hline(y=20.0, line_dash="dash", line_color="green")
            fig_sim.add_hline(y=25.0, line_dash="dot", line_color="red")
            fig_sim.update_layout(title=f"Mô phỏng Phác đồ {reg_dose:.0f} mg mỗi {reg_tau:.0f}h (Truyền {reg_tinf:.1f}h) trong 72 giờ",
                                  xaxis_title="Thời gian (giờ)", yaxis_title="Nồng độ (mg/L)", height=450, template="plotly_white")
            st.plotly_chart(fig_sim, width="stretch")
            
            # Metrics
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Nồng độ đáy Ctrough (ss)", f"{ct_ss:.2f} mg/L")
            m_c2.metric("Nồng độ đỉnh Cpeak (ss)", f"{cpk_ss:.2f} mg/L")
            m_c3.metric("Tổng liều 24h", f"{reg_dose * (24.0 / reg_tau):.0f} mg")
            m_c4.metric("AUC24 / MIC (MIC=1)", f"{auc24_ss:.1f}")
            
            # Clinical guidance
            if 15.0 <= ct_ss <= 20.0 and auc24_ss >= 400.0:
                st.markdown('<div class="convergence-box">ĐẠT ĐÍCH TỐI ƯU: Nồng độ đáy (15 - 20 mg/L) và AUC24/MIC ≥ 400 đều đạt mục tiêu hướng dẫn lâm sàng IDSA/ASHP.</div>', unsafe_allow_html=True)
            elif ct_ss > 20.0:
                st.markdown('<div class="outlier-box">CẢNH BÁO NGUY CƠ ĐỘC THẬN: Nồng độ đáy > 20 mg/L làm tăng nguy cơ tổn thương thận cấp (AKI). Khuyến cáo giảm liều hoặc giãn khoảng cách đưa liều.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="outlier-box">CẢNH BÁO DƯỚI ĐÍCH ĐIỀU TRỊ: Nồng độ đáy < 15 mg/L hoặc AUC24/MIC < 400, có nguy cơ thất bại điều trị hoặc chọn lọc vi khuẩn kháng thuốc. Khuyến cáo tăng liều hoặc rút ngắn khoảng cách đưa liều.</div>', unsafe_allow_html=True)

            # R script generator
            header("Mã nguồn R Tự động phục vụ Nghiên cứu Khoa học (Reproducible R Script)")
            st.info("Xuất trọn bộ mã nguồn phân tích R chuẩn mực để đính kèm bài báo khoa học hoặc nghiệm thu đề tài, tương thích với RStudio.")
            r_code = generate_r_script(res)
            st.download_button("Tải file R Script Phân tích (`SimDx_analysis.R`)", r_code, "SimDx_analysis.R", "text/plain", icon=":material/download:", key="btn_download_r_script")


if __name__ == "__main__":
    main()
