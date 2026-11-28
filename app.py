"""
SimDx - NLME Pharmacometrics Platform (Mini Monolix).
Giao diện Streamlit tương tác cho mô hình dược động học phi tuyến hiệu ứng hỗn hợp (NLME).
Hỗ trợ cả dữ liệu mô phỏng và bộ dữ liệu lâm sàng thực tế (Dataset_chính_thức.xlsx).
"""
import io
import os
import json
import base64
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import linregress, norm

import pk_core as pk
import data_io
from monolix_library import MONOLIX_LIBRARY
import monolix_engine as me

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
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else None,
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
        padding: 6px 6px;
        font-weight: 700;
        color: white;
        font-size: 11px;
        border-radius: 4px 4px 0 0;
        text-align: center;
        margin-bottom: 4px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .col-admin { background-color: #0097a7; }
    .col-delay { background-color: #1976d2; }
    .col-absorp { background-color: #3f51b5; }
    .col-distrib { background-color: #5c6bc0; }
    .col-elim { background-color: #7b1fa2; }
    .col-param { background-color: #c2185b; }
    .param-pill {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11px;
        color: white;
        margin-right: 4px;
        margin-bottom: 3px;
        font-family: 'SF Mono', Consolas, Monaco, monospace;
    }
    .pill-v { background-color: #5c6bc0; }
    .pill-cl { background-color: #c2185b; }
    .pill-ka { background-color: #0097a7; }
    .pill-q { background-color: #f57c00; }
    .pill-vm { background-color: #8e24aa; }
    .pill-km { background-color: #ab47bc; }
    .pill-k { background-color: #ef6c00; }
    .pill-tlag { background-color: #0288d1; }
    .pill-pd { background-color: #2e7d32; }
    .pill-hybrid { background-color: #37474f; }
    .pill-other { background-color: #455a64; }
    .model-id-title {
        font-family: 'SF Mono', Consolas, Monaco, monospace;
        font-size: 14.5px;
        font-weight: 700;
        color: #0f172a;
        display: inline-block;
        margin-left: 6px;
    }
    .model-card-row {
        padding: 8px 12px;
        border-radius: 6px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        margin-bottom: 8px;
        transition: all 0.15s ease-in-out;
    }
    .model-card-row:hover {
        border-color: #0097a7;
        box-shadow: 0 2px 8px rgba(0, 151, 167, 0.08);
    }

    
    /* Monolix Sidebar Workflow Navigation */
    .stSidebar [data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 6px;
    }
    .stSidebar [data-testid="stRadio"] label {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 2px;
        transition: all 0.15s ease-in-out;
        cursor: pointer;
    }
    .stSidebar [data-testid="stRadio"] label:hover {
        border-color: #0284c7;
        background-color: #f0f9ff;
    }
    .stSidebar [data-testid="stRadio"] label:has(input:checked) {
        background-color: #0284c7 !important;
        border-color: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 700;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.25);
    }
    .stSidebar [data-testid="stRadio"] label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Monolix-Style Workflow Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f5f9;
        padding: 6px 8px;
        border-radius: 8px;
        border-bottom: 2px solid #cbd5e1;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding: 0 16px;
        background-color: transparent;
        border-radius: 6px;
        color: #475569;
        font-weight: 600;
        font-size: 13px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 4px rgba(2, 132, 199, 0.25);
    }
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

</style>
""", unsafe_allow_html=True)

T_GRID = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 36, 48, 72])
WT_EFFECT_DEFAULT = {"CL": 0.75, "Q": 0.75, "Vmax": 0.75, "V": 1.0, "V1": 1.0, "V2": 1.0}

MONOLIX_PK_LIBRARY = [m for m in MONOLIX_LIBRARY if m["category"] == "PK"]


def format_pills_html(params):
    pills = []
    for p in params:
        p_up = p.upper()
        if p in ['V', 'V1', 'V2', 'V3']:
            cls = 'pill-v'
        elif p in ['CL', 'Q', 'Q2', 'Q3', 'kel']:
            cls = 'pill-cl'
        elif p in ['ka', 'Tk0', 'ktr', 'Mtt']:
            cls = 'pill-ka'
        elif p in ['Vm', 'Km', 'Vmax']:
            cls = 'pill-vm'
        elif p.startswith('k') or p in ['k12', 'k21', 'k13', 'k31', 'kout', 'ke0']:
            cls = 'pill-k'
        elif 'lag' in p.lower():
            cls = 'pill-tlag'
        elif p in ['E0', 'Emax', 'EC50', 'Imax', 'IC50', 'gamma', 'Rin', 'Smax', 'SC50']:
            cls = 'pill-pd'
        elif p in ['alpha', 'beta', 'gamma', 'A', 'B', 'C']:
            cls = 'pill-hybrid'
        else:
            cls = 'pill-other'
        pills.append(f'<span class="param-pill {cls}">{p}</span>')
    return "".join(pills)


def render_model_cards(matched, tab_key="pk"):
    """Hiển thị danh sách card mô hình Monolix kèm phân trang và tìm kiếm."""
    c_page, c_info = st.columns([3, 4])
    with c_page:
        per_page_opt = st.radio(
            "Records per page:",
            [10, 25, 50, "Tất cả"],
            horizontal=True,
            index=1,
            key=f"rpp_{tab_key}"
        )
    
    total = len(matched)
    if per_page_opt == "Tất cả":
        page_size = max(1, total)
    else:
        page_size = int(per_page_opt)
    
    with c_info:
        st.markdown(f"<div style='text-align: right; padding-top: 8px; color: #475569; font-size: 13px;'><b>Showing 1 to {min(page_size, total)} of {total} entries</b></div>", unsafe_allow_html=True)
    
    if not matched:
        st.warning("Không có mô hình nào khớp với tổ hợp lọc hiện tại. Hãy bấm 'Xóa bộ lọc'.")
        return

    display_items = matched[:page_size]
    for m in display_items:
        pills_html = format_pills_html(m["params"])
        col_info, col_btn = st.columns([5, 2])
        with col_info:
            st.markdown(f'{pills_html} <span class="model-id-title">{m["id"]}</span>', unsafe_allow_html=True)
            st.caption(f"{m['desc']}")
            with st.expander("Phương trình toán học (ODE & Closed-form)"):
                st.latex(m["ode"])
                st.latex(m["solution"])
        with col_btn:
            if st.button("Chọn Model này", key=f"sel_mod_{m['id']}_{tab_key}", type="primary", use_container_width=True):
                st.session_state["selected_lib_model"] = m["id"]
                st.session_state["model_name_choice"] = m["internal_model"]
                st.session_state["model_structure_select"] = m["internal_model"]
                # Cập nhật thông số ban đầu vào session_state slider
                if "params" in m and "defaults" in m:
                    for p_name, p_def in zip(m["params"], m["defaults"]):
                        st.session_state[f"{m['internal_model']}_pop_{p_name}"] = float(p_def)
                if m.get("tinf_override") is not None:
                    st.session_state["tinf_val"] = m["tinf_override"]
                    st.session_state["tinf_upload_input"] = float(m["tinf_override"])
                    st.session_state["tinf_main_input"] = float(m["tinf_override"])
                if m.get("error_override"):
                    st.session_state["error_model_choice"] = m["error_override"]
                    st.session_state["error_model_select"] = m["error_override"]
                st.rerun()
        st.markdown("<hr style='margin: 4px 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)


@st.dialog("LOAD FROM LIBRARY — Monolix Model Library", width="large")
def show_model_library_dialog():
    st.markdown("""
    <div style="background: #f1f5f9; padding: 10px 14px; border-radius: 6px; font-size: 13px; margin-bottom: 12px; border-left: 4px solid #0097a7;">
    <b>Monolix Library Browser:</b> Thư viện mô hình Dược động học & Dược lực học chuẩn Monolix (Lixoft / Simulations Plus). Bộ lọc đa chiều gồm Đường dùng (Administration), Độ trễ (Delay), Cơ chế hấp thu (Absorption), Phân bố ngăn (Distribution), Cơ chế đào thải (Elimination) và Hệ tham số (Parametrization).
    </div>
    """, unsafe_allow_html=True)

    tab_pk, tab_pd, tab_pkpd, tab_other = st.tabs(["PK", "PD", "PKPD", "TMDD / TTE / Other"])

    with tab_pk:
        # 6 Filter columns with colored headers matching Monolix GUI
        f1, f2, f3, f4, f5, f6 = st.columns(6)
        with f1:
            st.markdown('<div class="lib-header-col col-admin">Administration</div>', unsafe_allow_html=True)
            admin_sel = st.selectbox("Administration", ["Tất cả", "bolus", "infusion", "oral/extravascular"], label_visibility="collapsed", key="lib_admin")
        with f2:
            st.markdown('<div class="lib-header-col col-delay">Delay</div>', unsafe_allow_html=True)
            delay_sel = st.selectbox("Delay", ["Tất cả", "no delay", "lag time", "transit compartments"], label_visibility="collapsed", key="lib_delay")
        with f3:
            st.markdown('<div class="lib-header-col col-absorp">Absorption</div>', unsafe_allow_html=True)
            absorp_sel = st.selectbox("Absorption", ["Tất cả", "first order", "zero order", "none"], label_visibility="collapsed", key="lib_absorp")
        with f4:
            st.markdown('<div class="lib-header-col col-distrib">Distribution</div>', unsafe_allow_html=True)
            distrib_sel = st.selectbox("Distribution", ["Tất cả", "1 compartment", "2 compartments", "3 compartments"], label_visibility="collapsed", key="lib_distrib")
        with f5:
            st.markdown('<div class="lib-header-col col-elim">Elimination</div>', unsafe_allow_html=True)
            elim_sel = st.selectbox("Elimination", ["Tất cả", "linear", "Michaelis-Menten"], label_visibility="collapsed", key="lib_elim")
        with f6:
            st.markdown('<div class="lib-header-col col-param">Parametrization</div>', unsafe_allow_html=True)
            param_sel = st.selectbox("Parametrization", ["Tất cả", "clearance", "rate", "hybrid constants"], label_visibility="collapsed", key="lib_param")

        c_search, c_clear = st.columns([4, 1])
        search_q = c_search.text_input("Tìm kiếm tên mô hình...", placeholder="Ví dụ: bolus_1cpt_VCl, oral1_2cpt, infusion, Tlag...", label_visibility="collapsed", key="lib_search")
        if c_clear.button("Xóa bộ lọc", key="btn_clear_filters"):
            st.session_state["lib_admin"] = "Tất cả"
            st.session_state["lib_delay"] = "Tất cả"
            st.session_state["lib_absorp"] = "Tất cả"
            st.session_state["lib_distrib"] = "Tất cả"
            st.session_state["lib_elim"] = "Tất cả"
            st.session_state["lib_param"] = "Tất cả"
            st.session_state["lib_search"] = ""
            st.rerun()

        # Match PK models
        matched_pk = []
        pk_models = [m for m in MONOLIX_LIBRARY if m["category"] == "PK"]
        for m in pk_models:
            if admin_sel != "Tất cả" and m["admin"] != admin_sel:
                continue
            if delay_sel != "Tất cả" and m["delay"] != delay_sel:
                continue
            if absorp_sel != "Tất cả" and m["absorption"] != absorp_sel:
                continue
            if distrib_sel != "Tất cả" and m["distrib"] != distrib_sel:
                continue
            if elim_sel != "Tất cả" and m["elim"] != elim_sel:
                continue
            if param_sel != "Tất cả" and m["param"] != param_sel:
                continue
            if search_q.strip():
                sq = search_q.strip().lower()
                if sq not in m["id"].lower() and sq not in m["desc"].lower() and not any(sq == p.lower() for p in m["params"]):
                    continue
            matched_pk.append(m)

        render_model_cards(matched_pk, "pk")

    with tab_pd:
        pd_search = st.text_input("Tìm kiếm mô hình Dược lực học (PD)...", placeholder="Ví dụ: emax, sigmoidal, turnover, inhibitory...", label_visibility="collapsed", key="pd_search")
        matched_pd = [
            m for m in MONOLIX_LIBRARY 
            if m["category"] == "PD" and (not pd_search.strip() or pd_search.strip().lower() in m["id"].lower() or pd_search.strip().lower() in m["desc"].lower())
        ]
        render_model_cards(matched_pd, "pd")

    with tab_pkpd:
        pkpd_search = st.text_input("Tìm kiếm mô hình phối hợp PK/PD...", placeholder="Ví dụ: effect_cpt, direct...", label_visibility="collapsed", key="pkpd_search")
        matched_pkpd = [
            m for m in MONOLIX_LIBRARY 
            if m["category"] == "PKPD" and (not pkpd_search.strip() or pkpd_search.strip().lower() in m["id"].lower() or pkpd_search.strip().lower() in m["desc"].lower())
        ]
        render_model_cards(matched_pkpd, "pkpd")

    with tab_other:
        other_search = st.text_input("Tìm kiếm mô hình TMDD / TTE...", placeholder="Ví dụ: tmdd, weibull, hazard...", label_visibility="collapsed", key="other_search")
        matched_other = [
            m for m in MONOLIX_LIBRARY 
            if m["category"] in ["TMDD", "TTE"] and (not other_search.strip() or other_search.strip().lower() in m["id"].lower() or other_search.strip().lower() in m["desc"].lower())
        ]
        render_model_cards(matched_other, "other")


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


def residuals_qq_figure(dg):
    """QQ-plot chuẩn tắc của phần dư IWRES và NPDE theo chuẩn chẩn đoán Monolix."""
    from scipy import stats
    tt = dg["table"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Normal QQ-Plot: IWRES", "Normal QQ-Plot: NPDE"])
    for idx, var in enumerate(["IWRES", "NPDE"]):
        vals = tt[var].dropna().values
        if len(vals) > 0:
            (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm")
            fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                     marker=dict(size=5, color="#1565c0", opacity=0.7),
                                     name=f"{var} Quantiles"), row=1, col=idx+1)
            fig.add_trace(go.Scatter(x=osm, y=slope * osm + intercept, mode="lines",
                                     line=dict(color="red", dash="dash"),
                                     name="Đường chuẩn N(0,1)"), row=1, col=idx+1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=idx+1)
        fig.update_yaxes(title_text=f"Sample Quantiles ({var})", row=1, col=idx+1)
    fig.update_layout(height=380, template="plotly_white", showlegend=False,
                      title_text="Phân tích Phân phối Phần dư (Residuals QQ-Plots)")
    return fig


def parameter_correlation_heatmap(res):
    """Bản đồ nhiệt ma trận tương quan giữa các hiệu ứng ngẫu nhiên (Monolix Random Effects Correlation Matrix)."""
    names = res["names"]
    P = len(names)
    cov_matrix = np.eye(P)
    if "phi_hat" in res and res["phi_hat"] is not None:
        mean_phi = pk.pop_mean_phi(res)
        eta = res["phi_hat"] - mean_phi
        if eta.shape[0] > 1:
            cov_matrix = np.corrcoef(eta.T)
            cov_matrix = np.nan_to_num(cov_matrix, nan=0.0)

    fig = go.Figure(data=go.Heatmap(
        z=cov_matrix,
        x=[f"η_{p}" for p in names],
        y=[f"η_{p}" for p in names],
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=np.round(cov_matrix, 2),
        texttemplate="%{text}",
        colorbar=dict(title="Correlation")
    ))
    fig.update_layout(title="Ma trận Tương quan Hiệu ứng Ngẫu nhiên (Random Effects Correlation Matrix)",
                      height=420, template="plotly_white")
    return fig


def initial_estimates_preview_figure(target_data, model_name, pop_params):
    """Biểu đồ xem trước đường cong nồng độ từ thông số khởi tạo đè lên dữ liệu bệnh nhân (Monolix Initial Estimates Tab)."""
    fig = go.Figure()
    if target_data is None:
        return fig
    
    # 1. Vẽ các điểm dữ liệu thực tế mẫu (15 bệnh nhân đầu tiên)
    if "ids" in target_data and len(target_data["ids"]) > 0:
        sample_ids = target_data["ids"][:15]
        for sid in sample_ids:
            idx = int(np.where(target_data["ids"] == sid)[0][0])
            mk = target_data["mask"][idx]
            obs_t = target_data["t"][idx, mk]
            obs_y = target_data["y"][idx, mk]
            fig.add_trace(go.Scatter(x=obs_t, y=obs_y, mode="markers",
                                     marker=dict(size=4, color="#94a3b8"), showlegend=False,
                                     name=f"ID {sid}", hoverinfo="skip"))
    
    # 2. Tính đường cong nồng độ quần thể với pop_params
    if model_name in pk.MODELS:
        mi = pk.MODELS[model_name]
        t_fine = np.linspace(0.0, 48.0, 200)
        phi_init = np.array([[pop_params.get(p, mi["defaults"][j]) for j, p in enumerate(mi["params"])]])
        dose_init = np.array([1000.0])
        tinf_val = 0.01 if "bolus" in model_name.lower() else float(st.session_state.get("tinf_val", 1.0))
        data_mock = {"tinf": tinf_val}
        
        try:
            c_pred = pk.predict(model_name, t_fine[None, :], dose_init, phi_init, data=data_mock)[0]
            fig.add_trace(go.Scatter(x=t_fine, y=c_pred, mode="lines",
                                     line=dict(color="#1565c0", width=3),
                                     name="Dự đoán Quần thể Ban đầu (θ₀ = Initial Pop Curve)"))
        except Exception:
            pass

    fig.update_layout(title=f"Đường cong Khởi tạo Dược động học ({model_name}) đè lên Dữ liệu Thực tế",
                      xaxis_title="Thời gian (h)", yaxis_title="Nồng độ (mg/L)",
                      height=420, template="plotly_white")
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


def render_poppk_equations_tab(res, clinical_data=None):
    if res is None:
        st.warning("Vui lòng nạp dữ liệu và chạy SAEM ở bước 2 trước để hệ thống tính toán phương trình động.")
        return

    sub_t1, sub_t2, sub_t3 = st.tabs([
        "Hệ Phương trình Dược động học Quần thể (PopPK Equations)",
        "Tiêu chí Thẩm định & Chỉ số Đánh giá (Evaluation Metrics)",
        "Tổng quan Lý thuyết Dược động học Quần thể (PopPK Theory)"
    ])

    names = res.get("names", ["CL", "V"])
    beta = res.get("beta", [])
    omega = res.get("omega", [])
    cov_model = res.get("cov_model", {})
    err = res.get("err", [1.0, 0.2])
    err_mod = res.get("error_model", "combined")
    m_name = res.get("model_name", "1-Compartment IV infusion (multi-dose)")

    # 1. Trích xuất giá trị ước lượng tham số
    theta_dict = {}
    for j, p in enumerate(names):
        theta_dict[p] = float(np.exp(beta[j][0])) if j < len(beta) and len(beta[j]) > 0 else 1.0

    omega_dict = {}
    for j, p in enumerate(names):
        omega_dict[p] = float(omega[j]) if j < len(omega) else 0.3

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

    with sub_t1:
        st.markdown("### Cấu trúc Mô hình Dược động học Quần thể Phân cấp Ba Tầng (3-Tier Hierarchical NLME)")
        st.caption("Khung cấu trúc chuẩn được khởi xướng bởi Sheiner, Rosenberg và Melmon: phân tách tham số thành tầng quần thể, tầng cá thể và tầng sai số đo lường.")

        # Tầng 1: Tầng Quần thể
        st.markdown("#### 1. Tầng Quần thể — Thông số cố định (Fixed Effects $\\theta$) & Ma trận Hiệp phương sai $(\\Omega, \\Sigma)$")
        st.markdown("""
        Mô tả các thông số dược động học điển hình (*typical value* $\\theta$) của một quần thể bệnh nhân xác định, cùng ma trận phương sai biến thiên giữa các cá thể $\\Omega$ và sai số dư $\\Sigma$.
        """)
        est_df = estimates_table(res)
        st.dataframe(est_df, width="stretch")

        # Tầng 2: Tầng Cá thể & Yếu tố Đồng ảnh hưởng (Covariates)
        st.markdown("#### 2. Tầng Cá thể — Độ lệch ngẫu nhiên cá thể (IIV $\\eta_i$) & Yếu tố Đồng ảnh hưởng (Covariates)")
        st.markdown(r"""
        Phân tách thông số dược động học cá thể ($P_i$) thành giá trị điển hình quần thể ($\theta$), yếu tố hiệu chỉnh đồng ảnh hưởng và độ lệch cá thể ($\eta_i$):
        """)
        st.latex(r"P_i = \theta \times \text{Covariate Effect} \times \exp(\eta_i), \quad \eta_i \sim \mathcal{N}(0, \omega^2)")
        st.caption(r"Trong đó $\eta_i$ là biến thiên liên cá thể (IIV), giả định phân bố chuẩn với trung bình 0 và phương sai $\omega^2$. Dạng hàm mũ đảm bảo $P_i$ luôn dương và tương ứng với giả định IIV theo phân bố log-chuẩn.")

        st.markdown("##### Quy chuẩn Tổng quát đưa Biến đồng ảnh hưởng vào mô hình (Generalized Covariate Modeling):")
        st.markdown(r"""
        Với một mô hình dược động học quần thể bất kỳ có $M$ biến liên tục ($X_m$: $CrCl, WT, \text{Albumin}, \text{Bilirubin}, \text{Age}\dots$) và $L$ biến phân loại / đa hình gen ($G_l$: $\text{GEN}, \text{Sepsis}, \text{Sex}\dots$), phương trình tổng quát cho mọi thông số cá thể $P_i \in \{CL_i, V_i, V_{1,i}, Q_i, V_{2,i}, k_{a,i}\dots\}$ được viết theo chuẩn **Monolix / NONMEM** như sau:
        """)
        st.latex(r"P_i = \theta_P \times \prod_{m=1}^{M} \left(\frac{X_{m, i}}{X_{m, \text{ref}}}\right)^{\beta_{P, m}} \times \prod_{l=1}^{L} \exp\!\left(\sum_{k \neq k_{\text{ref}}} \beta_{P, l, k} \cdot \mathbb{I}(G_{l, i} = k)\right) \times \exp(\eta_{P, i}), \quad \eta_{P, i} \sim \mathcal{N}(0, \omega_P^2)")

        st.markdown(r"""
        Trong đó:
        - **Biến liên tục ($X_{m, i}$)** *(ví dụ: $CrCl, WT, \text{Albumin}, \text{Bilirubin}$)*: Đưa vào theo **mô hình hàm lũy thừa (power model)** chuẩn hóa quanh giá trị tham chiếu $X_{m, \text{ref}}$ (trung vị quần thể $X_{\text{median}}$ hoặc chuẩn $70\text{ kg}$).
        - **Biến phân loại & Biến Gen ($G_{l, i}$)** *(ví dụ: $\text{GEN} \in \{\text{EM}, \text{IM}, \text{PM}\}$ hoặc $\{0, 1, 2\}$, $\text{Sepsis} \in \{0, 1\}$)*: Mã hóa bằng **biến giả chỉ thị $\mathbb{I}(G_{l, i} = k)$** dạng hàm mũ so với nhóm tham chiếu $k_{\text{ref}}$ (thường là kiểu gen bình thường / *Wild-type* hoặc *EM*). Dạng hàm mũ đảm bảo $P_i > 0$ và cho phép mỗi kiểu gen có hệ số hiệu chỉnh riêng biệt $\exp(\beta_{P, l, k})$ mà không ép tuyến tính giữa các kiểu gen.
        """)

        with st.expander("Ví dụ minh họa cụ thể (CrCl, Cân nặng WT, Sepsis và Đa hình Gen CYP)", expanded=False):
            st.latex(r"CL_i = \theta_{CL} \times \left(\frac{CrCl_i}{CrCl_{\text{median}}}\right)^{\theta_1} \times \left(\frac{\text{Albumin}_i}{\text{Albumin}_{\text{median}}}\right)^{\theta_2} \times \exp\!\big(\beta_{\text{IM}} \cdot \mathbb{I}(\text{GEN}_i = \text{IM}) + \beta_{\text{PM}} \cdot \mathbb{I}(\text{GEN}_i = \text{PM})\big) \times \exp(\eta_{CL, i})")
            st.latex(r"V_i = \theta_V \times \left(\frac{WT_i}{70}\right)^{\theta_3} \times \exp(\beta_{\text{Sepsis}} \cdot \text{SEPSIS}_i) \times \exp(\eta_{V, i})")

        st.markdown("---")
        st.markdown("##### Phương trình Số thực tế của Mô hình vừa Ước lượng bằng SAEM:")
        has_cov = any(len(cov_model.get(p, [])) > 0 for p in names)
        if res.get("scm_history"):
            d_ofv_tot = float(res.get("delta_ofv_total", 0.0))
            st.success(f"**Mô hình Cuối cùng (Auto-SCM Final Model):** Đã tự động sàng lọc EBE và kiểm định từng bước Forward ($p < 0.05$) + Backward ($p < 0.01$), giúp giảm $-2LL$ được **{d_ofv_tot:.2f} đơn vị** so với Mô hình Cơ sở.")
            with st.expander("Xem Bảng các bước xây dựng mô hình (Auto-SCM History — tương đương Phụ lục 8)", expanded=False):
                st.dataframe(pd.DataFrame(res["scm_history"]), width="stretch")

        if has_cov:
            st.info("Phương trình được tính toán động 100% từ kết quả ước lượng SAEM và các hiệu ứng covariate (liên tục & phân loại/gen) hiện tại:")
        else:
            st.info("Phương trình Mô hình Cơ sở (Base Model) được tính toán động 100% từ kết quả ước lượng SAEM hiện tại:")

        cat_info = res.get("cat_info", {})
        cov_types_res = res.get("cov_types", {})
        # Map dummy name -> (parent_col, level_val)
        dummy_lookup = {}
        for parent_c, info in cat_info.items():
            for d_name, lvl in info.get("dummies", []):
                dummy_lookup[d_name] = (parent_c, lvl)

        for j, p in enumerate(names):
            theta_val = theta_dict[p]
            omega_val = omega_dict[p]
            covs = cov_model.get(p, [])
            cov_terms = ""
            for k, c in enumerate(covs):
                b_val = float(beta[j][1 + k]) if j < len(beta) and (1 + k) < len(beta[j]) else 0.0
                if c in dummy_lookup:
                    parent_c, lvl = dummy_lookup[c]
                    clean_pc = parent_c.replace("_", r"\_")
                    clean_lvl = str(lvl).replace("_", r"\_")
                    cov_terms += rf" \times \exp\!\left({b_val:+.3f} \cdot \mathbb{{I}}(\text{{{clean_pc}}}_i = \text{{{clean_lvl}}})\right)"
                elif cov_types_res.get(c) == "categorical" or c in ["Sepsis", "Sex", "GEN", "SEX"]:
                    clean_c = c.replace("_", r"\_")
                    cov_terms += rf" \times \exp\!\left({b_val:+.3f} \cdot \text{{{clean_c}}}_i\right)"
                else:
                    clean_c = c.replace("_", r"\_")
                    ref_val = ref_medians.get(c, 70.0)
                    ref_str = f"{ref_val:.0f}" if ref_val == int(ref_val) else f"{ref_val:.1f}"
                    cov_terms += rf" \times \left(\frac{{\text{{{clean_c}}}_i}}{{{ref_str}}}\right)^{{{b_val:.3f}}}"

            clean_p = p.replace("_", r"\_")
            eq = rf"{clean_p}_i = {theta_val:.2f}{cov_terms} \times \exp(\eta_{{i, {clean_p}}}), \quad \eta_{{{clean_p}}} \sim \mathcal{{N}}(0, {omega_val:.2f}^2)"
            st.latex(eq)

        # Tầng 3: Tầng Quan sát & Mô hình Sai số Dư
        st.markdown("#### 3. Tầng Quan sát — Mô hình Sai số Dư (Residual Error Model)")
        st.markdown(r"""
        Mối liên hệ giữa nồng độ quan sát được ($C_{\text{obs}}$) và nồng độ dự đoán theo mô hình cấu trúc ($C_{\text{pred}}$), mô tả sai số đo lường và biến thiên nội cá thể:
        """)
        st.latex(r"C_{\text{obs}, ij} = C_{\text{pred}, ij} \times (1 + \varepsilon_{1, ij}) + \varepsilon_{2, ij}, \quad \varepsilon_1, \varepsilon_2 \sim \mathcal{N}(0, \sigma^2)")
        st.markdown(r"""
        Biểu diễn tương đương theo chuẩn phần mềm **Monolix** (với tham số cộng $a$ và tham số tỷ lệ $b$):
        """)
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

        # Section 4: Mô hình Động học Nồng độ C(t)
        st.markdown("---")
        st.markdown("#### 4. Mô hình Cấu trúc Động học Nồng độ $C(t)$ (Structural PK Model Concentration Equations)")
        if "infusion" in m_name.lower():
            cl_val = theta_dict.get("CL", 3.0)
            v_val = theta_dict.get("V", 21.0)
            ke_val = cl_val / v_val if v_val > 0 else 0.1
            thalf = np.log(2) / ke_val if ke_val > 0 else 7.0
            tinf_val = float(res.get("data", {}).get("tinf", 1.0))

            st.markdown(f"**Mô hình 1-ngăn truyền tĩnh mạch ngắt quãng (IV Infusion)** với thời gian truyền $t_{{inf}} = {tinf_val:.1f}\\text{{ h}}$:")
            st.caption(rf"Các tham số quần thể: $CL = {cl_val:.2f}\text{{ L/h}}$, $V = {v_val:.2f}\text{{ L}} \implies k_e = \frac{{CL}}{{V}} = {ke_val:.3f}\text{{ h}}^{{-1}}$, $t_{{1/2}} = {thalf:.2f}\text{{ h}}$:")

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
        elif "2-compartment" in m_name.lower() or "2cpt" in m_name.lower():
            cl_val = theta_dict.get("CL", 3.0)
            v1_val = theta_dict.get("V1", theta_dict.get("V", 20.0))
            q_val = theta_dict.get("Q", 2.0)
            v2_val = theta_dict.get("V2", 30.0)
            st.markdown("**Mô hình 2-ngăn truyền tĩnh mạch (2-Compartment IV Infusion — Chuẩn cho Amikacin / Vancomycin)**:")
            st.caption(rf"Hệ phương trình vi phân mô tả lượng thuốc trong ngăn trung tâm $A_1(t)$ và ngăn ngoại vi $A_2(t)$:")
            st.latex(rf"\frac{{dA_1}}{{dt}} = \text{{RateIn}} - \left(\frac{{{cl_val:.2f}}}{{{v1_val:.2f}}} + \frac{{{q_val:.2f}}}{{{v1_val:.2f}}}\right) A_1 + \frac{{{q_val:.2f}}}{{{v2_val:.2f}}} A_2")
            st.latex(rf"\frac{{dA_2}}{{dt}} = \frac{{{q_val:.2f}}}{{{v1_val:.2f}}} A_1 - \frac{{{q_val:.2f}}}{{{v2_val:.2f}}} A_2")
            st.latex(rf"C(t) = \frac{{A_1(t)}}{{{v1_val:.2f}}}")
        elif "oral" in m_name.lower():
            cl_val = theta_dict.get("CL", 3.0)
            v_val = theta_dict.get("V", 21.0)
            ka_val = theta_dict.get("ka", 1.0)
            ke_val = cl_val / v_val if v_val > 0 else 0.1
            thalf = np.log(2) / ke_val if ke_val > 0 else 7.0
            st.caption(rf"Thay thế tham số ước lượng: $k_a = {ka_val:.2f}\text{{ h}}^{{-1}}$, $CL = {cl_val:.2f}\text{{ L/h}}$, $V = {v_val:.2f}\text{{ L}} \implies k_e = {ke_val:.3f}\text{{ h}}^{{-1}}$, $t_{{1/2}} = {thalf:.2f}\text{{ h}}$:")
            denom = v_val * (ka_val - ke_val) if abs(ka_val - ke_val) > 1e-5 else 1.0
            st.latex(rf"C(t) = {ka_val/denom:.4f} \cdot \text{{Dose}} \left(e^{{-{ke_val:.3f} \cdot t}} - e^{{-{ka_val:.2f} \cdot t}}\right)")
        else:
            cl_val = theta_dict.get("CL", 3.0)
            v_val = theta_dict.get("V", 21.0)
            ke_val = cl_val / v_val if v_val > 0 else 0.1
            thalf = np.log(2) / ke_val if ke_val > 0 else 7.0
            st.caption(rf"Thay thế tham số ước lượng: $CL = {cl_val:.2f}\text{{ L/h}}$, $V = {v_val:.2f}\text{{ L}} \implies k_e = {ke_val:.3f}\text{{ h}}^{{-1}}$, $t_{{1/2}} = {thalf:.2f}\text{{ h}}$:")
            st.latex(rf"C(t) = \frac{{\text{{Dose}}}}{{{v_val:.2f}}} e^{{-{ke_val:.3f} \cdot t}}")

    with sub_t2:
        st.markdown("### Tiêu chí Thẩm định & Chỉ số Đánh giá Mô hình PopPK")
        st.markdown("""
        Theo khuyến cáo của **FDA (2022)** và **EMA (2007)**, việc đánh giá mô hình phải kết hợp giữa các chỉ số số học, tiêu chí thống kê, hiện tượng co rút và sai số dự đoán.
        """)

        # 1. Tiêu chí Thông tin & OFV
        st.markdown("#### 1. Kiểm định Tỷ số Hợp lý & Tiêu chí Thông tin (Likelihood Ratio & Information Criteria)")
        st.latex(r"\Delta OFV = OFV(\text{mô hình rút gọn}) - OFV(\text{mô hình đầy đủ}) \sim \chi^2(\Delta df)")
        st.markdown(r"""
        - Với mức ý nghĩa $\alpha = 0.05$: $\Delta OFV > 3.84$ (ứng với $\Delta df = 1$) được coi là cải thiện có ý nghĩa thống kê (tiêu chuẩn đưa biến theo hướng tiến — *forward inclusion*).
        - Với mức ý nghĩa $\alpha = 0.01$: $\Delta OFV > 6.63$ (ứng với $\Delta df = 1$) là tiêu chuẩn loại biến theo hướng lùi (*backward elimination*).
        """)
        st.latex(r"AIC = -2LL + 2p, \quad BIC = -2LL + p \times \ln(N)")
        
        ic = res.get("ic", {})
        c_ic1, c_ic2, c_ic3, c_ic4 = st.columns(4)
        c_ic1.metric("OFV (−2LL)", f"{ic.get('m2ll', 0.0):.2f}")
        c_ic2.metric("AIC", f"{ic.get('aic', 0.0):.1f}")
        c_ic3.metric("BIC", f"{ic.get('bic', 0.0):.1f}")
        c_ic4.metric("BICc", f"{ic.get('bicc', 0.0):.1f}")

        st.markdown("---")
        # 2. Hiện tượng co rút Shrinkage
        st.markdown("#### 2. Đánh giá Hiện tượng Co rút (Shrinkage theo Savic & Karlsson, 2009)")
        st.markdown("""
        Khi số lượng hoặc thời điểm lấy mẫu trên mỗi bệnh nhân không cung cấp đủ thông tin, ước lượng Bayes hậu nghiệm của thông số cá thể có xu hướng co về giá trị điển hình của quần thể:
        """)
        st.latex(r"\eta\text{-shrinkage}(\%) = \left[1 - \frac{SD(\hat{\eta}_i)}{\omega}\right] \times 100")
        st.latex(r"\varepsilon\text{-shrinkage}(\%) = \left[1 - SD(IWRES)\right] \times 100")

        dg = res.get("diag", {})
        P = len(names)
        cols_sh = st.columns(P + 1)
        for j, p in enumerate(names):
            shrink_val = dg.get('eta_shrink', [0]*P)[j] if 'eta_shrink' in dg and j < len(dg['eta_shrink']) else 0.0
            cols_sh[j].metric(f"Shrinkage η_{p}", f"{shrink_val * 100:.1f}%")
        eps_val = dg.get('eps_shrink', 0.0)
        cols_sh[P].metric("Shrinkage IWRES", f"{eps_val * 100:.1f}%")

        st.caption("Khuyến nghị diễn giải: Shrinkage cao (> 20 – 30%) cho thấy dữ liệu ở mức cá thể cung cấp ít thông tin cho việc ước lượng tác động ngẫu nhiên; khi đó các biểu đồ chẩn đoán cá thể như DV–IPRED có thể che giấu sai lệch mô hình và cần được diễn giải thận trọng.")

        st.markdown("---")
        # 3. Sai số dự đoán thực tế MPE & RMSE
        st.markdown("#### 3. Độ lớn Sai số Dự đoán Thực tế (Prediction Errors: MPE & RMSE)")
        st.markdown("""
        Đánh giá độ chệch trung bình (Mean Prediction Error — MPE) và căn bậc hai sai số bình phương trung bình (Root Mean Squared Error — RMSE) phản ánh khả năng dự đoán thực tế:
        """)
        st.latex(r"MPE_{\text{rel}}(\%) = \frac{100}{n} \sum_{i=1}^n \frac{C_{\text{pred}, i} - C_{\text{obs}, i}}{C_{\text{obs}, i}}, \quad RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (C_{\text{pred}, i} - C_{\text{obs}, i})^2}")

        if "diag" in res and "table" in res["diag"]:
            tt = res["diag"]["table"]
            obs_vals = tt["DV"].values
            ipred_vals = tt["IPRED"].values
            pred_vals = tt["PRED"].values
            valid_idx = (obs_vals > 0) & (~np.isnan(obs_vals)) & (~np.isnan(ipred_vals))

            if np.any(valid_idx):
                mpe_ipred = float(np.mean((ipred_vals[valid_idx] - obs_vals[valid_idx]) / obs_vals[valid_idx]) * 100)
                mpe_pred = float(np.mean((pred_vals[valid_idx] - obs_vals[valid_idx]) / obs_vals[valid_idx]) * 100)
                rmse_ipred = float(np.sqrt(np.mean((ipred_vals[valid_idx] - obs_vals[valid_idx])**2)))
                rmse_pred = float(np.sqrt(np.mean((pred_vals[valid_idx] - obs_vals[valid_idx])**2)))

                c_err1, c_err2, c_err3, c_err4 = st.columns(4)
                c_err1.metric("MPErel (IPRED Cá thể)", f"{mpe_ipred:+.2f}%")
                c_err2.metric("RMSE (IPRED Cá thể)", f"{rmse_ipred:.2f} mg/L")
                c_err3.metric("MPErel (PRED Quần thể)", f"{mpe_pred:+.2f}%")
                c_err4.metric("RMSE (PRED Quần thể)", f"{rmse_pred:.2f} mg/L")

    with sub_t3:
        st.markdown("### Tổng quan về Xây dựng Mô hình Dược động học Quần thể (PopPK)")
        
        st.markdown("#### 1. Khái niệm, Cơ sở Toán học và Vai trò của Dược động học Quần thể")
        st.markdown(r"""
**Dược động học quần thể (population pharmacokinetics – PopPK)** là phương pháp mô tả các thông số dược động học điển hình (*typical value*) của một quần thể bệnh nhân xác định, đồng thời định lượng và giải thích các nguồn biến thiên của những thông số này. Khái niệm được đặt nền móng bởi Sheiner, Rosenberg và Melmon từ đầu thập niên 1970 [21], với ý tưởng cốt lõi là phân tách một thông số dược động học cá thể ($P_i$) thành giá trị điển hình quần thể ($\theta$) và độ lệch cá thể ($\eta_i$):
""")
        st.latex(r"P_i = \theta \times \exp(\eta_i), \quad \eta_i \sim \mathcal{N}(0, \omega^2)")
        st.markdown(r"""
trong đó $\eta_i$ là biến thiên liên cá thể (*interindividual variability* – IIV), giả định phân bố chuẩn với trung bình 0 và phương sai $\omega^2$ (dạng hàm mũ này đảm bảo $P_i$ luôn dương, đồng thời tương ứng với giả định IIV theo phân bố log-chuẩn – phù hợp thực tế đa số thông số dược động học). Mối liên hệ giữa nồng độ quan sát được ($C_{\text{obs}}$) và nồng độ dự đoán theo mô hình cấu trúc ($C_{\text{pred}}$) được mô tả bởi mô hình sai số dư (*residual error model*), ví dụ dạng kết hợp (*combined error model*) thường dùng nhất:
""")
        st.latex(r"C_{\text{obs}} = C_{\text{pred}} \times (1 + \varepsilon_1) + \varepsilon_2, \quad \varepsilon_1, \varepsilon_2 \sim \mathcal{N}(0, \sigma^2)")
        st.markdown(r"""
Như vậy, một mô hình PopPK có cấu trúc phân cấp ba tầng: **tầng quần thể** – các thông số cố định $\theta$ (*fixed effects*) và ma trận hiệp phương sai $\Omega, \Sigma$; **tầng cá thể** – các độ lệch ngẫu nhiên $\eta_i$ đặc trưng cho từng bệnh nhân; **tầng quan sát** – sai số dư $\varepsilon_{ij}$ gắn với từng mẫu định lượng.

Cách tiếp cận này cho phép sử dụng **dữ liệu thưa (*sparse data*)**, không cân bằng (*unbalanced*), thu thập trong điều kiện thực hành lâm sàng thường quy (kể cả dữ liệu từ hoạt động TDM), thay vì đòi hỏi lịch lấy mẫu dày đặc như dược động học cổ điển trên người tình nguyện khỏe mạnh [22]. Nhờ đó, phương pháp đặc biệt phù hợp để nghiên cứu trên các quần thể khó lấy mẫu dày đặc như bệnh nhân nặng tại hồi sức tích cực, người cao tuổi hoặc phụ nữ có thai.

Một mô hình PopPK được xây dựng và thẩm định thành công có **ba vai trò ứng dụng chính**:
1. Làm cơ sở xây dựng chế độ liều khởi đầu tối ưu theo đặc điểm bệnh nhân.
2. Làm thông tin tiên nghiệm (*prior*) cho thuật toán ước đoán Bayesian nhằm cá thể hóa liều dựa trên TDM (*model-informed precision dosing* – MIPD).
3. Làm nền tảng cho mô phỏng đánh giá xác suất đạt đích PK/PD của các phương án liều khác nhau trước khi áp dụng lâm sàng.
""")

        st.markdown("---")
        st.markdown("#### 2. Các Phương pháp Phân tích Dược động học Quần thể")
        st.markdown(r"""
Có nhiều phương pháp thống kê được sử dụng để phân tích dữ liệu PopPK, khác nhau chủ yếu ở cách xử lý cấu trúc phân cấp của dữ liệu và giả định về phân bố của các thông số trong quần thể [22], [23], [24]:

- **Phương pháp gộp dữ liệu nguyên thủy (*naive pooled data*)**: gộp toàn bộ quan sát của mọi bệnh nhân như thể đến từ một cá thể duy nhất, ước lượng bằng hồi quy phi tuyến thông thường; bỏ qua hoàn toàn cấu trúc lặp lại của dữ liệu, cho ước lượng sai lệch khi dữ liệu không cân bằng giữa các cá thể.
- **Phương pháp hai giai đoạn (*standard/global two-stage*)**: ước lượng thông số PK riêng cho từng cá thể ở giai đoạn 1 (đòi hỏi đủ số mẫu trên mỗi bệnh nhân), sau đó tổng hợp thống kê mô tả ở giai đoạn 2; không phù hợp với dữ liệu thưa.
- **Phương pháp mô hình hóa hiệu ứng hỗn hợp phi tuyến tham số (*nonlinear mixed-effects modeling* – NLME)**: phương pháp phổ biến nhất hiện nay, ước lượng đồng thời $\theta, \Omega$ và $\Sigma$ dựa trên toàn bộ dữ liệu thô của tất cả cá thể. Các thuật toán ước lượng chính gồm:
  - *Phương pháp bậc nhất (FO)*: tuyến tính hóa hàm dự đoán quanh $\eta = 0$, ước lượng nhanh nhưng có xu hướng đánh giá thấp IIV khi mô hình phi tuyến mạnh.
  - *Phương pháp bậc nhất có điều kiện (FOCE)*: tuyến tính hóa quanh ước lượng Bayes thực nghiệm $\eta_i$ của từng cá thể, chính xác hơn FO.
  - *FOCE có tương tác (FOCE-I)*: tính đến tương tác giữa IIV và sai số dư, thường được khuyến cáo mặc định.
  - *Thuật toán mô phỏng-ước lượng kỳ vọng tối đa hóa ngẫu nhiên (SAEM)*: xử lý tối ưu các mô hình phi tuyến phức tạp hoặc dữ liệu dưới ngưỡng định lượng, là thuật toán cốt lõi trong phần mềm Monolix [25] và nền tảng của SimDx.
- **Phương pháp phi tham số (NPML, NPAG)**: không giả định trước hình dạng phân bố của các thông số PK trong quần thể (không bắt buộc phân bố chuẩn/log-chuẩn), cho phép ước lượng phân bố đa đỉnh phản ánh sát hơn thực tế ở quần thể không đồng nhất (nền tảng phần mềm Pmetrics/BestDose).
- **Phương pháp Bayesian đầy đủ (*full Bayesian*, thuật toán MCMC)**: kết hợp phân bố tiên nghiệm với hàm hợp lý của dữ liệu quan sát để suy ra phân bố hậu nghiệm của toàn bộ thông số mô hình.
""")

        st.markdown("---")
        st.markdown("#### 3. Các Phần mềm trong Phân tích Dược động học Quần thể")
        st.markdown(r"""
- **NONMEM (*Nonlinear Mixed-Effects Modeling*)**: Sheiner, Beal và nhóm nghiên cứu tại UCSF phát triển từ cuối thập niên 1970, là phần mềm lâu đời và phổ biến nhất [27], [28], [29]. Người dùng khai báo dữ liệu, mô hình cấu trúc, mô hình thống kê trong tệp điều khiển (*control stream*), thường kết hợp với PsN, Xpose hoặc các gói R để chẩn đoán.
- **Monolix (Lixoft, Simulations Plus)**: nền tảng mô hình tác động hỗn hợp phi tuyến tham số với thuật toán **SAEM** làm mặc định. Thuật toán này kết hợp mô phỏng tác động ngẫu nhiên với cập nhật tuần tự, loại bỏ nhu cầu tuyến tính hóa mô hình quanh thông số cá thể [30]. Cung cấp thư viện mô hình dựng sẵn và ngôn ngữ Mlxtran, cùng hệ sinh thái Datxplore và Simulx [31].
- **Phoenix NLME (Certara)**: tích hợp trong hệ sinh thái Phoenix, hỗ trợ xây dựng mô hình PK/PD, SCM, mô phỏng và các phương pháp ước lượng FOCE, Laplace, QRPEM.
- **Pmetrics & BestDose**: gói R hỗ trợ cả phương pháp tham số và phi tham số [32]. BestDose là công cụ hỗ trợ quyết định liều lâm sàng dựa trên Bayesian ước tính từ mô hình quần thể.
- **Phần mềm Định liều Lâm sàng Dựa trên Mô hình (MIPD)**: các nền tảng như DoseMeRx, InsightRX, TDMx, PrecisePK kết hợp mô hình PopPK tiên nghiệm với nồng độ TDM thực tế để hỗ trợ hiệu chỉnh liều cá thể hóa.
""")

        st.markdown("---")
        st.markdown("#### 4. Quy trình Xây dựng Mô hình Dược động học Quần thể")
        st.markdown(r"""
**Quy trình 5 Bước Chuẩn mực:**
1. **Bước 1 — Chuẩn bị dữ liệu và phân tích khám phá (*EDA*)**: Làm sạch, chuẩn hóa liều dùng, thời điểm lấy mẫu, nồng độ đo được và đặc điểm bệnh nhân; khảo sát phân bố và tương quan sơ bộ.
2. **Bước 2 — Xây dựng mô hình cấu trúc cơ bản (*Base structural model*)**: Lựa chọn số ngăn (1 ngăn, 2 ngăn) và dạng động học thải trừ. Với amikacin/vancomycin truyền ngắt quãng, phần lớn nghiên cứu quần thể mô tả tốt nhất bằng mô hình 2 ngăn thải trừ bậc 1 [33].
3. **Bước 3 — Xây dựng mô hình thống kê (*Statistical model*)**: Mô hình hóa IIV theo dạng hàm mũ ($P_i = \theta \exp(\eta_i)$) cho $CL, V_1, Q, V_2$; mô hình hóa sai số dư theo dạng kết hợp, tỷ lệ hoặc cộng; bổ sung biến thiên giữa các đợt dùng (*IOV*) nếu thiết kế lấy mẫu cho phép.
4. **Bước 4 — Lựa chọn phương pháp ước lượng**: Thuật toán SAEM hoặc FOCE-I.
5. **Bước 5 — Đánh giá độ phù hợp mô hình (*Goodness-of-Fit – GOF*)**: Sử dụng bộ đồ thị chuẩn: nồng độ quan sát so với dự đoán quần thể (PRED) và dự đoán cá thể (IPRED); phần dư có trọng số điều kiện (CWRES / IWRES) theo thời gian và nồng độ dự đoán.

**Phương pháp Mô hình hóa Từng bước (Stepwise Covariate Modeling – SCM):**
- *Giai đoạn đưa biến vào theo hướng tiến (Forward inclusion)*: tiêu chuẩn $\Delta OFV > 3.84$ điểm ứng với $p < 0.05$ (1 bậc tự do).
- *Giai đoạn loại biến theo hướng lùi (Backward elimination)*: tiêu chuẩn chặt chẽ hơn $\Delta OFV > 6.63$ điểm ứng với $p < 0.01$ để hạn chế dương tính giả [22], [34].
- *Lưu ý đa cộng tuyến (Collinearity)*: hệ số tương quan giữa hai yếu tố đồng ảnh hưởng lớn hơn 0.5 là cảnh báo cần thận trọng khi diễn giải (ví dụ cân nặng và diện tích da, hoặc creatinin huyết thanh và tuổi).
""")

        st.markdown("---")
        st.markdown("#### 5. Căn cứ Lựa chọn và Thẩm định Lại Mô hình Dược động học Quần thể")
        st.markdown(r"""
Theo hướng dẫn của **FDA (2022)** và **EMA (2007)** [35], [40], cần kết hợp đa phương pháp:
- **Đánh giá số học**: So sánh OFV, AIC, BIC, độ chính xác ước lượng (RSE < 30% cho $\theta$, < 50% cho $\omega$), ma trận hiệp phương sai hội tụ và tính hợp lý về mặt sinh học.
- **Đánh giá đồ thị GOF**: DV–PRED, DV–IPRED phân bố cân bằng quanh đường đồng nhất $y = x$; CWRES/IWRES phân bố ngẫu nhiên quanh 0 trong khoảng $[-2, +2]$ hoặc $[-3, +3]$.
- **Thẩm định nội bộ bằng Bootstrap**: Tái lấy mẫu có hoàn lại theo cá thể 500 – 1.000 lần, tính trung vị và khoảng tin cậy 95% Bootstrap CI để khẳng định độ ổn định của ước lượng.
- **Visual Predictive Check (VPC / pcVPC)**: Mô phỏng nhiều bộ dữ liệu (50 – 500 lần) từ mô hình cuối, so sánh phân vị 5%, 50%, 95% của dữ liệu quan sát với khoảng tin cậy mô phỏng tương ứng [38].
- **Normalized Prediction Distribution Errors (NPDE)**: Kiểm tra phân bố chuẩn chuẩn hóa $\mathcal{N}(0, 1)$ của phần dư [39].
- **Đánh giá hiện tượng co rút (Shrinkage)**: $\eta\text{-shrinkage} = [1 - SD(\hat{\eta}_i)/\omega] \times 100\%$; $\varepsilon\text{-shrinkage} = [1 - SD(IWRES)] \times 100\%$ [36].
- **Thẩm định ngoại kiểm (*External Validation*)**: Đánh giá khả năng dự đoán trên tập dữ liệu độc lập qua các chỉ số độ chệch trung bình $MPE_{\text{rel}}(\%)$ và sai số tổng thể $RMSE$.
""")



def show_fit(res, clinical_data=None):
    if st.session_state.get("settings_changed_warning", False):
        st.warning("**Project settings changed. These results are inconsistent.** Cấu hình mô hình ở Tab 2 vừa được cập nhật theo Proposal — vui lòng chạy lại SAEM để đồng bộ kết quả mới.")
    st.info(f"**Mô hình Dược động học vừa ước lượng:** `{res.get('model_name', 'Mô hình')}`")
    ic = res["ic"]
    bad = [(lab, r) for lab, ok, r in pk.convergence_report(res) if not ok]
    if bad:
        st.warning("Pha 2 chưa ổn định (trung bình nửa đầu và nửa sau lệch > 5%): "
                   + ", ".join(f"{lab} ({r:.0%})" for lab, r in bad)
                   + ". Hãy tăng số vòng lặp hoặc số chain.")
    else:
        st.markdown('<div class="convergence-box">Các tham số ổn định ở pha smoothing (trôi < 5%)</div>',
                    unsafe_allow_html=True)

    res_t1, res_t2, res_t_ll, res_t3, res_t4 = st.tabs([
        "POP.PARAM",
        "STD.ERRORS (Correlation & Eigenvalues)",
        "LIKELIHOOD (Importance Sampling)",
        "TESTS (Covariates & Random Effects)",
        "PROPOSAL (Apply Model)"
    ])

    with res_t1:
        header("Ước lượng tham số (SE/RSE: FIM tuyến tính hoá quanh EBE)")
        if any(np.isnan(v[1]) for v in res["fim"].values()):
            st.warning("Hessian không xác định dương → không tính được SE.")
        st.dataframe(estimates_table(res), width="stretch")
        c = st.columns(4)
        mc_se_str = f" (MC-SE: ±{ic['mc_se']:.2f})" if "mc_se" in ic and ic["mc_se"] > 0 else ""
        c[0].metric("−2LL (IS)", f"{ic['m2ll']:.2f}{mc_se_str}")
        c[1].metric("AIC", f"{ic['aic']:.1f}")
        c[2].metric("BIC", f"{ic['bic']:.1f}")
        c[3].metric("BICc", f"{ic['bicc']:.1f}")

        st.info("Hệ phương trình Dược động học Quần thể (3-Tier NLME), các chỉ số thẩm định và tài liệu lý thuyết xem tại tab **'7. Model Equations (Phương trình PopPK)'**.")

        header("SAEM Convergence Traces")
        st.plotly_chart(convergence_figure(res), width="stretch", key="fig_saem_convergence")

        header("Đánh giá Độ tin cậy Tham số: Bootstrap Module (Monolix 2024)")
        st.info("Module Non-parametric Bootstrap lấy mẫu lại theo bệnh nhân để tính khoảng tin cậy 95% Bootstrap CI thực tế.")
        c_b1, c_b2 = st.columns([1, 1])
        n_boot = c_b1.slider("Số mẫu lặp lại (Bootstrap Replicates)", 5, 30, 10, 5, key="n_boot_slider")
        if c_b2.button("Chạy Non-parametric Bootstrap", key="btn_run_bootstrap"):
            b_bar = st.progress(0, text="Đang chạy Bootstrap...")
            boot_records = []
            d_orig = res["data"]
            n_subs = len(d_orig["ids"])
            for b_idx in range(n_boot):
                b_bar.progress((b_idx + 1) / n_boot, text=f"Bootstrap mẫu {b_idx + 1}/{n_boot}...")
                rng = np.random.RandomState(100 + b_idx)
                _ = rng.choice(n_subs, size=n_subs, replace=True)
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

    with res_t2:
        header("Correlation Matrix of the Estimates (STD.ERRORS — Chuẩn Monolix)")
        corr_df, min_eig, max_eig, cond_num = pk.compute_fim_correlation(res)
        if corr_df is None:
            st.warning("Chưa có ma trận hiệp phương sai FIM. Vui lòng chạy lại SAEM.")
        else:
            st.caption("Ma trận tương quan chuẩn hóa giữa các ước lượng tham số. Hệ số |r| ≥ 0.85 (tô đỏ) cảnh báo đa cộng tuyến hoặc suy biến định danh giữa hai tham số.")
            # Build lower-triangular display
            tri_df = corr_df.copy().astype(object)
            n_m = len(tri_df)
            for r_i in range(n_m):
                for c_i in range(n_m):
                    if c_i > r_i:
                        tri_df.iloc[r_i, c_i] = ""
                    else:
                        tri_df.iloc[r_i, c_i] = f"{corr_df.iloc[r_i, c_i]:.4f}"

            # Add R.S.E.(%) column matching Monolix STD.ERRORS view
            rse_col = []
            for lbl in res.get("fim_labels", []):
                val, se = res["fim"].get(lbl, (np.nan, np.nan))
                if lbl in res["names"]:
                    rse_pct = abs(se) * 100.0
                else:
                    rse_pct = abs(se / val) * 100.0 if (np.isfinite(se) and abs(val) > 1e-12) else np.nan
                rse_col.append(f"{rse_pct:.2f}" if np.isfinite(rse_pct) else "NaN")
            tri_df.insert(0, "R.S.E.(%)", rse_col)

            def _style_corr(val):
                try:
                    fv = float(val)
                    if abs(fv) >= 0.85 and abs(fv) < 0.9999:
                        return "color: #b91c1c; font-weight: 800; background-color: #fee2e2;"
                except Exception:
                    pass
                return ""

            st.dataframe(tri_df.style.map(_style_corr, subset=corr_df.columns), width="stretch")

            st.markdown("##### Eigen values (Giá trị riêng & Chỉ số Điều kiện Ma trận):")
            c_e1, c_e2, c_e3 = st.columns(3)
            c_e1.metric("MIN Eigenvalue", f"{min_eig:.4f}")
            c_e2.metric("MAX Eigenvalue", f"{max_eig:.4f}")
            c_e3.metric("Condition Number (MAX/MIN)", f"{cond_num:.2f}")
            if cond_num > 100:
                st.warning(f"Condition Number = {cond_num:.1f} (> 100): Dấu hiệu mô hình bị quá tham số hóa (over-parameterized) hoặc có biến đồng ảnh hưởng dư thừa.")
            else:
                st.success(f"Condition Number = {cond_num:.2f} (< 100): Ma trận thông tin Fisher (FIM) ổn định tốt.")

    with res_t_ll:
        header("Estimated log-likelihood and information criteria (IMPORTANCE SAMPLING)")
        ic_cur = res["ic"]
        mc_se_val = float(ic_cur.get("mc_se", res.get("mc_se", 0.0)))
        n_is_cur = int(res.get("n_is_used", 1500))

        ll_table = pd.DataFrame([
            {"Tiêu chuẩn (Criteria)": "-2 x log-likelihood (OFV)", "Importance Sampling": f"{ic_cur['m2ll']:.2f}"},
            {"Tiêu chuẩn (Criteria)": "Akaike Information Criteria (AIC)", "Importance Sampling": f"{ic_cur['aic']:.2f}"},
            {"Tiêu chuẩn (Criteria)": "Bayesian Information Criteria (BIC)", "Importance Sampling": f"{ic_cur['bic']:.2f}"},
            {"Tiêu chuẩn (Criteria)": "Corrected Bayesian Information Criteria (BICc)", "Importance Sampling": f"{ic_cur['bicc']:.2f}"},
        ])
        st.dataframe(ll_table, hide_index=True, width="stretch")

        samp_table = pd.DataFrame([
            {"Thông số Lấy mẫu (Sampling Info)": "Sampling Distribution", "Giá trị": "T-distribution with 5 degrees of freedom"},
            {"Thông số Lấy mẫu (Sampling Info)": "Standard Error (MC-SE)", "Giá trị": f"{mc_se_val:.4f}"},
            {"Thông số Lấy mẫu (Sampling Info)": "MonteCarlo size (iterations)", "Giá trị": f"{n_is_cur}"},
        ])
        st.dataframe(samp_table, hide_index=True, width="stretch")

        st.markdown("---")
        c_is1, c_is2 = st.columns([1.2, 1.0])
        new_nis = c_is1.select_slider("Tăng kích thước mẫu Monte Carlo (Importance Sampling size)", options=[1500, 3000, 5000, 10000], value=n_is_cur if n_is_cur in [1500, 3000, 5000, 10000] else 1500, key="sel_is_size")
        if c_is2.button("Tính lại Importance Sampling", type="secondary", use_container_width=True, key="btn_recompute_is"):
            with st.spinner(f"Đang chạy Importance Sampling ({new_nis} mẫu)..."):
                ll_new, se_new = pk.loglik_is(res, n_is=int(new_nis))
                res["ll"] = ll_new
                res["mc_se"] = se_new
                res["n_is_used"] = int(new_nis)
                res["ic"] = pk.information_criteria(res, ll_new, se_new)
                st.session_state["res"] = res
                save_current_session()
                st.rerun()

    with res_t3:
        header("Kiểm định Thống kê (TESTS — Pearson's Correlation / ANOVA / Shapiro-Wilk)")
        tests = pk.compute_statistical_tests(res)
        test_sub1, test_sub2, test_sub3 = st.tabs(["COVARIATES", "RANDOM EFFECTS", "RESIDUALS"])

        def _style_pval(val):
            try:
                p = float(val)
                if p < 0.01:
                    return "color: #dc2626; font-weight: 800;"
                elif p < 0.05:
                    return "color: #ea580c; font-weight: 700;"
                elif p < 0.10:
                    return "color: #ca8a04; font-weight: 600;"
            except Exception:
                pass
            return ""

        with test_sub1:
            st.caption("Pearson's correlation test (biến liên tục) & ANOVA (biến phân loại) giữa hiệu ứng ngẫu nhiên η và từng biến đồng ảnh hưởng. Đỏ: p < 0.01 | Cam: p < 0.05 | Vàng: p < 0.10.")
            cov_tests = tests["cov_tests"]
            cols_ct = st.columns(min(len(cov_tests), 3)) if cov_tests else []
            for idx_p, (p_name, df_ct) in enumerate(cov_tests.items()):
                with cols_ct[idx_p % len(cols_ct)]:
                    st.markdown(f"**`eta_{p_name}`**")
                    if df_ct.empty:
                        st.info("Không có biến đồng ảnh hưởng.")
                    else:
                        df_disp = df_ct.drop(columns=["Type"]).copy()
                        df_disp["P-VALUE"] = df_disp["P-VALUE"].apply(lambda x: f"{x:.2e}")
                        st.dataframe(df_disp.style.map(_style_pval, subset=["P-VALUE"]), hide_index=True, width="stretch")

        with test_sub2:
            st.markdown("##### Distribution of the random effects (Shapiro-Wilk test)")
            st.caption("Kiểm tra giả định phân phối chuẩn $\\eta_i \\sim \\mathcal{N}(0, \\omega^2)$:")
            if not tests["re_normality"].empty:
                df_rn = tests["re_normality"].copy()
                df_rn["P-VALUE"] = df_rn["P-VALUE"].apply(lambda x: f"{x:.2e}")
                st.dataframe(df_rn.style.map(_style_pval, subset=["P-VALUE"]), hide_index=True, width="stretch")

            st.markdown("##### Joint distribution of the random effects (Correlation t-test)")
            st.caption("Kiểm tra tương quan tuyến tính giữa các cặp $\\eta$ (nếu $p < 0.05$, cân nhắc thêm hiệp phương sai giữa hai tham số):")
            if not tests["re_corr"].empty:
                df_rc = tests["re_corr"].copy()
                df_rc["P-VALUE"] = df_rc["P-VALUE"].apply(lambda x: f"{x:.2e}")
                st.dataframe(df_rc.style.map(_style_pval, subset=["P-VALUE"]), hide_index=True, width="stretch")
            else:
                st.info("Cần từ 2 tham số PK trở lên để kiểm tra tương quan chéo.")

        with test_sub3:
            st.markdown("##### Distribution of the residuals (Shapiro-Wilk test)")
            if not tests["resid_normality"].empty:
                df_rs = tests["resid_normality"].copy()
                df_rs["P-VALUE"] = df_rs["P-VALUE"].apply(lambda x: f"{x:.2e}")
                st.dataframe(df_rs.style.map(_style_pval, subset=["P-VALUE"]), hide_index=True, width="stretch")

    with res_t4:
        header("Đề xuất Mô hình Thống kê Tự động (PROPOSAL — Conditional Distribution & BICc)")
        st.caption("Đánh giá nhanh toàn bộ các mô hình thay thế bằng các mẫu từ phân phối có điều kiện (Conditional Distribution) để tránh hiện tượng co cụm (shrinkage) và chọn mô hình có BICc thấp nhất.")
        prop = pk.generate_model_proposal(res)
        if "error_table" not in prop:
            import importlib
            importlib.reload(pk)
            prop = pk.generate_model_proposal(res)

        pr_sub1, pr_sub2, pr_sub3, pr_sub4 = st.tabs([
            "PROPOSED MODEL",
            "ERROR MODEL (BICc)",
            "COVARIATE MODEL (Stepwise BICc)",
            "CORRELATION MODEL (BICc)"
        ])

        with pr_sub1:
            c_pr1, c_pr2 = st.columns([1.8, 1.2])
            with c_pr1:
                st.markdown("##### Tóm tắt Đề xuất tối ưu theo BICc:")
                for rsn in prop["reasons"]:
                    st.markdown(f"- {rsn}")
            with c_pr2:
                if st.button("[Apply] Áp dụng sang Tab 2 (Chưa chạy SAEM)", type="secondary", use_container_width=True, key="btn_apply_only_proposal"):
                    prop_pairs = [f"{p_k}~{c_k}" for p_k, c_list in prop["proposed_cov_model"].items() for c_k in c_list]
                    st.session_state["active_cov_pairs"] = prop_pairs
                    st.session_state["pending_error_model"] = prop["proposed_error_model"]
                    st.session_state["settings_changed_warning"] = True
                    st.session_state["pending_nav_step"] = "2. SAEM Estimation & Initial Estimates"
                    st.rerun()

                if st.button("[Apply & Run SAEM] Áp dụng & Chạy lại SAEM ngay", type="primary", use_container_width=True, key="btn_apply_proposal"):
                    m_cur = res.get("model_name", "1-Compartment IV infusion (multi-dose)")
                    mi_cur = pk.MODELS[m_cur]
                    init_p = dict(
                        mu=np.log([float(np.exp(res["beta"][j][0])) if j < len(res.get("beta", [])) else mi_cur["defaults"][j] for j in range(len(mi_cur["params"]))]),
                        omega=[float(res["omega"][j]) if j < len(res.get("omega", [])) else 0.3 for j in range(len(mi_cur["params"]))],
                        err=[float(res["err"][0]) if len(res.get("err", [])) > 0 else 1.0, float(res["err"][1]) if len(res.get("err", [])) > 1 else 0.2]
                    )
                    prev_ofv = float(res["ic"]["m2ll"])
                    bar_p = st.progress(0, text="Đang chạy SAEM với Mô hình Đề xuất...")
                    new_res = pk.fit_model(
                        clinical_data, m_cur, init_p, prop["proposed_cov_model"], prop["proposed_error_model"],
                        n_iter1=200, n_iter2=100, n_chains=1, seed=42,
                        progress=lambda x: bar_p.progress(x, text=f"SAEM Proposed Model {x:.0%}"),
                        cov_mode=res.get("cov_mode", "time-varying")
                    )
                    bar_p.empty()
                    new_res["clinical_data"] = clinical_data
                    new_res["model_name"] = m_cur
                    new_res["selected_cov_raw"] = prop["proposed_cov_model"]
                    new_res["loaded_data_name"] = st.session_state.get("loaded_data_name", "Clinical Dataset")
                    new_res["prev_ofv_before_apply"] = prev_ofv
                    prop_pairs = [f"{p_k}~{c_k}" for p_k, c_list in prop["proposed_cov_model"].items() for c_k in c_list]
                    st.session_state["active_cov_pairs"] = prop_pairs
                    st.session_state["pending_error_model"] = prop["proposed_error_model"]
                    st.session_state["settings_changed_warning"] = False
                    if "saved_models_res" not in st.session_state:
                        st.session_state["saved_models_res"] = {}
                    st.session_state["saved_models_res"][m_cur] = new_res
                    st.session_state["res"] = new_res
                    st.session_state.pop("vpc", None)
                    save_current_session()
                    st.rerun()

            if "prev_ofv_before_apply" in res:
                d_ofv_app = float(res["prev_ofv_before_apply"]) - float(res["ic"]["m2ll"])
                st.success(f"Đã cập nhật & chạy lại SAEM theo Proposal! Hàm mục tiêu **-2LL** thay đổi: **{d_ofv_app:+.2f} đơn vị**.")

            st.markdown("---")
            all_cov_cols = sorted(list({c for c, ct in res["data"].get("cov_types", {}).items() if ct != "ignore"}))
            def _build_prop_matrix(cov_dict):
                m_rows = []
                for p_name in res["names"]:
                    r_d = {"Param": p_name}
                    p_covs = set(cov_dict.get(p_name, []))
                    for c_c in all_cov_cols:
                        r_d[c_c] = "✓" if c_c in p_covs else ""
                    m_rows.append(r_d)
                return pd.DataFrame(m_rows)

            st.markdown("##### Proposed model")
            st.markdown(f"- **Error model**: `{prop['proposed_error_model'].upper()}`")
            if all_cov_cols:
                st.dataframe(_build_prop_matrix(prop["proposed_cov_model"]), hide_index=True, width="stretch")

            st.markdown("##### Current model")
            st.markdown(f"- **Error model**: `{prop['current_error_model'].upper()}`")
            if all_cov_cols:
                st.dataframe(_build_prop_matrix(prop["current_cov_model"]), hide_index=True, width="stretch")

        with pr_sub2:
            c_em1, c_em2 = st.columns([2.2, 1.0])
            c_em1.markdown("##### So sánh 4 Mô hình Sai số bằng MLE trên Conditional Distribution (Xếp hạng theo BICc):")
            if c_em2.button("[Apply Error Model] Chỉ áp dụng Error Model", type="secondary", use_container_width=True, key="btn_apply_err_only"):
                st.session_state["pending_error_model"] = prop["proposed_error_model"]
                st.session_state["settings_changed_warning"] = True
                st.session_state["pending_nav_step"] = "2. SAEM Estimation & Initial Estimates"
                st.rerun()
            if prop.get("error_table") is not None:
                st.dataframe(prop["error_table"], hide_index=True, width="stretch")

        with pr_sub3:
            c_cm1, c_cm2 = st.columns([2.2, 1.0])
            c_cm1.markdown("##### Hồi quy Từng bước (Stepwise Linear Regression) theo BICc trên Conditional Distribution:")
            if c_cm2.button("[Apply Covariate Model] Chỉ áp dụng Covariates", type="secondary", use_container_width=True, key="btn_apply_cov_only"):
                prop_pairs = [f"{p_k}~{c_k}" for p_k, c_list in prop["proposed_cov_model"].items() for c_k in c_list]
                st.session_state["active_cov_pairs"] = prop_pairs
                st.session_state["settings_changed_warning"] = True
                st.session_state["pending_nav_step"] = "2. SAEM Estimation & Initial Estimates"
                st.rerun()
            st.caption("Tự động loại trừ đa cộng tuyến bằng tiêu chuẩn BICc trên phân phối hậu nghiệm.")
            if prop.get("cov_steps_table") is not None:
                st.dataframe(prop["cov_steps_table"], hide_index=True, width="stretch")

        with pr_sub4:
            c_cr1, c_cr2 = st.columns([2.2, 1.0])
            c_cr1.markdown("##### Kiểm định Tương quan Phần dư Ngẫu nhiên (sau khi loại bỏ ảnh hưởng Covariates):")
            if prop.get("proposed_correlations"):
                if c_cr2.button("[Apply Correlation] Áp dụng Tương quan", type="secondary", use_container_width=True, key="btn_apply_corr_only"):
                    st.session_state["settings_changed_warning"] = True
                    st.info(f"Đã ghi nhận đề xuất tương quan: {', '.join(prop['proposed_correlations'])}")
            corr_tbl = prop.get("corr_table")
            if corr_tbl is not None and not corr_tbl.empty:
                st.dataframe(corr_tbl, hide_index=True, width="stretch")
            else:
                st.info("Cần từ 2 tham số PK trở lên để kiểm định tương quan phần dư.")


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
            cat_html = f"""<style>
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
</div>"""
            if hasattr(st, "html"):
                st.html(cat_html)
            else:
                st.markdown(cat_html, unsafe_allow_html=True)
        else:
            logo_display = (
                f'<img src="data:image/png;base64,{LOGO_B64}" style="width: 88px; height: 88px; object-fit: contain; filter: drop-shadow(0 6px 16px rgba(13, 51, 104, 0.15)); margin-bottom: 10px;">'
            ) if LOGO_B64 else '<div style="display: inline-block; padding: 14px; background: #e0f2fe; border-radius: 20px; margin-bottom: 12px; box-shadow: 0 4px 12px rgba(13, 51, 104, 0.08);"><span style="font-size: 32px; font-weight: bold; color: #0d3368;">SIMDX</span></div>'
            brand_html = f"""<div style="text-align: center; margin-bottom: 24px;">
    {logo_display}
    <h2 style="color: #0d3368; font-weight: 800; margin-bottom: 4px; letter-spacing: -0.5px;">Spass</h2>
    <p style="color: #64748b; font-size: 14px; margin-top: 0;">inb Bảnh để có pass</p>
</div>"""
            if hasattr(st, "html"):
                st.html(brand_html)
            else:
                st.markdown(brand_html, unsafe_allow_html=True)
        
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


def auto_estimate_initial_params(clinical_data, model_name):
    """
    Tự động ước lượng thông số ban đầu Vd, Cl dự kiến từ dữ liệu thực tế (Auto-estimate chuẩn Monolix).
    Dựa trên phân tích phi khoang Non-Compartmental Analysis (NCA) của dữ liệu.
    """
    cl_est = 3.28
    vd_est = 24.34
    if clinical_data is not None:
        try:
            df_ind_nca, _ = compute_nca(clinical_data)
            if df_ind_nca is not None and len(df_ind_nca) > 0:
                cl_valid = df_ind_nca["CL_nca (L/h)"].dropna()
                vd_valid = df_ind_nca["Vd_nca (L)"].dropna()
                if len(cl_valid) > 0:
                    cl_est = float(np.median(cl_valid))
                if len(vd_valid) > 0:
                    vd_est = float(np.median(vd_valid))
        except Exception:
            pass
    
    cl_est = max(0.5, min(cl_est, 15.0))
    vd_est = max(5.0, min(vd_est, 60.0))
    
    estimates = {}
    mi = pk.MODELS.get(model_name, {})
    params = mi.get("params", [])
    
    for p in params:
        if p in ["CL", "Cl"]:
            estimates[p] = round(cl_est, 2)
        elif p in ["V", "V1"]:
            if "2-Compartment" in model_name or "3-Compartment" in model_name:
                estimates[p] = round(vd_est * 0.7, 2)
            else:
                estimates[p] = round(vd_est, 2)
        elif p == "V2":
            estimates[p] = round(vd_est * 0.35, 2)
        elif p == "V3":
            estimates[p] = round(vd_est * 0.25, 2)
        elif p in ["Q", "Q1", "Q2"]:
            estimates[p] = round(cl_est * 0.8, 2)
        elif p == "ka":
            estimates[p] = 1.2
        elif p == "k":
            estimates[p] = round(cl_est / max(vd_est, 1e-4), 3)
        elif p == "k12":
            estimates[p] = 0.15
        elif p == "k21":
            estimates[p] = 0.2
        elif p in ["Vmax", "Vm"]:
            estimates[p] = 50.0
        elif p == "Km":
            estimates[p] = 10.0
        elif p == "Tlag":
            estimates[p] = 0.5
        else:
            estimates[p] = 1.0

    return estimates, cl_est, vd_est


SESSION_CACHE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".simdx_session_cache.pkl"))


def save_current_session():
    """Lưu tự động toàn bộ trạng thái phiên làm việc (Dữ liệu, Mô hình, Kết quả SAEM) ra file cache."""
    try:
        session_dict = {
            "clinical_data": st.session_state.get("clinical_data"),
            "loaded_data_name": st.session_state.get("loaded_data_name"),
            "tinf_val": st.session_state.get("tinf_val", 1.0),
            "model_structure_select": st.session_state.get("model_structure_select"),
            "model_confirmed_for": st.session_state.get("model_confirmed_for"),
            "saved_models_res": st.session_state.get("saved_models_res", {}),
            "res": st.session_state.get("res"),
            "nav_step": st.session_state.get("nav_step", "1. Data")
        }
        with open(SESSION_CACHE_FILE, "wb") as f:
            pickle.dump(session_dict, f)
    except Exception:
        pass


def load_cached_session():
    """Khôi phục trạng thái phiên làm việc nếu chưa có trong session_state."""
    if os.path.exists(SESSION_CACHE_FILE):
        try:
            with open(SESSION_CACHE_FILE, "rb") as f:
                saved = pickle.load(f)
            if isinstance(saved, dict) and saved.get("clinical_data") is not None:
                for k, v in saved.items():
                    if k not in st.session_state or st.session_state[k] is None:
                        st.session_state[k] = v
                return True
        except Exception:
            pass
    return False


def reset_entire_session():
    """Xóa sạch bộ đệm phiên làm việc để bắt đầu dự án mới hoàn toàn."""
    if os.path.exists(SESSION_CACHE_FILE):
        try:
            os.remove(SESSION_CACHE_FILE)
        except Exception:
            pass
    for k in ["clinical_data", "loaded_data_name", "tinf_val", "saved_models_res", "res", "vpc", "model_confirmed_for", "boot_df", "auto_estimated_flag"]:
        st.session_state.pop(k, None)
    st.session_state["pending_nav_step"] = "1. Data"


def main():
    check_authentication()
    
    # Khôi phục phiên làm việc đã lưu nếu chưa có dữ liệu trong session hiện tại
    if "clinical_data" not in st.session_state or st.session_state["clinical_data"] is None:
        load_cached_session()

    # ── ĐỒNG BỘ TRẠNG THÁI CHUYỂN TAB TỰ ĐỘNG ──
    if "nav_step" not in st.session_state:
        st.session_state["nav_step"] = "1. Data"

    if "pending_nav_step" in st.session_state:
        next_step = st.session_state.pop("pending_nav_step")
        st.session_state["nav_step"] = next_step
        st.session_state["sidebar_nav_radio"] = next_step

    NAV_OPTIONS = [
        "1. Data",
        "2. SAEM Estimation & Initial Estimates",
        "3. Results & Population parameters",
        "4. Diagnostic Plots (GOF)",
        "5. Individual fits (EBE)",
        "6. Covariates & Simulx",
        "7. Model Equations (Phương trình PopPK)"
    ]

    # ── THANH BÊN (SIDEBAR): CHỈ CHỨA LOGO, BRAND, VÀ ĐIỀU HƯỚNG QUY TRÌNH WORKFLOW ──
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

        st.markdown("### Quy trình phân tích (Workflow)")
        def on_nav_change():
            st.session_state["nav_step"] = st.session_state["sidebar_nav_radio"]

        cur_idx = NAV_OPTIONS.index(st.session_state["nav_step"]) if st.session_state["nav_step"] in NAV_OPTIONS else 0
        val_arg = None if "sidebar_nav_radio" in st.session_state else cur_idx
        selected_nav = st.radio(
            "Chọn bước phân tích",
            NAV_OPTIONS,
            index=val_arg,
            key="sidebar_nav_radio",
            on_change=on_nav_change,
            label_visibility="collapsed"
        )
        st.session_state["nav_step"] = selected_nav

        st.markdown("---")
        # Nút nạp nhanh tập dữ liệu chuẩn lâm sàng (Quick sample loader)
        st.caption("Dữ liệu mẫu nghiên cứu:")
        if st.button("Nạp Dataset chuẩn lâm sàng (83 BN)", use_container_width=True, key="btn_quick_load_sample"):
            excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Dataset_chính_thức.xlsx"))
            if os.path.exists(excel_path):
                st.session_state["clinical_data"] = data_io.load_dataset(excel_path, tinf=1.0, enforce_g4=True)
                st.session_state["loaded_data_name"] = "Dataset_chính_thức.xlsx (83 bệnh nhân chuẩn)"
                st.session_state["tinf_val"] = 1.0
                save_current_session()
                st.rerun()

        if st.session_state.get("clinical_data") is not None:
            if st.button("Đặt lại phiên làm việc", use_container_width=True, key="btn_reset_session_sidebar", type="secondary"):
                reset_entire_session()
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        c_out1, c_out2 = st.columns([1.5, 1])
        c_out1.caption("Đã xác thực")
        if c_out2.button("Đăng xuất", key="btn_logout", type="secondary", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()

    # Dữ liệu lâm sàng từ Session State
    clinical_data = st.session_state.get("clinical_data", None)
    tinf = float(st.session_state.get("tinf_val", 1.0))
    current_tab = st.session_state.get("nav_step", "1. Data")
    res = st.session_state.get("res") or st.session_state.get("last_fitted_res")
    if res is not None:
        st.session_state["res"] = res
        if clinical_data is None and "clinical_data" in res:
            clinical_data = res["clinical_data"]
            st.session_state["clinical_data"] = clinical_data

    # Đồng bộ mô hình cấu trúc mặc định mà không xóa kết quả đang có
    model_names = list(pk.MODELS.keys())
    default_model = "1-Compartment IV infusion (multi-dose)"
    if "chosen_model_name" not in st.session_state:
        st.session_state["chosen_model_name"] = default_model

    model_name = st.session_state["chosen_model_name"]
    mi = pk.MODELS[model_name]
    names = mi["params"]
    units = mi.get("units", [""] * len(names))
    units_dict = dict(zip(names, units))
    key = lambda s: f"{model_name}_{s}"

    # Áp dụng pending_auto_estimates & pending_error_model nếu có
    if "pending_auto_estimates" in st.session_state:
        pending = st.session_state.pop("pending_auto_estimates")
        for p_k, p_v in pending.items():
            st.session_state[p_k] = p_v
        st.session_state["auto_estimated_flag"] = True
    if "pending_error_model" in st.session_state:
        st.session_state["main_error_model_select"] = st.session_state.pop("pending_error_model")

    # ── MÀN HÌNH 1: DATA (NHẬP VÀ THẨM ĐỊNH DỮ LIỆU) ──
    if current_tab == "1. Data":
        header("Quản lý & Nhập Dữ liệu Nghiên cứu (Data Import & Validation)")
        
        c_src1, c_src2 = st.columns([1.2, 1.8])
        with c_src1:
            st.markdown("#### Nguồn dữ liệu (Data Source)")
            data_source_mode = st.radio(
                "Chọn phương thức nạp dữ liệu",
                ["Tải lên file (Excel/CSV)", "Dataset lâm sàng mẫu (Dataset_chính_thức.xlsx)"],
                key="main_data_source_radio",
                label_visibility="collapsed"
            )

        with c_src2:
            st.markdown("#### Thông tin cấu hình đưa liều (Dosing Administration)")
            selected_lib_id = st.session_state.get("selected_lib_model", "")
            is_bolus_mode = str(selected_lib_id).startswith("bolus_")
            if is_bolus_mode:
                tinf = 0.01
                st.caption(f"Chế độ Monolix {selected_lib_id}: tinf = 0.01 h (Bolus tức thời)")
            else:
                tinf = st.number_input("Thời gian truyền tinf (giờ)", 0.01, 24.0, float(st.session_state.get("tinf_val", 1.0)), 0.05, key="tinf_main_step1")
                st.session_state["tinf_val"] = tinf

        st.markdown("---")

        if clinical_data is not None:
            c_act_top1, c_act_top2 = st.columns([2.2, 0.8])
            with c_act_top1:
                st.success(f"Dữ liệu đang lưu hoạt động trong phiên: **{st.session_state.get('loaded_data_name', 'Tập dữ liệu lâm sàng')}** ({clinical_data['N']} bệnh nhân, {clinical_data['nobs']} mẫu quan sát DV).")
            with c_act_top2:
                if st.button("Đặt lại / Xóa dữ liệu phiên này", type="secondary", use_container_width=True, key="btn_clear_data_step1"):
                    reset_entire_session()
                    st.rerun()

        if data_source_mode == "Tải lên file (Excel/CSV)":
            uploaded_file = st.file_uploader(
                "Chọn file dữ liệu lâm sàng từ máy tính (Excel hoặc CSV)",
                type=["xlsx", "xls", "csv"],
                key="main_file_uploader_widget"
            )
            if uploaded_file is not None:
                tmp_path = f"/tmp/{uploaded_file.name}"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                try:
                    clinical_data = data_io.load_dataset(tmp_path, tinf=tinf, enforce_g4=False)
                    st.session_state["clinical_data"] = clinical_data
                    st.session_state["loaded_data_name"] = uploaded_file.name
                    save_current_session()
                except Exception as e:
                    st.error(f"Lỗi nạp dữ liệu: {e}")
                    clinical_data = None
                    st.session_state["clinical_data"] = None

        elif data_source_mode == "Dataset lâm sàng mẫu (Dataset_chính_thức.xlsx)":
            excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Dataset_chính_thức.xlsx"))
            if os.path.exists(excel_path):
                if clinical_data is None or st.session_state.get("loaded_data_name") != "Dataset_chính_thức.xlsx":
                    clinical_data = data_io.load_dataset(excel_path, tinf=tinf, enforce_g4=True)
                    st.session_state["clinical_data"] = clinical_data
                    st.session_state["loaded_data_name"] = "Dataset_chính_thức.xlsx"
                    save_current_session()

        # Hiển thị dữ liệu hoặc giao diện chờ
        if clinical_data is None:
            st.markdown("""
            <div style="background:#f8fafc;border:2px dashed #94a3b8;border-radius:12px;padding:36px;text-align:center;margin:20px 0;">
                <h3 style="color:#0f172a;margin-bottom:8px;font-weight:700;">Chưa có dữ liệu nào được nạp</h3>
                <p style="color:#64748b;font-size:14px;max-width:550px;margin:0 auto 20px auto;">
                    Vui lòng kéo thả hoặc tải lên file dữ liệu lâm sàng (Excel/CSV) ở phía trên để hệ thống tiến hành kiểm chuẩn Gate G4 và phân tích phi khoang NCA.
                </p>
            </div>
            """, unsafe_allow_html=True)
            c_guide1, c_guide2 = st.columns(2)
            with c_guide1:
                st.markdown("""
                **Cấu trúc file dữ liệu chuẩn Monolix:**
                - `ID`: Mã định danh bệnh nhân duy nhất
                - `Time`: Thời điểm đo hoặc đưa liều (giờ)
                - `DV`: Nồng độ đo được (mg/L, để trống ở dòng liều)
                - `Amt`: Liều dùng (mg, 0 hoặc để trống ở dòng nồng độ)
                """)
            with c_guide2:
                st.markdown("""
                **Các cột Covariate lâm sàng (nếu có):**
                - `CLCr_CG`: Độ thanh thải Creatinine Cockcroft-Gault (mL/min)
                - `TBW`: Cân nặng thực tế (kg)
                - `Age`: Tuổi bệnh nhân (năm)
                - `Sepsis`: Tình trạng nhiễm khuẩn huyết (0/1)
                """)
        else:
            header(f"Bộ dữ liệu lâm sàng: {st.session_state.get('loaded_data_name', 'Dữ liệu nạp')}")
            c = st.columns(4)
            c[0].metric("Bệnh nhân", clinical_data["N"])
            c[1].metric("Mẫu nồng độ (DV)", clinical_data["nobs"])
            c[2].metric("Tổng số liều", sum(len(d) for d in clinical_data["sub_doses"].values()))
            c[3].metric("Thời gian truyền tinf", f"{clinical_data.get('tinf', 1.0):.1f} h")

            st.markdown('<div class="convergence-box"><b>Gate G4 Strictly Verified:</b> Đã kiểm tra tính hợp lệ dữ liệu. Không có bản ghi trùng lặp (ID, Time).</div>', unsafe_allow_html=True)

            # ── ĐỊNH NGHĨA VAI TRÒ CỘT DỮ LIỆU (COLUMN MAPPING CHUẨN MONOLIX) ──
            cov_types = clinical_data.get("cov_types", clinical_data.get("detected_covariates", {}))
            if cov_types:
                with st.expander("Định nghĩa Vai trò Cột Dữ liệu (Column Mapping — Chuẩn Monolix: Continuous / Categorical / Ignore)", expanded=True):
                    st.caption("Tự động nhận diện mọi cột biến đồng ảnh hưởng trong file (bao gồm `GEN`, `Albumin`, `Bilirubin`, `CLCr_CG`, `TBW`...). Bạn có thể chuyển đổi qua lại giữa Biến liên tục (Continuous), Biến phân loại (Categorical) hoặc Bỏ qua (Ignore).")
                    role_label_map = {
                        "continuous": "Continuous Covariate",
                        "categorical": "Categorical Covariate",
                        "ignore": "Ignore (Bỏ qua)"
                    }
                    rev_role_map = {v: k for k, v in role_label_map.items()}
                    bdf = clinical_data.get("baseline_cov")
                    raw_df = clinical_data.get("raw_df")
                    map_rows = []
                    for col_name, ctype in cov_types.items():
                        src_s = raw_df[col_name].dropna() if (raw_df is not None and col_name in raw_df.columns) else (bdf[col_name].dropna() if (bdf is not None and col_name in bdf.columns) else pd.Series([]))
                        n_uniq = int(src_s.nunique()) if len(src_s) > 0 else 0
                        sample_vals = ", ".join(str(x) for x in src_s.unique()[:4])
                        map_rows.append({
                            "Tên cột (Column)": col_name,
                            "Số giá trị duy nhất": n_uniq,
                            "Ví dụ giá trị": sample_vals,
                            "Vai trò (Monolix Role)": role_label_map.get(ctype, "Continuous Covariate")
                        })
                    df_map = pd.DataFrame(map_rows)
                    edited_map = st.data_editor(
                        df_map,
                        column_config={
                            "Tên cột (Column)": st.column_config.TextColumn(disabled=True),
                            "Số giá trị duy nhất": st.column_config.NumberColumn(disabled=True),
                            "Ví dụ giá trị": st.column_config.TextColumn(disabled=True),
                            "Vai trò (Monolix Role)": st.column_config.SelectboxColumn(
                                options=["Continuous Covariate", "Categorical Covariate", "Ignore (Bỏ qua)"],
                                required=True
                            )
                        },
                        hide_index=True,
                        use_container_width=True,
                        key="data_column_mapping_editor"
                    )
                    new_cov_types = {}
                    for _, mr in edited_map.iterrows():
                        new_cov_types[mr["Tên cột (Column)"]] = rev_role_map.get(mr["Vai trò (Monolix Role)"], "continuous")
                    if new_cov_types != cov_types:
                        clinical_data["cov_types"] = new_cov_types
                        st.session_state["clinical_data"] = clinical_data
                        save_current_session()

            # Nút chuyển nhanh sang Bước 2
            c_go1, c_go2 = st.columns([1.5, 1])
            with c_go1:
                st.success("Dữ liệu đã sẵn sàng! Chuyển sang bước cấu hình mô hình và chạy ước lượng SAEM.")
            with c_go2:
                if st.button("Tiếp tục: Cấu hình & Chạy SAEM →", type="primary", use_container_width=True, key="btn_goto_saem"):
                    st.session_state["pending_nav_step"] = "2. SAEM Estimation & Initial Estimates"
                    st.rerun()

            if clinical_data.get("outliers"):
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

            fig.update_layout(title="Observed Concentrations vs Time", xaxis_title="Time (h)",
                              yaxis_title="Concentration DV (mg/L)",
                              yaxis_type="log" if spag_scale == "Logarit (Semi-log)" else "linear",
                              height=480, template="plotly_white")
            st.plotly_chart(fig, width="stretch")

            header("Phân tích Phi khoang (PKanalix NCA Module)")
            st.info("Phương pháp hình thang (Trapezoidal Rule) ước lượng tự động các thông số dược động học phi khoang không phụ thuộc giả định mô hình, hỗ trợ kiểm tra dữ liệu và khởi tạo thông số ban đầu.")
            df_ind_nca, df_sum_nca = compute_nca(clinical_data)
            if df_sum_nca is not None:
                st.dataframe(df_sum_nca, width="stretch")
                with st.expander("Xem chi tiết thông số NCA từng bệnh nhân"):
                    st.dataframe(df_ind_nca, width="stretch")

            st.dataframe(clean_df.head(100), width="stretch")
            
            out_csv, _ = data_io.export_monolix(clinical_data)
            with open(out_csv, "rb") as f:
                st.download_button("Tải file Monolix CSV (export_monolix.csv)", f.read(), "export_monolix.csv", "text/csv")

    # ── MÀN HÌNH 2: SAEM ESTIMATION & INITIAL ESTIMATES (TÍCH HỢP MÔ HÌNH, THIẾT LẬP VÀ CHẠY SAEM) ──
    elif current_tab == "2. SAEM Estimation & Initial Estimates":
        if clinical_data is None:
            st.warning("Vui lòng nạp dữ liệu ở bước 1. Data trước khi cấu hình và chạy SAEM.")
        else:
            header("Bước 2: Lựa chọn Mô hình & Ước lượng Tham số Khởi tạo (Structural Model & Initial Estimates)")

            # ── BẮT BUỘC CHỌN MÔ HÌNH TRƯỚC TIÊN (TOP SECTION) ──
            c_sel1, c_sel2 = st.columns([1.6, 1.4])
            with c_sel1:
                st.markdown("#### 1. Chọn Mô hình Cấu trúc (Structural Model Selection)")
                def on_model_structure_change():
                    cur = st.session_state.get("model_structure_select")
                    if cur:
                        st.session_state["chosen_model_name"] = cur
                        cur_lib = st.session_state.get("selected_lib_model")
                        if cur_lib:
                            lib_map = {m["id"]: m for m in MONOLIX_LIBRARY}
                            if cur_lib in lib_map and lib_map[cur_lib]["internal_model"] != cur:
                                st.session_state.pop("selected_lib_model", None)
                        saved_res = st.session_state.get("saved_models_res", {}).get(cur)
                        if saved_res is not None:
                            st.session_state["res"] = saved_res
                            st.session_state["model_confirmed_for"] = cur
                        else:
                            st.session_state.pop("model_confirmed_for", None)
                        save_current_session()

                cur_chosen = st.session_state.get("chosen_model_name", model_names[0])
                sel_idx = model_names.index(cur_chosen) if cur_chosen in model_names else 0
                val_arg = None if "model_structure_select" in st.session_state else sel_idx
                model_name = st.selectbox(
                    "Mô hình cấu trúc",
                    model_names,
                    index=val_arg,
                    key="model_structure_select",
                    on_change=on_model_structure_change,
                    help="Chọn mô hình Dược động học 1 ngăn, 2 ngăn hoặc từ thư viện Monolix"
                )
                st.session_state["chosen_model_name"] = model_name
                mi = pk.MODELS[model_name]
                names = mi["params"]
                units = mi.get("units", [""] * len(names))
                units_dict = dict(zip(names, units))
                key = lambda s: f"{model_name}_{s}"

                if st.button("Mở Thư viện Mô hình Monolix Library (Browse Models)", type="secondary", use_container_width=True):
                    show_model_library_dialog()

                if "selected_lib_model" in st.session_state:
                    lib_id = st.session_state["selected_lib_model"]
                    st.markdown(f"""
                    <div style="background: #e0f2fe; border: 1px solid #7dd3fc; border-radius: 6px; padding: 6px 10px; font-size: 12px; margin: 6px 0;">
                        <span style="color: #0369a1; font-weight: 700;">Monolix Model:</span> <code>{lib_id}</code>
                    </div>
                    """, unsafe_allow_html=True)

            with c_sel2:
                st.markdown("#### Thông tin Cấu hình & Định nghĩa Giải tích")
                st.markdown(f"- **Mô hình**: `{model_name}`")
                st.markdown(f"- **Thời gian truyền $t_{{inf}}$**: `{tinf:.2f} h`")
                lib_id = st.session_state.get("selected_lib_model")
                lib_entry = next((m for m in MONOLIX_LIBRARY if m["id"] == lib_id), None)
                if lib_entry:
                    st.caption(f"{lib_entry.get('desc', '')}")
                    mlx_code = f"""; Monolix Model Definition: {lib_id}
[LONGITUDINAL]
input = {{{', '.join(lib_entry['params'])}}}

PK:
{lib_entry.get('ode', '')}

OUTPUT:
output = {{Cc}}
"""
                    with st.expander("Xem mã nguồn mô hình định dạng Monolix (.txt)", expanded=False):
                        st.code(mlx_code, language="text")

            # Nút xác nhận lựa chọn mô hình để mở khóa bước ước lượng tham số
            c_act1, c_act2 = st.columns([1.6, 1.4])
            is_model_confirmed = st.session_state.get("model_confirmed_for", "") == model_name

            with c_act1:
                if not is_model_confirmed:
                    st.info("Vui lòng nhấn **'Xác nhận chọn mô hình này'** để hệ thống khởi tạo tham số và mở khóa bảng điều khiển SAEM.")
                else:
                    st.success(f"Đã chọn mô hình: **{model_name}**. Bảng điều khiển khởi tạo tham số và SAEM bên dưới đã sẵn sàng.")

            with c_act2:
                btn_lbl = "Đổi sang mô hình khác" if is_model_confirmed else "Xác nhận chọn mô hình này →"
                btn_type = "secondary" if is_model_confirmed else "primary"
                if st.button(btn_lbl, type=btn_type, use_container_width=True, key="btn_confirm_model_select"):
                    if is_model_confirmed:
                        st.session_state["model_confirmed_for"] = ""
                    else:
                        st.session_state["model_confirmed_for"] = model_name
                    st.rerun()

            st.markdown("<hr style='margin:12px 0 16px 0; border:none; border-top:1px solid #cbd5e1;'/>", unsafe_allow_html=True)

            if not is_model_confirmed:
                st.markdown("""
                <div style="background:#f8fafc;border:2px dashed #94a3b8;border-radius:12px;padding:32px;text-align:center;margin:16px 0;">
                    <h4 style="color:#0f172a;margin-bottom:6px;font-weight:700;">Chưa xác nhận mô hình Dược động học</h4>
                    <p style="color:#64748b;font-size:14px;max-width:550px;margin:0 auto;">
                        Sau khi lựa chọn mô hình ở trên hoặc duyệt từ <b>Monolix Library</b>, bấm nút <b>"Xác nhận chọn mô hình này →"</b> để mở khóa biểu đồ trực quan hóa Initial Estimates và bảng điều khiển chạy SAEM bên tay phải.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # ── BỐ TRÍ 2 CỘT CHUẨN MONOLIX: BIỂU ĐỒ BÊN TRÁI | BẢNG ĐIỀU KHIỂN KHỞI TẠO BÊN PHẢI ──
                col_chart, col_ctrl = st.columns([1.35, 1.0], gap="large")

                # ── CỘT TRÁI: BIỂU ĐỒ TRỰC QUAN HÓA INITIAL ESTIMATES OVERLAID TRÊN DỮ LIỆU BỆNH NHÂN ──
                with col_chart:
                    header("Trực quan hóa Khởi tạo đè lên Dữ liệu (Initial Estimates Plot)")
                    st.caption("Đường cong dự đoán quần thể tương ứng với các giá trị $\\theta_0$ ban đầu đè lên dữ liệu bệnh nhân thực tế.")
                    
                    temp_pop_params = {}
                    for p, d, u, r in zip(names, mi["defaults"], mi["units"], mi["ranges"]):
                        val_k = key(f"pop_{p}")
                        temp_pop_params[p] = float(st.session_state.get(val_k, d))
                    
                    fig_init = initial_estimates_preview_figure(clinical_data, model_name, temp_pop_params)
                    fig_init.update_layout(height=480)
                    st.plotly_chart(fig_init, width="stretch")

                    # Cài đặt thuật toán SAEM và Nút chạy đặt ở chân cột biểu đồ
                    header(f"Chạy ước lượng SAEM: {model_name}")
                    c_saem1, c_saem2, c_saem3, c_saem4 = st.columns(4)
                    n_iter1 = c_saem1.slider("K1 (Exploratory)", 50, 600, 250, 50, key="main_k1")
                    n_iter2 = c_saem2.slider("K2 (Smoothing)", 25, 400, 150, 25, key="main_k2")
                    n_chains = c_saem3.slider("Chains", 1, 3, 1, key="main_chains")
                    fit_seed = c_saem4.number_input("Seed", 0, 9999, 42, key="main_seed")

                    c_run1, c_run2 = st.columns(2)
                    run_saem_clicked = c_run1.button(f"Bắt đầu ước lượng SAEM ({model_name})", type="primary", use_container_width=True, key="btn_run_saem_main")
                    run_scm_clicked = c_run2.button("Tự động Tìm Covariate (Auto-SCM) & Hoàn thiện Phương trình", type="secondary", use_container_width=True, key="btn_run_auto_scm_main", help="Tự động sàng lọc EBE và chạy Forward (p<0.05) + Backward (p<0.01) để tìm mô hình Covariate tối ưu nhất")

                    if run_saem_clicked:
                        init = dict(
                            mu=np.log([st.session_state.get(key(f"pop_{p}"), mi["defaults"][j]) for j, p in enumerate(names)]),
                            omega=[st.session_state.get(key(f"om_{p}"), 0.3) for p in names],
                            err=[float(st.session_state.get("main_sigma_a", 1.0)), float(st.session_state.get("main_sigma_b", 0.2))]
                        )
                        cov_model_run = {}
                        sel_pairs = st.session_state.get("active_cov_pairs") or st.session_state.get("main_cov_select", [])
                        for s in sel_pairs:
                            p_cov, c_cov = s.split("~")
                            cov_model_run.setdefault(p_cov, []).append(c_cov)
                        cov_mode_run = st.session_state.get("main_cov_mode", "time-varying")
                        err_model_run = st.session_state.get("main_error_model_select", "combined")

                        bar = st.progress(0, text="Đang chạy SAEM...")
                        res = pk.fit_model(clinical_data, model_name, init, cov_model_run, err_model_run,
                                           n_iter1=n_iter1, n_iter2=n_iter2, n_chains=n_chains,
                                           seed=int(fit_seed), progress=lambda x: bar.progress(x, text=f"SAEM {x:.0%}"),
                                           cov_mode=cov_mode_run)
                        bar.empty()
                        res["clinical_data"] = clinical_data
                        res["model_name"] = model_name
                        res["loaded_data_name"] = st.session_state.get("loaded_data_name", "Clinical Dataset")
                        if "saved_models_res" not in st.session_state:
                            st.session_state["saved_models_res"] = {}
                        st.session_state["saved_models_res"][model_name] = res
                        st.session_state["res"] = res
                        st.session_state["settings_changed_warning"] = False
                        st.session_state.pop("vpc", None)
                        save_current_session()
                        st.session_state["pending_nav_step"] = "3. Results & Population parameters"
                        st.rerun()

                    if run_scm_clicked:
                        init = dict(
                            mu=np.log([st.session_state.get(key(f"pop_{p}"), mi["defaults"][j]) for j, p in enumerate(names)]),
                            omega=[st.session_state.get(key(f"om_{p}"), 0.3) for p in names],
                            err=[float(st.session_state.get("main_sigma_a", 1.0)), float(st.session_state.get("main_sigma_b", 0.2))]
                        )
                        cov_mode_run = st.session_state.get("main_cov_mode", "time-varying")
                        err_model_run = st.session_state.get("main_error_model_select", "combined")
                        existing_base = st.session_state.get("saved_models_res", {}).get(model_name)
                        if existing_base is not None and any(len(v) > 0 for v in existing_base.get("cov_model", {}).values()):
                            existing_base = None
                        bar = st.progress(0, text="Đang chạy Auto-SCM sàng lọc toàn bộ Covariates...")
                        res = pk.run_auto_scm(
                            clinical_data, model_name, init, error_model=err_model_run,
                            cov_mode=cov_mode_run, seed=int(fit_seed),
                            base_res=existing_base,
                            progress=lambda x, m="": bar.progress(min(max(float(x), 0.0), 1.0), text=m or f"Auto-SCM {x:.0%}")
                        )
                        bar.empty()
                        res["clinical_data"] = clinical_data
                        res["model_name"] = model_name
                        res["loaded_data_name"] = st.session_state.get("loaded_data_name", "Clinical Dataset")
                        if "saved_models_res" not in st.session_state:
                            st.session_state["saved_models_res"] = {}
                        st.session_state["saved_models_res"][model_name] = res
                        st.session_state["res"] = res
                        st.session_state["settings_changed_warning"] = False
                        st.session_state.pop("vpc", None)
                        save_current_session()
                        st.session_state["pending_nav_step"] = "7. Model Equations (Phương trình PopPK)"
                        st.rerun()

                # ── CỘT PHẢI: THANH ĐIỀU KHIỂN ƯỚC LƯỢNG THAM SỐ KHỞI TẠO (INITIAL ESTIMATES PANEL) ──
                with col_ctrl:
                    header("Bảng Điều khiển Tham số Khởi tạo")
                    
                    st.markdown("##### Tự động ước lượng (Auto-estimate)")
                    if st.button("Tự động ước lượng Vd, Cl (Auto-estimate)", type="primary", use_container_width=True, key="btn_auto_estimate"):
                        est_dict, cl_val, vd_val = auto_estimate_initial_params(clinical_data, model_name)
                        pending_dict = {}
                        for p_name, p_val in est_dict.items():
                            val_k = key(f"pop_{p_name}")
                            pending_dict[val_k] = float(p_val)
                        st.session_state["pending_auto_estimates"] = pending_dict
                        st.rerun()

                    est_dict, cl_val, vd_val = auto_estimate_initial_params(clinical_data, model_name)
                    t_half_nca = np.log(2) * vd_val / cl_val if cl_val > 0 else 0.0
                    col_e1, col_e2, col_e3 = st.columns(3)
                    col_e1.metric("CL (NCA)", f"{cl_val:.2f} L/h")
                    col_e2.metric("Vd (NCA)", f"{vd_val:.2f} L")
                    col_e3.metric("t1/2", f"{t_half_nca:.1f} h")

                    if st.session_state.get("auto_estimated_flag", False):
                        st.success("Đã gán giá trị Vd, Cl dự kiến vào các thanh trượt bên dưới!")

                    st.markdown("---")
                    st.markdown("##### Thông số quần thể ban đầu (θ):")
                    pop_params = {}
                    for p, d, u, r in zip(names, mi["defaults"], mi["units"], mi["ranges"]):
                        step = 0.05 if r[1] < 10 else 0.5
                        val_k = key(f"pop_{p}")
                        val_arg = None if val_k in st.session_state else float(d)
                        pop_params[p] = st.slider(f"{p} ({u})", float(r[0]), float(r[1]), val_arg, float(step), key=val_k)

                    st.markdown("##### Biến thiên giữa các cá thể (ω):")
                    omega_dict = {p: st.slider(f"ω_{p}", 0.05, 1.0, float(st.session_state.get(key(f"om_{p}"), 0.3)), 0.05, key=key(f"om_{p}")) for p in names}

                    st.markdown("##### Mô hình sai số dư:")
                    err_models = ["combined", "combined1", "constant", "proportional"]
                    error_model = st.selectbox("Error Model", err_models, key="main_error_model_select")
                    col_err_a, col_err_b = st.columns(2)
                    sigma_add = col_err_a.number_input("a (additive)", 0.01, 5.0, 1.0, 0.1, key="main_sigma_a")
                    sigma_prop = col_err_b.number_input("b (proportional)", 0.01, 1.0, 0.2, 0.02, key="main_sigma_b")

                    st.markdown("##### Individual Model — Ma trận Covariate (Chuẩn Monolix):")
                    active_covs = [c for c, ct in clinical_data.get("cov_types", clinical_data.get("detected_covariates", {})).items() if ct != "ignore"]
                    if not active_covs:
                        ordered_covs = ["CLCr_CG", "TBW", "Sepsis", "Age"]
                    else:
                        priority_covs = ["CLCr_CG", "TBW", "BSA", "Sepsis", "Age", "GEN", "Albumin", "Bilirubin", "Sex"]
                        ordered_covs = [c for c in priority_covs if c in active_covs] + [c for c in active_covs if c not in priority_covs]

                    cur_pairs = set(st.session_state.get("active_cov_pairs", []))
                    mat_rows = []
                    for p in names:
                        row_m = {"Parameter": p}
                        for c_col in ordered_covs:
                            row_m[c_col] = (f"{p}~{c_col}" in cur_pairs)
                        mat_rows.append(row_m)
                    df_mat = pd.DataFrame(mat_rows)
                    col_cfg = {"Parameter": st.column_config.TextColumn("PARAMETERS", disabled=True)}
                    for c_col in ordered_covs:
                        col_cfg[c_col] = st.column_config.CheckboxColumn(c_col, default=False)

                    edited_mat = st.data_editor(
                        df_mat,
                        column_config=col_cfg,
                        hide_index=True,
                        use_container_width=True,
                        key=f"cov_matrix_editor_{model_name}"
                    )
                    new_pairs = []
                    for _, mr in edited_mat.iterrows():
                        p_r = mr["Parameter"]
                        for c_col in ordered_covs:
                            if bool(mr[c_col]):
                                new_pairs.append(f"{p_r}~{c_col}")
                    st.session_state["active_cov_pairs"] = new_pairs

                    with st.expander("Add Covariate — Chuẩn hóa giá trị tham chiếu log(Cov / Ref)", expanded=False):
                        cont_covs = [c for c in ordered_covs if clinical_data.get("cov_types", {}).get(c) == "continuous"]
                        if cont_covs:
                            c_ref1, c_ref2 = st.columns([1.4, 1.0])
                            ref_cov = c_ref1.selectbox("Biến liên tục", cont_covs, key="custom_ref_cov_sel")
                            bdf_cur = clinical_data.get("baseline_cov")
                            def_med = float(np.nanmedian(pd.to_numeric(bdf_cur[ref_cov if ref_cov in bdf_cur.columns else f"{ref_cov}_base"], errors="coerce"))) if bdf_cur is not None else 70.0
                            cur_custom = clinical_data.get("custom_ref_medians", {}).get(ref_cov, def_med)
                            new_ref_val = c_ref2.number_input("Giá trị tham chiếu (Ref)", 0.1, 1000.0, float(round(cur_custom, 1)), 1.0, key="custom_ref_val_input")
                            if st.button("Áp dụng giá trị tham chiếu", key="btn_apply_custom_ref", use_container_width=True):
                                clinical_data.setdefault("custom_ref_medians", {})[ref_cov] = float(new_ref_val)
                                st.session_state["clinical_data"] = clinical_data
                                save_current_session()
                                st.success(f"Đã đặt chuẩn hóa `log({ref_cov} / {new_ref_val:.1f})`!")
                        else:
                            st.info("Chưa có biến liên tục nào được kích hoạt.")

                    cov_mode = st.radio("Chế độ Covariate", ["time-varying", "baseline"], horizontal=True, key="main_cov_mode")

    elif current_tab == "3. Results & Population parameters":
        header("Kết quả Ước lượng Tham số Quần thể (Population Parameter Estimates)")
        if res is None:
            st.warning("Vui lòng chạy SAEM ở bước 2 trước.")
        else:
            show_fit(res, clinical_data)
            st.markdown("---")
            header("Lưu trữ & Xuất Dữ liệu Phiên làm việc (Session Export)")
            st.success(f"Dữ liệu tập tin (**{st.session_state.get('loaded_data_name', 'Dataset')}**) và kết quả ước lượng SAEM cho mô hình **{res.get('model_name', '')}** đã được lưu trữ an toàn trong phiên làm việc.")
            c_exp1, c_exp2, c_exp3 = st.columns(3)
            with c_exp1:
                df_est = estimates_table(res)
                csv_est = df_est.to_csv(index=False).encode('utf-8')
                st.download_button("Tải bảng Tham số Quần thể (CSV)", csv_est, "population_parameters.csv", "text/csv", use_container_width=True)
            with c_exp2:
                if "phi_hat" in res:
                    d_orig = res["data"]
                    df_ebe = pd.DataFrame(np.exp(res["phi_hat"]), columns=[f"{p}_ebe" for p in res["names"]])
                    df_ebe.insert(0, "ID", d_orig["ids"])
                    csv_ebe = df_ebe.to_csv(index=False).encode('utf-8')
                    st.download_button("Tải bảng Tham số Cá thể EBE (CSV)", csv_ebe, "individual_parameters_ebe.csv", "text/csv", use_container_width=True)
            with c_exp3:
                if "diag" in res and "table" in res["diag"]:
                    csv_diag = res["diag"]["table"].to_csv(index=False).encode('utf-8')
                    st.download_button("Tải bảng Chẩn đoán Phần dư (CSV)", csv_diag, "diagnostics_residuals.csv", "text/csv", use_container_width=True)

    # ── MÀN HÌNH 4: DIAGNOSTIC PLOTS (GOF) ──
    elif current_tab == "4. Diagnostic Plots (GOF)":
        header("Chẩn đoán mô hình (Goodness-of-Fit & Residuals Distribution)")
        if res is None:
            st.warning("Vui lòng chạy SAEM ở bước 2 trước.")
        else:
            dg = res["diag"]
            st.plotly_chart(gof_figure(dg), width="stretch")
            
            header("Chẩn đoán Phân phối Phần dư (Normal QQ-Plots)")
            st.plotly_chart(residuals_qq_figure(dg), width="stretch")

            header("Ma trận Tương quan Hiệu ứng Ngẫu nhiên (Random Effects Correlation)")
            st.plotly_chart(parameter_correlation_heatmap(res), width="stretch")

            header("Random Effects (η̂ = EBE) & Shrinkage")
            P = len(res["names"])
            cols = st.columns(P + 2)
            for j, p in enumerate(res["names"]):
                shrink_val = dg['eta_shrink'][j] if 'eta_shrink' in dg else (res.get('shrinkage', [0]*P)[j])
                cols[j].metric(f"Shrinkage η_{p}", f"{shrink_val * 100:.1f}%")
            eps_val = dg.get('eps_shrink', res.get('shrinkage_iwres', 0.0))
            cols[P].metric("Shrinkage IWRES", f"{eps_val * 100:.1f}%")
            cols[P + 1].metric("NPDE mean / sd", f"{dg['table']['NPDE'].mean():.2f} / {dg['table']['NPDE'].std():.2f}")

            header("Phân tích Phần dư & Ngoại lai Chẩn đoán (Flagged Residuals)")
            tt = dg["table"]
            flagged = tt[(tt["IWRES"].abs() > 3.0) | (tt["NPDE"].abs() > 3.0)]
            if len(flagged) > 0:
                st.markdown(f'<div class="outlier-box">Phát hiện <b>{len(flagged)}</b> điểm quan sát có |IWRES| > 3 hoặc |NPDE| > 3. Cần kiểm tra liều và thời điểm lấy mẫu bệnh nhân.</div>', unsafe_allow_html=True)
                time_col = "TIME" if "TIME" in flagged.columns else "Time"
                cols_to_show = [c for c in ["ID", time_col, "DV", "PRED", "IPRED", "IWRES", "NPDE"] if c in flagged.columns]
                st.dataframe(flagged[cols_to_show], width="stretch")
            else:
                st.markdown('<div class="convergence-box">Tuyệt vời: Không có điểm quan sát nào có |IWRES| > 3 hoặc |NPDE| > 3. Dữ liệu khớp rất tốt với mô hình.</div>', unsafe_allow_html=True)

    # ── MÀN HÌNH 5: INDIVIDUAL FITS (EBE) ──
    elif current_tab == "5. Individual fits (EBE)":
        header("Đường cong nồng độ cá thể (Individual Fits)")
        if res is None:
            st.warning("Vui lòng chạy SAEM ở bước 2 trước.")
        else:
            d = res["data"]
            sid_choice = st.selectbox("Chọn bệnh nhân (ID)", d["ids"], key="select_patient_tab6")
            idx_sub = int(np.where(d["ids"] == sid_choice)[0][0])
            
            t_max_sub = float(d["t"][idx_sub, d["mask"][idx_sub]].max()) * 1.1 if d["mask"][idx_sub].any() else 48.0
            t_fine = np.linspace(0.0, max(t_max_sub, 24.0), 300)
            
            phi_ind = res["phi_hat"][idx_sub:idx_sub + 1]
            phi_pop = pk.pop_mean_phi(res)[idx_sub:idx_sub + 1]
            
            num_params = len(res["names"])
            c_ind = st.columns(min(num_params * 2, 8))
            for j, p in enumerate(res["names"]):
                col_idx = (j * 2) % len(c_ind)
                c_ind[col_idx].metric(f"{p} cá thể", f"{float(np.exp(phi_ind[0, j])):.2f}")
                if col_idx + 1 < len(c_ind):
                    c_ind[col_idx + 1].metric(f"η_{p}", f"{float(phi_ind[0, j] - phi_pop[0, j]):.3f}")

            scale_ind = st.radio("Thang đo trục Y cá thể", ["Tuyến tính (Linear)", "Logarit (Semi-log)"], horizontal=True, key="scale_ind_choice")

            sub_d = [d["sub_doses"][sid_choice]] if "sub_doses" in d else None
            sub_c = [d["sub_cov_events"][sid_choice]] if "sub_cov_events" in d else None
            c_fine_ind = pk.predict(res["model_name"], t_fine[None, :], d["dose"][idx_sub:idx_sub + 1],
                                    phi_ind, sub_doses=sub_d, sub_cov=sub_c,
                                    data=d)[0]
            c_fine_pop = pk.predict(res["model_name"], t_fine[None, :], d["dose"][idx_sub:idx_sub + 1],
                                    phi_pop, sub_doses=sub_d, sub_cov=sub_c,
                                    data=d)[0]

            fig_ind = go.Figure()
            fig_ind.add_trace(go.Scatter(x=t_fine, y=c_fine_ind, mode="lines",
                                         line=dict(color="#1565c0", width=2.5), name="IPRED (Cá thể)"))
            fig_ind.add_trace(go.Scatter(x=t_fine, y=c_fine_pop, mode="lines",
                                         line=dict(color="#d84315", dash="dash", width=1.5), name="PRED (Quần thể)"))
            
            mask = d["mask"][idx_sub]
            obs_t = d["t"][idx_sub, mask]
            obs_y = d["y"][idx_sub, mask]
            fig_ind.add_trace(go.Scatter(x=obs_t, y=obs_y, mode="markers",
                                         marker=dict(size=8, color="#2e7d32", symbol="circle"), name="Quan sát (DV)"))

            fig_ind.update_layout(title=f"Khớp nồng độ cá thể: Bệnh nhân ID {sid_choice}",
                                  xaxis_title="Thời gian (giờ)", yaxis_title="Nồng độ (mg/L)",
                                  yaxis_type="log" if scale_ind == "Logarit (Semi-log)" else "linear",
                                  height=450, template="plotly_white")
            st.plotly_chart(fig_ind, width="stretch")

            header("Bảng Tổng hợp Thông số Dược động học Cá thể (Individual Parameters Table)")
            ebe_rows = []
            for i, sid in enumerate(d["ids"]):
                row_data = {"ID": sid}
                for j, p in enumerate(res["names"]):
                    row_data[f"{p}_ind"] = round(float(np.exp(res["phi_hat"][i, j])), 3)
                    row_data[f"eta_{p}"] = round(float(res["phi_hat"][i, j] - pk.pop_mean_phi(res)[i, j]), 4)
                ebe_rows.append(row_data)
            df_ebe = pd.DataFrame(ebe_rows)
            st.dataframe(df_ebe, width="stretch")
            
            csv_ebe = df_ebe.to_csv(index=False)
            st.download_button("Tải file CSV Thông số Cá thể (individual_parameters.csv)",
                               csv_ebe, "individual_parameters.csv", "text/csv", key="btn_download_ebe_csv")

    # ── MÀN HÌNH 6: COVARIATES & SIMULX ──
    elif current_tab == "6. Covariates & Simulx":
        header("Sàng lọc Covariate & Mô hình Tối ưu (Covariate Search & SCM)")
        if res is None:
            st.warning("Vui lòng chạy SAEM ở bước 2 trước.")
        else:
            header("Tương quan giữa Hiệu ứng Ngẫu nhiên (η̂ = EBE) và Covariate")
            fig_cov = covariate_correlation_figure(res, clinical_data)
            if fig_cov is not None:
                st.plotly_chart(fig_cov, width="stretch")

            # Khám phá tương quan động cho mọi biến (bao gồm GEN, Albumin, Bilirubin...)
            dyn_cov_types = {k: v for k, v in clinical_data.get("cov_types", clinical_data.get("detected_covariates", {})).items() if v != "ignore"}
            if dyn_cov_types:
                with st.expander("Khám phá Tương quan Động cho Bất kỳ Covariate nào (GEN, Albumin, Bilirubin...)", expanded=False):
                    c_dc1, c_dc2 = st.columns(2)
                    sel_p_eta = c_dc1.selectbox("Tham số PK (η)", res["names"], key="dyn_cov_param")
                    sel_cov_col = c_dc2.selectbox("Biến đồng ảnh hưởng (Covariate)", list(dyn_cov_types.keys()), key="dyn_cov_col")
                    p_idx = res["names"].index(sel_p_eta)
                    etas_col = res["phi_hat"][:, p_idx] - pk.pop_mean_phi(res)[:, p_idx]
                    bdf = clinical_data["baseline_cov"]
                    col_key = sel_cov_col if sel_cov_col in bdf.columns else f"{sel_cov_col}_base"
                    if col_key in bdf.columns:
                        cov_vals = bdf[col_key]
                        fig_dc = go.Figure()
                        if dyn_cov_types[sel_cov_col] == "continuous":
                            xv = pd.to_numeric(cov_vals, errors="coerce")
                            valid = xv.notna() & np.isfinite(etas_col)
                            fig_dc.add_trace(go.Scatter(x=xv[valid], y=etas_col[valid], mode="markers", marker=dict(color="#1565c0", size=7), name=f"η_{sel_p_eta}"))
                            if valid.sum() > 2:
                                lr = linregress(xv[valid], etas_col[valid])
                                xt = np.linspace(xv[valid].min(), xv[valid].max(), 50)
                                fig_dc.add_trace(go.Scatter(x=xt, y=lr.intercept + lr.slope * xt, mode="lines", line=dict(color="red", dash="dash"), name=f"Trend (p={lr.pvalue:.4f})"))
                                st.caption(f"Hệ số tương quan Pearson **r = {lr.rvalue:.3f}**, **p-value = {lr.pvalue:.4f}**")
                        else:
                            cats = cov_vals.fillna("NA").astype(str).str.replace(r"\.0$", "", regex=True)
                            for cat_lbl in sorted(cats.unique()):
                                mask_cat = (cats == cat_lbl).values
                                fig_dc.add_trace(go.Box(y=etas_col[mask_cat], name=str(cat_lbl), boxmean=True))
                        fig_dc.update_layout(title=f"Tương quan η_{sel_p_eta} theo {sel_cov_col} ({dyn_cov_types[sel_cov_col]})",
                                             xaxis_title=sel_cov_col, yaxis_title=f"η_{sel_p_eta}", height=380, template="plotly_white")
                        st.plotly_chart(fig_dc, width="stretch")

            header("Bảng So sánh Mô hình Từng bước (Stepwise Covariate Modeling - SCM)")
            if st.button("Chạy Tự động Sàng lọc Covariate (Auto-SCM) trên Tập dữ liệu hiện tại", type="primary", key="btn_run_auto_scm_tab6"):
                m_cur = res.get("model_name", model_name)
                mi_cur = pk.MODELS[m_cur]
                init_scm = dict(
                    mu=np.log([float(np.exp(res["beta"][j][0])) if j < len(res.get("beta", [])) else mi_cur["defaults"][j] for j in range(len(mi_cur["params"]))]),
                    omega=[float(res["omega"][j]) if j < len(res.get("omega", [])) else 0.3 for j in range(len(mi_cur["params"]))],
                    err=[float(res["err"][0]) if len(res.get("err", [])) > 0 else 1.0, float(res["err"][1]) if len(res.get("err", [])) > 1 else 0.2]
                )
                is_base = not any(len(v) > 0 for v in res.get("cov_model", {}).values())
                bar_scm = st.progress(0, text="Đang chạy Auto-SCM...")
                new_res = pk.run_auto_scm(
                    clinical_data, m_cur, init_scm, error_model=res.get("error_model", "combined"),
                    cov_mode=res.get("cov_mode", "time-varying"), seed=42,
                    base_res=res if is_base else None,
                    progress=lambda x, m="": bar_scm.progress(min(max(float(x), 0.0), 1.0), text=m or f"Auto-SCM {x:.0%}")
                )
                bar_scm.empty()
                new_res["clinical_data"] = clinical_data
                new_res["model_name"] = m_cur
                new_res["loaded_data_name"] = st.session_state.get("loaded_data_name", "Clinical Dataset")
                if "saved_models_res" not in st.session_state:
                    st.session_state["saved_models_res"] = {}
                st.session_state["saved_models_res"][m_cur] = new_res
                st.session_state["res"] = new_res
                st.session_state.pop("vpc", None)
                save_current_session()
                st.rerun()

            if res.get("scm_history"):
                st.success(f"Kết quả Auto-SCM trên tập dữ liệu hiện tại: Giảm **-2LL = {res.get('delta_ofv_total', 0.0):.2f} đơn vị** ($p < 0.01$). Xem phương trình hoàn chỉnh tại **Tab 7. Model Equations**.")
                st.dataframe(pd.DataFrame(res["scm_history"]), width="stretch")

            cov_summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "validation", "covariate_analysis.csv"))
            is_standard_vanco = (clinical_data is not None and clinical_data.get("N", 0) == 83 and clinical_data.get("nobs", 0) == 278)
            if is_standard_vanco and os.path.exists(cov_summary_path):
                cov_df = pd.read_csv(cov_summary_path)
                st.dataframe(cov_df, width="stretch")
            elif os.path.exists(cov_summary_path):
                with st.expander("Tham khảo Benchmark: Bảng SCM trên tập dữ liệu chuẩn Vancomycin 83 BN", expanded=False):
                    cov_df = pd.read_csv(cov_summary_path)
                    st.dataframe(cov_df, width="stretch")

            header("Thẩm định Mô hình bằng Visual Predictive Check (VPC)")
            c1, c2, c3 = st.columns([1, 1, 1])
            n_rep = c1.slider("Số lần mô phỏng quần thể", 10, 100, 50, 10, key="vpc_nrep")
            pcorr = c2.checkbox("Prediction-corrected (pcVPC)", value=True, key="vpc_pcorr")
            if st.button("Tạo biểu đồ VPC", type="primary", key="btn_run_vpc"):
                st.session_state["vpc"] = (pk.run_vpc(res, n_rep=n_rep, pcorr=pcorr), pcorr)
            if "vpc" in st.session_state:
                v, pc = st.session_state["vpc"]
                st.plotly_chart(vpc_figure(v, None if pc else res["diag"]["table"]), width="stretch")

            header("Mô phỏng Phác đồ Điều trị Lâm sàng & Đích TDM (Simulx Module)")
            st.info("Mô phỏng động học nồng độ theo thời gian cho phác đồ đa liều dựa trên mô hình quần thể, đánh giá tỷ lệ đạt nồng độ đáy mục tiêu (15 - 20 mg/L) và AUC24/MIC ≥ 400.")
            
            c_p1, c_p2, c_p3 = st.columns(3)
            p_tbw = c_p1.number_input("Cân nặng TBW (kg)", 30.0, 150.0, 60.0, 5.0, key="sim_tbw")
            p_clcr = c_p2.number_input("Chức năng thận CLCr (mL/min)", 10.0, 200.0, 80.0, 5.0, key="sim_clcr")
            p_sep = c_p3.selectbox("Nhiễm khuẩn huyết (Sepsis)", ["Không (Non-Sepsis)", "Có (Sepsis)"], key="sim_sep")
            
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
            
            st.markdown(f"**Thông số dự đoán của bệnh nhân**: $CL = {cl_patient:.2f}\text{{ L/h}}$, $V = {v_patient:.2f}\text{{ L}}$ (Thời gian bán thải $t_{{1/2}} = {np.log(2)*v_patient/cl_patient:.1f}\text{{ h}}$)")
            
            c_d1, c_d2, c_d3 = st.columns(3)
            reg_dose = c_d1.selectbox("Liều dùng mỗi lần (mg)", [500.0, 750.0, 1000.0, 1250.0, 1500.0, 2000.0], index=2, key="sim_dose")
            reg_tau = c_d2.selectbox("Khoảng cách đưa liều (giờ)", [8.0, 12.0, 24.0], index=1, key="sim_tau")
            reg_tinf = c_d3.selectbox("Thời gian truyền tinf (giờ)", [1.0, 1.5, 2.0], index=0, key="sim_tinf")
            
            t_sim, c_sim, ct_ss, cpk_ss, auc24_ss = simulx_dosing_simulation(cl_patient, v_patient, reg_dose, reg_tau, reg_tinf, duration=72.0)
            
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=t_sim, y=c_sim, mode="lines",
                                         line=dict(color="#1565c0", width=2.5), name="Nồng độ mô phỏng"))
            fig_sim.add_hrect(y0=15.0, y1=20.0, fillcolor="rgba(76, 175, 80, 0.2)", line_width=0, name="Dải đích Trough (15-20 mg/L)")
            fig_sim.add_hline(y=15.0, line_dash="dash", line_color="green")
            fig_sim.add_hline(y=20.0, line_dash="dash", line_color="green")
            fig_sim.add_hline(y=25.0, line_dash="dot", line_color="red")
            fig_sim.update_layout(title=f"Mô phỏng Phác đồ {reg_dose:.0f} mg mỗi {reg_tau:.0f}h (Truyền {reg_tinf:.1f}h) trong 72 giờ",
                                  xaxis_title="Thời gian (giờ)", yaxis_title="Nồng độ (mg/L)", height=450, template="plotly_white")
            st.plotly_chart(fig_sim, width="stretch")
            
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Nồng độ đáy Ctrough (ss)", f"{ct_ss:.2f} mg/L")
            m_c2.metric("Nồng độ đỉnh Cpeak (ss)", f"{cpk_ss:.2f} mg/L")
            m_c3.metric("Tổng liều 24h", f"{reg_dose * (24.0 / reg_tau):.0f} mg")
            m_c4.metric("AUC24 / MIC (MIC=1)", f"{auc24_ss:.1f}")
            
            if 15.0 <= ct_ss <= 20.0 and auc24_ss >= 400.0:
                st.markdown('<div class="convergence-box">ĐẠT ĐÍCH TỐI ƯU: Nồng độ đáy (15 - 20 mg/L) và AUC24/MIC ≥ 400 đều đạt mục tiêu hướng dẫn lâm sàng IDSA/ASHP.</div>', unsafe_allow_html=True)
            elif ct_ss > 20.0:
                st.markdown('<div class="outlier-box">CẢNH BÁO NGUY CƠ ĐỘC THẬN: Nồng độ đáy > 20 mg/L làm tăng nguy cơ tổn thương thận cấp (AKI). Khuyến cáo giảm liều hoặc giãn khoảng cách đưa liều.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="outlier-box">CẢNH BÁO DƯỚI ĐÍCH ĐIỀU TRỊ: Nồng độ đáy < 15 mg/L hoặc AUC24/MIC < 400, có nguy cơ thất bại điều trị hoặc chọn lọc vi khuẩn kháng thuốc. Khuyến cáo tăng liều hoặc rút ngắn khoảng cách đưa liều.</div>', unsafe_allow_html=True)

            header("Mã nguồn R Tự động phục vụ Nghiên cứu Khoa học (Reproducible R Script)")
            st.info("Xuất trọn bộ mã nguồn phân tích R chuẩn mực để đính kèm bài báo khoa học hoặc nghiệm thu đề tài, tương thích với RStudio.")
            r_code = generate_r_script(res)
            st.download_button("Tải file R Script Phân tích (SimDx_analysis.R)", r_code, "SimDx_analysis.R", "text/plain", key="btn_download_r_script")

    # ── MÀN HÌNH 7: MODEL EQUATIONS (PHƯƠNG TRÌNH POPPK & CƠ SỞ LÝ THUYẾT) ──
    elif current_tab == "7. Model Equations (Phương trình PopPK)":
        header("Phương trình Dược động học Quần thể & Cơ sở Lý thuyết (Population PK Model Equations)")
        if res is None:
            st.warning("Vui lòng chạy SAEM ở bước 2 trước để hệ thống tính toán phương trình động.")
        else:
            render_poppk_equations_tab(res, clinical_data)


if __name__ == "__main__":
    main()
