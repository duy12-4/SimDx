"""
Monolix 2024R1 Integration Engine for SimDx
Enables direct headless execution and head-to-head benchmarking with local Monolix installation.
"""

import os
import subprocess
import pandas as pd
import numpy as np

MONOLIX_APP_PATH = "/Applications/MonolixSuite2024R1.app"
MONOLIX_BIN_PATH = "/Applications/MonolixSuite2024R1.app/Contents/Resources/monolixSuite/lib/monolix"
MONOLIX_LIB_PATH = "/Applications/MonolixSuite2024R1.app/Contents/Resources/monolixSuite/lib"

def is_monolix_installed():
    """Kiểm tra phần mềm Monolix có được cài đặt trên máy người dùng hay không."""
    return os.path.exists(MONOLIX_BIN_PATH)

def run_monolix_project(project_path, timeout=120):
    """
    Chạy dự án Monolix ở chế độ dòng lệnh không cần GUI (--no-gui).
    Trả về: (success: bool, stdout: str, stderr: str)
    """
    if not is_monolix_installed():
        return False, "", "Monolix 2024R1 không được tìm thấy tại /Applications/MonolixSuite2024R1.app"
    
    if not os.path.exists(project_path):
        return False, "", f"File dự án Monolix không tồn tại: {project_path}"

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = f"{env.get('DYLD_LIBRARY_PATH', '')}:{MONOLIX_LIB_PATH}"
    env["DYLD_FRAMEWORK_PATH"] = f"{env.get('DYLD_FRAMEWORK_PATH', '')}:{MONOLIX_LIB_PATH}"

    cmd = [
        MONOLIX_BIN_PATH,
        "--no-gui",
        "--nosplash",
        "-p", project_path
    ]

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return (proc.returncode == 0), proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Monolix chạy quá thời gian chờ (Timeout)."
    except Exception as e:
        return False, "", str(e)

def parse_monolix_results(export_dir):
    """
    Đọc toàn bộ kết quả xuất ra từ thư mục export của Monolix:
    - populationParameters.txt
    - LogLikelihood/logLikelihood.txt
    - IndividualParameters/estimatedIndividualParameters.txt
    """
    results = {}
    
    # 1. Population parameters
    pop_path = os.path.join(export_dir, "populationParameters.txt")
    if os.path.exists(pop_path):
        try:
            results["pop_params"] = pd.read_csv(pop_path)
        except Exception:
            results["pop_params"] = None
    else:
        results["pop_params"] = None

    # 2. Log-likelihood
    ll_path = os.path.join(export_dir, "LogLikelihood", "logLikelihood.txt")
    if os.path.exists(ll_path):
        try:
            df_ll = pd.read_csv(ll_path)
            ll_dict = dict(zip(df_ll["criteria"], df_ll["importanceSampling"]))
            results["log_likelihood"] = ll_dict
        except Exception:
            results["log_likelihood"] = None
    else:
        results["log_likelihood"] = None

    # 3. Individual parameters
    ind_path = os.path.join(export_dir, "IndividualParameters", "estimatedIndividualParameters.txt")
    if os.path.exists(ind_path):
        try:
            results["ind_params"] = pd.read_csv(ind_path)
        except Exception:
            results["ind_params"] = None
    else:
        results["ind_params"] = None

    return results

def build_benchmark_comparison(simdx_res, monolix_export_dir):
    """
    So sánh đối đầu trực diện kết quả giữa SimDx và Monolix:
    Trả về DataFrame gồm: Tham số, Monolix, SimDx, Sai lệch (%), Đánh giá.
    """
    m_res = parse_monolix_results(monolix_export_dir)
    if m_res["pop_params"] is None:
        return None, None

    df_m = m_res["pop_params"]
    m_map = {}
    for _, row in df_m.iterrows():
        p_name = row["parameter"]
        val = float(row["value"])
        se = float(row["se_sa"]) if pd.notnull(row.get("se_sa")) else np.nan
        rse = float(row["rse_sa"]) if pd.notnull(row.get("rse_sa")) else np.nan
        m_map[p_name] = (val, se, rse)

    rows = []
    # Fixed effects
    names = simdx_res["names"]
    theta_vals = np.exp([b[0] for b in simdx_res["beta"]])
    fim = simdx_res.get("fim", {})

    for p, val in zip(names, theta_vals):
        m_keys = [f"{p}_pop", f"{p}"]
        m_entry = None
        for mk in m_keys:
            if mk in m_map:
                m_entry = m_map[mk]
                break
        
        m_val = m_entry[0] if m_entry else np.nan
        m_se = m_entry[1] if m_entry else np.nan
        s_se = fim.get(p, (np.nan, np.nan))[1]

        diff_pct = abs(val - m_val) / max(abs(m_val), 1e-9) * 100 if np.isfinite(m_val) else np.nan
        if np.isnan(diff_pct):
            status = "—"
        elif diff_pct < 5.0:
            status = "🟢 Trùng khớp xuất sắc (< 5%)"
        elif diff_pct < 15.0:
            status = "🟡 Rất sát (< 15%)"
        elif diff_pct < 30.0:
            status = "🟠 Tương đồng (< 30%)"
        else:
            status = "🔴 Phân nhánh nghiệm (> 30%)"

        rows.append({
            "Tham số": f"θ_{p}",
            "Monolix 2024R1": f"{m_val:.4f}" if np.isfinite(m_val) else "—",
            "SimDx": f"{val:.4f}",
            "Monolix SE": f"{m_se:.4f}" if np.isfinite(m_se) and m_se < 1e5 else ("Inf" if m_se >= 1e5 else "—"),
            "SimDx SE": f"{s_se:.4f}" if np.isfinite(s_se) and s_se < 1e5 else ("Inf" if s_se >= 1e5 else "—"),
            "Sai lệch (%)": f"{diff_pct:.1f}%" if np.isfinite(diff_pct) else "—",
            "Đánh giá": status
        })

    # Random effects (omega)
    omega_vals = simdx_res["omega"]
    for p, val in zip(names, omega_vals):
        mk = f"omega_{p}"
        m_entry = m_map.get(mk)
        m_val = m_entry[0] if m_entry else np.nan
        m_se = m_entry[1] if m_entry else np.nan
        s_se = fim.get(mk, (np.nan, np.nan))[1]

        diff_pct = abs(val - m_val) / max(abs(m_val), 1e-9) * 100 if np.isfinite(m_val) else np.nan
        if np.isnan(diff_pct):
            status = "—"
        elif diff_pct < 10.0:
            status = "🟢 Trùng khớp xuất sắc (< 10%)"
        elif diff_pct < 25.0:
            status = "🟡 Rất sát (< 25%)"
        else:
            status = "🟠 Chấp nhận được"

        rows.append({
            "Tham số": f"ω_{p}",
            "Monolix 2024R1": f"{m_val:.4f}" if np.isfinite(m_val) else "—",
            "SimDx": f"{val:.4f}",
            "Monolix SE": f"{m_se:.4f}" if np.isfinite(m_se) and m_se < 1e5 else ("Inf" if m_se >= 1e5 else "—"),
            "SimDx SE": f"{s_se:.4f}" if np.isfinite(s_se) and s_se < 1e5 else ("Inf" if s_se >= 1e5 else "—"),
            "Sai lệch (%)": f"{diff_pct:.1f}%" if np.isfinite(diff_pct) else "—",
            "Đánh giá": status
        })

    # Error model
    fa, fb = simdx_res["err"][0], simdx_res["err"][1]
    for lab, val in [("a", fa), ("b", fb)]:
        m_entry = m_map.get(lab)
        m_val = m_entry[0] if m_entry else np.nan
        m_se = m_entry[1] if m_entry else np.nan
        s_se = fim.get(lab, (np.nan, np.nan))[1]
        diff_pct = abs(val - m_val) / max(abs(m_val), 1e-9) * 100 if np.isfinite(m_val) else np.nan
        if np.isnan(diff_pct):
            status = "—"
        elif diff_pct < 10.0:
            status = "🟢 Trùng khớp xuất sắc (< 10%)"
        elif diff_pct < 25.0:
            status = "🟡 Rất sát (< 25%)"
        else:
            status = "🟠 Chấp nhận được"

        rows.append({
            "Tham số": f"Sai số {lab}",
            "Monolix 2024R1": f"{m_val:.4f}" if np.isfinite(m_val) else "—",
            "SimDx": f"{val:.4f}",
            "Monolix SE": f"{m_se:.4f}" if np.isfinite(m_se) and m_se < 1e5 else ("Inf" if m_se >= 1e5 else "—"),
            "SimDx SE": f"{s_se:.4f}" if np.isfinite(s_se) and s_se < 1e5 else ("Inf" if s_se >= 1e5 else "—"),
            "Sai lệch (%)": f"{diff_pct:.1f}%" if np.isfinite(diff_pct) else "—",
            "Đánh giá": status
        })

    df_comp = pd.DataFrame(rows)

    # Information criteria comparison
    ic_rows = []
    if m_res["log_likelihood"]:
        m_ll = m_res["log_likelihood"]
        s_ic = simdx_res.get("ic", {})
        ic_rows = [
            {
                "Tiêu chuẩn": "−2LL (OFV)",
                "Monolix 2024R1": f"{float(m_ll.get('OFV', np.nan)):.2f}",
                "SimDx": f"{s_ic.get('m2ll', np.nan):.2f}",
                "Chênh lệch": f"{abs(float(m_ll.get('OFV', 0)) - s_ic.get('m2ll', 0)):.2f}"
            },
            {
                "Tiêu chuẩn": "AIC",
                "Monolix 2024R1": f"{float(m_ll.get('AIC', np.nan)):.2f}",
                "SimDx": f"{s_ic.get('aic', np.nan):.2f}",
                "Chênh lệch": f"{abs(float(m_ll.get('AIC', 0)) - s_ic.get('aic', 0)):.2f}"
            },
            {
                "Tiêu chuẩn": "BIC",
                "Monolix 2024R1": f"{float(m_ll.get('BIC', np.nan)):.2f}",
                "SimDx": f"{s_ic.get('bic', np.nan):.2f}",
                "Chênh lệch": f"{abs(float(m_ll.get('BIC', 0)) - s_ic.get('bic', 0)):.2f}"
            }
        ]
    df_ic = pd.DataFrame(ic_rows) if ic_rows else None

    return df_comp, df_ic
