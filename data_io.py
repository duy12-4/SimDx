"""
data_io.py - Data loader, validation, and Monolix export for clinical datasets.

Handles repeated IV infusions, sparse observations, and time-varying covariates.
Enforces Gate G4: 565 rows / 83 subjects / 287 doses / 278 observations.
Flags outliers without deleting or modifying them.
"""
import os
import sys
import numpy as np
import pandas as pd


OUTLIER_CRITERIA = [
    {"id": 76, "time_approx": 51.5, "dv_min": 100.0, "reason": "Extremely high post-infusion peak (124.9 mg/L at 51.5 h)"},
    {"id": 68, "time_approx": 123.5, "dv_min": 100.0, "reason": "Extremely high post-infusion peak (124.0 mg/L at 123.5 h)"},
    {"id": 10, "time_approx": 38.2, "dv_min": 80.0, "reason": "High post-infusion peak (84.0 mg/L at 38.2 h)"},
]


def load_dataset(path, tinf=1.0, sheet_name=None, enforce_g4=True):
    """
    Load and validate clinical PK dataset.
    
    Parameters:
    -----------
    path : str
        Path to .xlsx or .csv file.
    tinf : float
        Default infusion duration in hours (default: 1.0 h).
    sheet_name : str or int, optional
        Sheet name if Excel (defaults to 'Dataset' if present, else 0).
    enforce_g4 : bool
        If True, enforces Gate G4 exact row and count checks.
        
    Returns:
    --------
    dict containing:
        - raw_df: Original DataFrame
        - clean_df: DataFrame with numeric parsed values and EVID
        - ids: 1D array of subject IDs [N]
        - N: Number of subjects (83)
        - T: Max number of observations per subject
        - t: Observation times matrix [N, T]
        - y: DV observation matrix [N, T]
        - mask: Valid observation mask [N, T]
        - sub_doses: dict mapping subject ID -> list of (t_dose, amt, tinf)
        - sub_cov_events: dict mapping subject ID -> dict of time-varying covariates
        - baseline_cov: DataFrame of baseline covariates per subject
        - outliers: list of flagged outlier records
        - anomaly_report: string summary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    # 1. Read file
    if path.endswith(".xlsx") or path.endswith(".xls"):
        xl = pd.ExcelFile(path)
        target_sheet = sheet_name
        if target_sheet is None:
            target_sheet = "Dataset" if "Dataset" in xl.sheet_names else 0
        df = pd.read_excel(path, sheet_name=target_sheet)
    else:
        df = pd.read_csv(path)

    # Strip column names
    df.columns = [str(c).strip() for c in df.columns]

    # Deduplicate exact identical rows (e.g. accidental double entry)
    n_raw_rows = len(df)
    df = df.drop_duplicates()
    n_exact_dups = n_raw_rows - len(df)

    # Required core columns
    req_cols = ["ID", "Time", "Amount", "DV"]
    for col in req_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in dataset. Found: {list(df.columns)}")

    # Check duplicate (ID, Time)
    duplicates = df[df.duplicated(subset=["ID", "Time"], keep=False)]
    if len(duplicates) > 0:
        dup_info = duplicates[["ID", "Time"]].drop_duplicates().to_dict(orient="records")
        raise ValueError(f"CRITICAL: Found duplicate (ID, Time) entries in dataset: {dup_info}")

    # Standardize '.' to NaN
    parsed_df = df.copy()
    for col in parsed_df.columns:
        parsed_df[col] = parsed_df[col].replace(".", np.nan)

    # Convert numeric columns
    numeric_cols = ["ID", "Time", "Amount", "DV", "TBW", "Height", "BMI", "IBW",
                    "AdBW", "Dosing weight", "BSA", "Creatinine", "CLCr_CG",
                    "eGFR_CKD_EPI", "CLCr_Je", "Sepsis", "Sex", "Age"]
    for col in numeric_cols:
        if col in parsed_df.columns:
            parsed_df[col] = pd.to_numeric(parsed_df[col], errors="coerce")

    # Classify doses and observations
    is_dose = parsed_df["Amount"].notna() & (parsed_df["Amount"] > 0)
    is_obs = parsed_df["DV"].notna()

    n_rows = len(parsed_df)
    n_subjects = parsed_df["ID"].nunique()
    n_doses = int(is_dose.sum())
    n_obs = int(is_obs.sum())

    # Gate G4 Validation
    if enforce_g4:
        # Check against reference counts
        errs = []
        if n_rows != 565:
            errs.append(f"Row count mismatch: expected 565, got {n_rows}")
        if n_subjects != 83:
            errs.append(f"Subject count mismatch: expected 83, got {n_subjects}")
        if n_doses != 287:
            errs.append(f"Dose count mismatch: expected 287, got {n_doses}")
        if n_obs != 278:
            errs.append(f"Observation count mismatch: expected 278, got {n_obs}")

        if errs:
            raise ValueError("Gate G4 Verification Failed loudly:\n" + "\n".join(errs))

    # Check for invalid DV <= 0
    neg_obs = parsed_df[is_obs & (parsed_df["DV"] <= 0)]
    if len(neg_obs) > 0:
        raise ValueError(f"Found non-positive DV observations:\n{neg_obs[['ID', 'Time', 'DV']]}")

    # Flag outliers (NEVER drop or modify)
    outliers = []
    for crit in OUTLIER_CRITERIA:
        sid = crit["id"]
        matches = parsed_df[(parsed_df["ID"] == sid) & is_obs & (parsed_df["DV"] >= crit["dv_min"])]
        for _, row in matches.iterrows():
            outliers.append({
                "ID": int(row["ID"]),
                "Time": float(row["Time"]),
                "DV": float(row["DV"]),
                "Reason": crit["reason"],
                "Action": "FLAGGED (retained in dataset without modification)",
            })

    # Build per-subject event histories
    unique_ids = np.sort(parsed_df["ID"].unique())
    sub_doses = {}
    sub_cov_events = {}
    baseline_records = []

    time_varying_cols = [c for c in ["Creatinine", "CLCr_CG", "eGFR_CKD_EPI", "CLCr_Je"] if c in parsed_df.columns]
    baseline_cols = [c for c in ["TBW", "Height", "BMI", "IBW", "AdBW", "Dosing weight", "BSA", "Sepsis", "Sex", "Age"] if c in parsed_df.columns]

    subject_obs = []
    for sid in unique_ids:
        sdf = parsed_df[parsed_df["ID"] == sid].sort_values("Time")
        
        # Doses
        d_rows = sdf[sdf["Amount"].notna() & (sdf["Amount"] > 0)]
        d_list = [(float(r["Time"]), float(r["Amount"]), float(tinf)) for _, r in d_rows.iterrows()]
        sub_doses[sid] = d_list

        # Observations
        o_rows = sdf[sdf["DV"].notna()]
        subject_obs.append(o_rows)

        # Time-varying covariates
        sub_cov_events[sid] = {}
        for col in time_varying_cols:
            c_rows = sdf[sdf[col].notna()]
            events = [(float(r["Time"]), float(r[col])) for _, r in c_rows.iterrows()]
            sub_cov_events[sid][col] = events

        # Baseline covariates (from first record)
        first_row = sdf.iloc[0]
        base_dict = {"ID": sid}
        for col in baseline_cols:
            base_dict[col] = first_row[col]
        # Also baseline for time-varying
        for col in time_varying_cols:
            base_dict[f"{col}_base"] = first_row[col]
        baseline_records.append(base_dict)

    baseline_cov_df = pd.DataFrame(baseline_records)

    # Build padded matrices for NLME/SAEM
    N = len(unique_ids)
    T = max(len(o) for o in subject_obs)
    t_mat = np.zeros((N, T), dtype=float)
    y_mat = np.zeros((N, T), dtype=float)
    mask_mat = np.zeros((N, T), dtype=bool)

    for i, o_df in enumerate(subject_obs):
        k = len(o_df)
        if k > 0:
            times = o_df["Time"].values
            dvs = o_df["DV"].values
            t_mat[i, :k] = times
            y_mat[i, :k] = dvs
            mask_mat[i, :k] = True
            t_mat[i, k:] = times[-1]  # pad with last valid time

    # Generate Anomaly Report
    report_lines = [
        "=== CLINICAL DATASET ANOMALY & VALIDATION REPORT ===",
        f"File: {os.path.basename(path)}",
        f"Total rows: {n_rows}",
        f"Total subjects: {n_subjects}",
        f"Total doses (AMT present): {n_doses}",
        f"Total observations (DV present): {n_obs}",
        f"Configured infusion duration (tinf): {tinf:.2f} h",
        f"Observation range: {parsed_df[is_obs]['DV'].min():.2f} - {parsed_df[is_obs]['DV'].max():.2f} mg/L",
        f"Duplicate (ID, Time): 0 (Verified)",
        f"Negative or zero DV: 0 (Verified)",
    ]
    if n_exact_dups > 0:
        report_lines.append(f"Auto-cleaned identical rows: {n_exact_dups} duplicate row(s) removed.")
    report_lines.append(f"Flagged Outlier Observations ({len(outliers)}):")
    for o in outliers:
        report_lines.append(f"  - ID {o['ID']}: DV={o['DV']} at {o['Time']} h -> {o['Reason']} [{o['Action']}]")

    anomaly_report = "\n".join(report_lines)

    return {
        "raw_df": df,
        "clean_df": parsed_df,
        "ids": unique_ids,
        "N": N,
        "T": T,
        "t": t_mat,
        "y": y_mat,
        "mask": mask_mat,
        "nobs": n_obs,
        "sub_doses": sub_doses,
        "sub_cov_events": sub_cov_events,
        "baseline_cov": baseline_cov_df,
        "outliers": outliers,
        "anomaly_report": anomaly_report,
        "tinf": tinf,
    }


def export_monolix(loaded_data, out_path="validation/export_monolix.csv"):
    """
    Export loaded dataset to Monolix-compliant CSV format with EVID.
    
    Columns:
    ID, TIME, AMT, TINFUSION, Y, EVID, TBW, Height, BMI, IBW, AdBW, Dosing_weight,
    BSA, Creatinine, CLCr_CG, eGFR_CKD_EPI, CLCr_Je, Sepsis, Sex, Age
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    clean_df = loaded_data["clean_df"].copy()
    tinf = loaded_data.get("tinf", 1.0)

    rows = []
    cov_cols = ["TBW", "Height", "BMI", "IBW", "AdBW", "Dosing weight", "BSA",
                "Creatinine", "CLCr_CG", "eGFR_CKD_EPI", "CLCr_Je", "Sepsis", "Sex", "Age"]

    for _, r in clean_df.iterrows():
        sid = int(r["ID"])
        t = float(r["Time"])
        amt = r["Amount"]
        dv = r["DV"]
        
        is_dose = pd.notna(amt) and float(amt) > 0
        is_obs = pd.notna(dv)

        row_dict = {
            "ID": sid,
            "TIME": t,
            "AMT": float(amt) if is_dose else ".",
            "TINFUSION": float(tinf) if is_dose else ".",
            "Y": float(dv) if is_obs else ".",
            "EVID": 1 if is_dose else 0,
        }

        for c in cov_cols:
            col_key = "Dosing_weight" if c == "Dosing weight" else c
            val = r.get(c, np.nan)
            row_dict[col_key] = val if pd.notna(val) else "."

        rows.append(row_dict)

    monolix_df = pd.DataFrame(rows)
    # Sort by ID, TIME, -EVID (dose first if simultaneous)
    monolix_df["sort_evid"] = monolix_df["EVID"]
    monolix_df = monolix_df.sort_values(by=["ID", "TIME", "sort_evid"], ascending=[True, True, False])
    monolix_df = monolix_df.drop(columns=["sort_evid"])

    monolix_df.to_csv(out_path, index=False)

    # Also generate column mapping documentation
    mapping_doc = f"""# Monolix Data Column Mapping Documentation

File: `{out_path}`
Generated from clinical dataset `Dataset_chính_thức.xlsx`
Total Rows: {len(monolix_df)}
Subjects: {monolix_df['ID'].nunique()}
Doses (EVID=1): {(monolix_df['EVID'] == 1).sum()}
Observations (EVID=0): {(monolix_df['EVID'] == 0).sum()}

## Column Mapping Table

| Monolix Column | Dataset Column | Role / Type | Description / Units |
| :--- | :--- | :--- | :--- |
| **ID** | `ID` | ID | Unique subject identifier (83 subjects: 1..83) |
| **TIME** | `Time` | TIME | Time since first dose (hours) |
| **AMT** | `Amount` | AMOUNT | Dose amount (mg). '.' on observation rows |
| **TINFUSION** | Generated | TINF | Infusion duration (default {tinf:.1f} h). '.' on obs rows |
| **Y** | `DV` | OBSERVATION | Measured concentration (mg/L). '.' on dose rows |
| **EVID** | Generated | EVENT ID | 1 = Dose event, 0 = Observation event |
| **TBW** | `TBW` | COVARIATE | Total Body Weight (kg) |
| **Height** | `Height` | COVARIATE | Height (cm) |
| **BMI** | `BMI` | COVARIATE | Body Mass Index (kg/m²) |
| **IBW** | `IBW` | COVARIATE | Ideal Body Weight (kg) |
| **AdBW** | `AdBW` | COVARIATE | Adjusted Body Weight (kg) |
| **Dosing_weight** | `Dosing weight`| COVARIATE | Dosing weight (kg) |
| **BSA** | `BSA` | COVARIATE | Body Surface Area (m²) |
| **Creatinine** | `Creatinine` | TIME-VARYING COVARIATE | Serum Creatinine (µmol/L) |
| **CLCr_CG** | `CLCr_CG` | TIME-VARYING COVARIATE | Cockcroft-Gault Creatinine Clearance (mL/min) |
| **eGFR_CKD_EPI** | `eGFR_CKD_EPI`| TIME-VARYING COVARIATE | eGFR CKD-EPI formula (mL/min/1.73m²) |
| **CLCr_Je** | `CLCr_Je` | TIME-VARYING COVARIATE | Jelliffe Creatinine Clearance (mL/min) |
| **Sepsis** | `Sepsis` | CATEGORICAL COVARIATE | Sepsis diagnosis (0 = No, 1 = Yes) |
| **Sex** | `Sex` | CATEGORICAL COVARIATE | Sex (0 = Male/Female, 1 = Female/Male [uncertain]) |
| **Age** | `Age` | COVARIATE | Age in years |

## Rules Followed
1. Missing values denoted by '.' as standard in Monolix / NONMEM.
2. Doses and observations at the same timestamp are ordered with doses first (`EVID=1` before `EVID=0`).
3. No observations dropped, modified or winsorized.
"""
    doc_path = os.path.splitext(out_path)[0] + "_mapping.md"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(mapping_doc)

    return out_path, doc_path


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "Dataset_chính_thức.xlsx"
    loaded = load_dataset(data_path, tinf=1.0)
    print(loaded["anomaly_report"])
    csv_path, map_path = export_monolix(loaded)
    print(f"\nExported Monolix CSV: {csv_path}")
    print(f"Exported Monolix Mapping: {map_path}")
