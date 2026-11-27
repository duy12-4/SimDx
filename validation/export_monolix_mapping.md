# Monolix Data Column Mapping Documentation

File: `validation/export_monolix.csv`
Generated from clinical dataset `Dataset_chính_thức.xlsx`
Total Rows: 565
Subjects: 83
Doses (EVID=1): 287
Observations (EVID=0): 278

## Column Mapping Table

| Monolix Column | Dataset Column | Role / Type | Description / Units |
| :--- | :--- | :--- | :--- |
| **ID** | `ID` | ID | Unique subject identifier (83 subjects: 1..83) |
| **TIME** | `Time` | TIME | Time since first dose (hours) |
| **AMT** | `Amount` | AMOUNT | Dose amount (mg). '.' on observation rows |
| **TINFUSION** | Generated | TINF | Infusion duration (default 1.0 h). '.' on obs rows |
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
