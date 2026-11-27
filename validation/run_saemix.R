# validation/run_saemix.R
# Independent reference model fitting using R package saemix.
# Compares against SimDx Python NLME engine.

suppressPackageStartupMessages({
  if (!requireNamespace("saemix", quietly = TRUE)) {
    stop("Package 'saemix' is not installed. To install, run: install.packages('saemix')")
  }
  library(saemix)
})

# Path to Monolix-exported CSV
csv_path <- file.path(dirname(sys.frame(1)$ofile), "export_monolix.csv")
if (!file.exists(csv_path)) {
  csv_path <- "validation/export_monolix.csv"
}

cat("Loading data from:", csv_path, "\n")
saemix_data <- saemixData(
  name.data = csv_path,
  header = TRUE,
  sep = ",",
  na = ".",
  name.group = "ID",
  name.predictors = c("TIME", "AMT", "TINFUSION"),
  name.response = "Y",
  name.covariates = c("TBW", "CLCr_CG", "Sepsis", "Age")
)

# 1-Compartment IV infusion structural model function
# psi: matrix of individual parameters [N, 2]: CL, V
model_1cpt_infusion <- function(psi, id, xidep) {
  time <- xidep[, 1]
  amt <- xidep[, 2]
  tinf <- xidep[, 3]
  
  cl <- psi[id, 1]
  v <- psi[id, 2]
  k <- cl / v
  
  # Predict concentrations
  ypred <- numeric(length(time))
  # In saemix, evaluation is vectorised per observation
  # For repeated infusions, superposition of previous doses:
  unique_ids <- unique(id)
  for (i in unique_ids) {
    idx <- which(id == i)
    ti <- time[idx]
    cli <- cl[idx][1]
    vi <- v[idx][1]
    ki <- cli / vi
    
    # Dose history for subject i
    d_idx <- idx[!is.na(amt[idx]) & amt[idx] > 0]
    t_doses <- time[d_idx]
    amt_doses <- amt[d_idx]
    dur_doses <- ifelse(!is.na(tinf[d_idx]), tinf[d_idx], 1.0)
    
    ci <- numeric(length(ti))
    for (d in seq_along(t_doses)) {
      td <- t_doses[d]
      ad <- amt_doses[d]
      ti_dur <- dur_doses[d]
      rate <- ad / ti_dur
      
      for (j in seq_along(ti)) {
        t_curr <- ti[j]
        if (t_curr > td) {
          dt <- t_curr - td
          if (dt <= ti_dur) {
            ci[j] <- ci[j] + (rate / cli) * (1.0 - exp(-ki * dt))
          } else {
            ci[j] <- ci[j] + (rate / cli) * (1.0 - exp(-ki * ti_dur)) * exp(-ki * (dt - ti_dur))
          }
        }
      }
    }
    ypred[idx] <- ci
  }
  return(ypred)
}

saemix_model <- saemixModel(
  model = model_1cpt_infusion,
  description = "1-Cpt IV Infusion Multi-dose",
  psi0 = matrix(c(3.0, 25.0), ncol = 2, byrow = TRUE, dimnames = list(NULL, c("CL", "V"))),
  transform.par = c(1, 1), # 1 = log-normal
  error.model = "combined",
  error.init = c(1.0, 0.15)
)

saemix_options <- saemixControl(
  nbiter.saemix = c(250, 150),
  displayProgress = FALSE,
  save.graphs = FALSE,
  seed = 42
)

cat("Running saemix fit...\n")
fit <- saemix(saemix_model, saemix_data, saemix_options)

cat("\n=== SAEMIX ESTIMATION RESULTS ===\n")
print(summary(fit))
cat("-2LL:", fit@results@ll.is, "\n")
