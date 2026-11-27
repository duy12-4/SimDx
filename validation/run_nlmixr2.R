# validation/run_nlmixr2.R
# Independent reference model fitting using R package nlmixr2.
# Compares against SimDx Python NLME engine.

suppressPackageStartupMessages({
  if (!requireNamespace("nlmixr2", quietly = TRUE)) {
    stop("Package 'nlmixr2' is not installed. To install, run: install.packages('nlmixr2')")
  }
  library(nlmixr2)
})

csv_path <- file.path(dirname(sys.frame(1)$ofile), "export_monolix.csv")
if (!file.exists(csv_path)) {
  csv_path <- "validation/export_monolix.csv"
}

cat("Loading data from:", csv_path, "\n")
df <- read.csv(csv_path, stringsAsFactors = FALSE)
# Standardize columns for nlmixr2: ID, TIME, AMT, DV, EVID
df$DV <- suppressWarnings(as.numeric(df$Y))
df$AMT <- suppressWarnings(as.numeric(df$AMT))
df$RATE <- ifelse(!is.na(df$AMT) & df$AMT > 0, df$AMT / df$TINFUSION, 0)

# Define 1-Compartment IV infusion model in nlmixr2
model_1cpt <- function() {
  ini({
    tcl <- log(3.0)
    tv <- log(25.0)
    eta.cl ~ 0.3^2
    eta.v ~ 0.25^2
    add.err <- 1.0
    prop.err <- 0.15
  })
  model({
    cl <- exp(tcl + eta.cl)
    v <- exp(tv + eta.v)
    d/dt(central) <- - (cl / v) * central
    cp <- central / v
    cp ~ add(add.err) + prop(prop.err)
  })
}

cat("Running nlmixr2 SAEM fit...\n")
fit <- nlmixr2(model_1cpt, df, est = "saem", control = saemControl(nBurn = 250, nEm = 150, seed = 42))

cat("\n=== NLMIXR2 ESTIMATION RESULTS ===\n")
print(fit)
