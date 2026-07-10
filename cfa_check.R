
library(lavaan)
data_raw <- read.csv("C:/Users/yecan/PycharmProjects/pythonProject3/sem_data.csv", check.names = FALSE)
data <- data_raw[, sapply(data_raw, is.numeric), drop = FALSE]

model <- '
Cardiac_Variability =~ 1*AvgSdnn + AvgPnn20 + AvgPnn50 + StdHR + StdHRV
'

estimator_name <- "MLR"

if (estimator_name == "GLS") {
  data <- na.omit(data)
  fit <- cfa(model, data = data, estimator = estimator_name, std.lv = FALSE)
} else {
  fit <- cfa(model, data = data, estimator = estimator_name, std.lv = FALSE, missing = "fiml")
}

cat("MODEL\n")
cat(model)
cat("\n\nESTIMATOR\n")
cat(estimator_name)
cat("\nN_USED\n")
cat(lavInspect(fit, "nobs"))
cat("\n\nFIT_MEASURES\n")
print(round(fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr")), 4))
cat("\n\nSTANDARDIZED_LOADINGS\n")
std <- standardizedSolution(fit)
print(std[std$op == "=~", c("lhs", "rhs", "est.std", "pvalue")])
cat("\n\nSUMMARY\n")
print(summary(fit, fit.measures = TRUE, standardized = TRUE))
