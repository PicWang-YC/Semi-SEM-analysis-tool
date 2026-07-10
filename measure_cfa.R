
library(lavaan)
data <- read.csv("C:/Users/yecan/PycharmProjects/pythonProject3/measure_cfa_data.csv", check.names=FALSE)
model <- '
Usability =~ before_after_1 + before_after_2 + driving_1 + gamification_1 + gamification_2 + gamification_3 + gamification_4
Acceptance =~ Acceptance_P1 + Acceptance_P2 + Acceptance_P3
Satisfaction =~ before_after_10 + driving_after_4 + gamification_12
Continuance =~ before_after_7 + before_after_8 + driving_after_1 + gamification_10 + gamification_11 + continue_q1 + continue_q2
Effectiveness =~ effect_1 + effect_2 + effect_3 + effect_4 + effect_5 + effect_6 + effect_7 + driving_after_2 + driving_after_3 + gamification_6 + comfort_3
'
fit <- cfa(model, data=data, estimator="WLSMV", ordered=names(data))
cat("FIT\n")
print(round(fitMeasures(fit, c("cfi","tli","rmsea","srmr")), 4))
cat("\nSUMMARY\n")
print(summary(fit, fit.measures=TRUE, standardized=TRUE))
cat("\nMODIFICATION_INDICES\n")
mi <- modindices(fit, sort.=TRUE, maximum.number=30)
print(mi)
