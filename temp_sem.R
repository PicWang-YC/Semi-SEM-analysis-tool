
library(lavaan)

data <- read.csv("C:/Users/yecan/PycharmProjects/pythonProject3/cfa_data.csv")

model <- '
F2 =~ Q1 + Q2 + Q13 + Q16 + Q19 + Q20
F4 =~ Q3 + Q8 + Q10 + Q15
F1 =~ Q4 + Q5 + Q11 + Q17 + Q21 + Q22 + Q23 + Q24
F3 =~ Q6 + Q12 + Q18 + Q25
'

fit <- cfa(model, data=data, estimator="WLSMV")

fitMeasures(fit, c("cfi","tli","rmsea","srmr"))

summary(fit, fit.measures=TRUE, standardized=TRUE)
