
library(lavaan)

data <- read.csv("C:/Users/yecan/PycharmProjects/pythonProject3/mi_data.csv")

model <- '
F1 =~ Q1 + Q2 + Q4 + Q5 + Q7 + Q9 + Q11 + Q13 + Q16 + Q17 + Q19 + Q20 + Q21 + Q22 + Q23 + Q24
F2 =~ Q3 + Q8 + Q10 + Q14 + Q15
F3 =~ Q6 + Q12 + Q18 + Q25 + Q26
'

fit <- cfa(model, data=data, estimator="WLSMV")

mi <- modindices(fit)
mi <- mi[order(mi$mi, decreasing=TRUE),]

# 只保留残差相关
mi <- subset(mi, op == "~~" & lhs != rhs)

head(mi, 15)
