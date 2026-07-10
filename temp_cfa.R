
library(lavaan)

data <- read.csv("C:/Users/yecan/PycharmProjects/pythonProject3/mi_data.csv")

model <- '
F2 =~ Gender + stability_q1 + before_after_4 + gamification_1 + gamification_2 + gamification_3 + gamification_4 + gamification_5 + gamification_6 + gamification_8 + gamification_12 + comfort_1 + effect_4
F3 =~ before_after_1 + before_after_2 + before_after_5 + before_after_7 + comfort_2 + comfort_3 + effect_3
F1 =~ before_after_3 + before_after_6 + before_after_8 + before_after_9 + before_after_10 + driving_1 + driving_2 + driving_3 + driving_4 + driving_5 + driving_6 + driving_timing + seat_frequency + driving_after_1 + driving_after_2 + driving_after_3 + driving_after_4 + gamification_7 + gamification_9 + gamification_10 + gamification_11 + purchase_q1 + purchase_q2 + recommend_score + effect_1 + effect_2 + effect_5 + effect_6 + effect_7 + continue_q1 + continue_q2 + psychological_safety
'

fit <- cfa(model, data=data, estimator="WLSMV")

mi <- modindices(fit)
mi <- mi[order(mi$mi, decreasing=TRUE),]

# 只保留残差相关
mi <- subset(mi, op == "~~" & lhs != rhs)

head(mi, 15)
