
    library(lavaan)
    data_raw <- read.csv("C:/Users/yecan/PycharmProjects/pythonProject3/sem_data.csv", check.names = FALSE)
    data <- data_raw[, sapply(data_raw, is.numeric), drop = FALSE]

    model <- '
    Physiological_State =~ NA*AvgHR + AvgHRV + AvgSdnn + AvgRmssd + AvgPnn20 + AvgPnn50 + StdHR + StdHRV
Visual_Attention =~ NA*EyeYRight + EyeYLeft + EyeDimLeftStd + EyeDimRightStd + EyeYLeftStd + EyeYRightStd
Vehicle_Control_Anomaly =~ NA*steeringWheelStd + accCarZStd + accCarXStd + accCarYStd + accCarX + accCarY + accCarZ
Physiological_State ~~ 1*Physiological_State
Visual_Attention ~~ 1*Visual_Attention
Vehicle_Control_Anomaly ~~ 1*Vehicle_Control_Anomaly
Physiological_State ~ Age
UnsafeBehaviors ~ Gender
Vehicle_Control_Anomaly ~ Freq
Physiological_State ~ CortisolBefore
Physiological_State ~ voiceWeightAvg
Visual_Attention ~ Physiological_State
Vehicle_Control_Anomaly ~ Physiological_State
UnsafeBehaviors ~ Vehicle_Control_Anomaly
Visual_Attention ~ Freq
Visual_Attention ~ Vehicle_Control_Anomaly
UnsafeBehaviors ~ Visual_Attention
AvgHR ~~ AvgHRV
AvgSdnn ~~ StdHRV
EyeYRight ~~ EyeYLeft
EyeDimLeftStd ~~ EyeDimRightStd
AvgPnn20 ~~ AvgPnn50
steeringWheelStd ~~ accCarY
accCarXStd ~~ accCarYStd
AvgRmssd ~~ StdHR
accCarZStd ~~ accCarXStd
steeringWheelStd ~~ accCarYStd
AvgRmssd ~~ AvgPnn50
accCarZStd ~~ accCarYStd
accCarYStd ~~ accCarX
    '

    estimator_name <- "MLR"

    if (estimator_name == "GLS") {
      data <- na.omit(data)
      fit <- suppressWarnings(sem(model, data = data, estimator = estimator_name))
    } else {
      fit <- suppressWarnings(sem(model, data = data, estimator = estimator_name, missing = "fiml"))
    }

    cat("ESTIMATOR\n")
    cat(estimator_name)
    cat("\nN_USED\n")
    cat(lavInspect(fit, "nobs"))
    cat("\n\n")

    mi <- suppressWarnings(modindices(fit, sort.=TRUE, maximum.number=20))
    print(mi)
    