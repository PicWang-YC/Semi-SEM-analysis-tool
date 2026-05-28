
library(lavaan)
data <- read.csv("sem_data.csv")

model <- '
Physiological_State =~ 1*AvgHR + AvgHRV + AvgRmssd + AvgPnn50 + StressSdk
Oculomotor_Attention =~ 1*EyeYLeft + EyeYRight
Vehicle_Control =~ 1*steeringWheelStd + accCarX + accCarY
Driving_Dynamics =~ 1*Yaw + YawStd
Fatigue_Sleep =~ 1*meanSleepTot + stdSleepTot
Voice_Emotion =~ 1*VoiceN + VoiceT + VoiceA + voiceAvg + voiceWeightAvg
Physiological_State ~ Age
Voice_Emotion ~ Gender
Vehicle_Control ~ DrivingAge
Driving_Dynamics ~ Freq
Physiological_State ~ Fatigue_Sleep
Oculomotor_Attention ~ Physiological_State
Oculomotor_Attention ~ Voice_Emotion
Vehicle_Control ~ Oculomotor_Attention
Vehicle_Control ~ Driving_Dynamics
UnsafeBehaviors ~ Vehicle_Control
UnsafeBehaviors ~ Driving_Dynamics
UnsafeBehaviors ~ Physiological_State
UnsafeBehaviors ~ Voice_Emotion
Driving_Dynamics ~ Vehicle_Control
UnsafeBehaviors ~ Fatigue_Sleep
Oculomotor_Attention ~ Freq
AvgRmssd ~~ AvgPnn50
VoiceT ~~ voiceAvg
VoiceT ~~ VoiceA
voiceAvg ~~ voiceWeightAvg
meanSleepTot ~~ stdSleepTot
accCarX ~~ accCarY
'

fit <- sem(model, data = data, estimator = "MLR")
summary(fit, fit.measures=TRUE, standardized=TRUE)

