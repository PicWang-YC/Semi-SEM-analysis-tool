library(psych)

# 读取数据
data <- read.csv("efa_data.csv")

# 只取数值列
data <- data[sapply(data, is.numeric)]

# =========================
# 1. KMO
# =========================
kmo <- KMO(data)
print("KMO:")
print(kmo)

# =========================
# 2. Bartlett
# =========================
bart <- cortest.bartlett(cor(data), n=nrow(data))
print("Bartlett:")
print(bart)

# =========================
# 3. 平行分析（推荐因子数）
# =========================
fa.parallel(data, fa="fa")

# =========================
# 4. EFA
# =========================
# 这里可以让Python传入因子数
fa_result <- fa(data, nfactors=3, rotate="varimax")

print("Loadings:")
print(fa_result$loadings)

# =========================
# 5. 输出构念（简单规则）
# =========================
loadings <- as.data.frame(unclass(fa_result$loadings))

for (i in 1:ncol(loadings)) {
  cat(paste0("\nFactor", i, ":\n"))
  vars <- rownames(loadings)[abs(loadings[,i]) > 0.4]
  print(vars)
}