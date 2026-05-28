import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
# ============================
# 1. Pearson 相关性检验模块
# ============================
def find_low_correlation_variables(df, threshold=0.2):
    corr_matrix = df.corr(method='pearson')
    low_corr_vars = []

    for col in corr_matrix.columns:
        # 与所有变量相关性绝对值都小于 threshold
        if all(corr_matrix[col].abs().drop(col) < threshold):
            low_corr_vars.append(col)

    return low_corr_vars, corr_matrix


# ============================
# 2. 可视化函数：散点图矩阵
# ============================
def plot_scatter_matrix(df, var):
    sns.pairplot(df, vars=[var] + [c for c in df.columns if c != var])
    plt.suptitle(f"Scatterplot Matrix for {var}", y=1.02)
    plt.show()

# ============================
# 3. 可视化函数：部分残差图（Partial Residual)
# ============================
def plot_partial_residuals(df, var):
    y = df[var]
    X = df.drop(columns=[var])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    fig = plt.figure(figsize=(12, 8))
    sm.graphics.plot_ccpr_grid(model, fig=fig)
    plt.suptitle(f"Partial Residual Plots for {var}")
    plt.show()


# ============================
# 4. 共线性 VIF 检查
# ============================
def compute_vif(df):
    X = sm.add_constant(df)
    vif_df = pd.DataFrame()
    vif_df["Variable"] = df.columns
    vif_df["VIF"] = [
        variance_inflation_factor(X.values, i+1)
        for i in range(len(df.columns))
    ]
    return vif_df


# ============================
# 5. 完整验证流程模块
# ============================
def validation_module(df):
    print("\n===== Step 1: Pearson 相关性检验 =====")
    low_corr_vars, corr_matrix = find_low_correlation_variables(df)
    print("\n与所有变量相关性都 < 0.2 的变量：")
    print(low_corr_vars if low_corr_vars else "无")

    if low_corr_vars:
        print("\n你希望查看散点图矩阵(scatter)还是部分残差图(residual)？")
        plot_choice = input("输入 scatter / residual / none： ").strip()

        if plot_choice in ["scatter", "residual"]:
            print("\n你想查看哪个变量的图？可选：")
            print(low_corr_vars)
            var_choice = input("请输入变量名： ").strip()

            if var_choice in low_corr_vars:
                if plot_choice == "scatter":
                    plot_scatter_matrix(df, var_choice)
                else:
                    plot_partial_residuals(df, var_choice)

        print("\n是否删除这些低相关变量？输入变量名，或输入 none：")
        del_choice = input("删除： ").strip()

        if del_choice in low_corr_vars:
            df = df.drop(columns=[del_choice])
            print(f"已删除变量：{del_choice}")
        else:
            print("未删除变量。")

    print("\n===== Step 2: VIF 共线性检查 =====")
    vif_df = compute_vif(df)
    print(vif_df)

    high_vif_vars = vif_df[vif_df["VIF"] < 10]["Variable"].tolist()
    print("\nVIF < 10 的变量：")
    print(high_vif_vars if high_vif_vars else "无")

    if high_vif_vars:
        print("\n是否删除 VIF > 10 的变量？输入变量名，或 none：")
        del_vif = input("删除： ").strip()

        if del_vif in high_vif_vars:
            df = df.drop(columns=[del_vif])
            print(f"已删除变量：{del_vif}")
        else:
            print("未删除。")

    print("\n===== 最终保留的变量 =====")
    print(df.columns.tolist())

    return df

def kmo_bartlett_test(df, threshold=0.6):
    print("\n===== Step 0: KMO & Bartlett 球形检验 =====")

    # KMO
    kmo_all, kmo_model = calculate_kmo(df)
    print(f"\n整体 KMO: {kmo_model:.4f}")

    kmo_df = pd.DataFrame({
        "Variable": df.columns,
        "KMO": kmo_all
    })

    # 只筛选 KMO < threshold
    low_kmo_df = kmo_df[kmo_df["KMO"] > threshold]

    print(f"\nKMO < {threshold} 的变量：")
    if low_kmo_df.empty:
        print("无")
    else:
        #print(high_kmo_df.sort_values(by="KMO"))
        print(low_kmo_df.to_string(index=False))

    # Bartlett
    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    print("\nBartlett 球形检验：")
    print(f"Chi-square: {chi_square_value:.4f}, p-value: {p_value:.6f}")

    return kmo_model, kmo_df, low_kmo_df, chi_square_value, p_value