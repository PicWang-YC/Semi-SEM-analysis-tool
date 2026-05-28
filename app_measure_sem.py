import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import os
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity

st.set_page_config(page_title="SEM Tool (Scale Version)", layout="wide")

# =========================
# 初始化
# =========================
def init_session():
    defaults = {
        "df": None,
        "likert_items": [],
        "reverse_items": [],
        "constructs": {},
        "paths": [],
        "residual_covs": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =========================
# 工具函数
# =========================
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def detect_likert(series):
    vals = series.dropna().unique()
    return len(vals) <= 7 and all(float(v).is_integer() for v in vals)


def reverse_score(df, items):
    df = df.copy()
    for col in items:
        max_val = df[col].max()
        df[col] = max_val + 1 - df[col]
    return df


def cronbach_alpha(df):
    k = df.shape[1]
    var_sum = df.var(axis=0, ddof=1).sum()
    total_var = df.sum(axis=1).var(ddof=1)
    return (k/(k-1))*(1 - var_sum/total_var)


def build_model(measurement, paths, residuals):
    lines = []

    for c, items in measurement.items():
        if len(items) >= 2:
            lines.append(f"{c} =~ " + " + ".join(items))

    for p in paths:
        lines.append(f"{p['to']} ~ {p['from']}")

    for r in residuals:
        lines.append(f"{r[0]} ~~ {r[1]}")

    return "\n".join(lines)
def build_cfa_model(constructs):
    lines = []

    for c, items in constructs.items():
        # 👉 过滤单题构念（关键）
        if len(items) < 2:
            continue

        line = f"{c} =~ " + " + ".join(items)
        lines.append(line)

    return "\n".join(lines)


def run_r(script):
    # 👉 改成你自己的 R 路径
    r_path = r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"

    if not os.path.exists(r_path):
        return "", "❌ Rscript.exe 路径错误，请检查"

    script_path = os.path.join(os.getcwd(), "temp_cfa.R")

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    res = subprocess.run(
        [r_path, script_path],
        capture_output=True,
        text=True
    )

    return res.stdout, res.stderr


# =========================
# Step 1 上传
# =========================
def step1():
    st.header("Step 1 上传数据")
    file = st.file_uploader("CSV / Excel")

    if file:
        st.session_state["df"] = load_data(file)
        st.success("数据加载成功")

    if st.session_state["df"] is not None:
        st.dataframe(st.session_state["df"].head())


# =========================
# Step 2 量表处理
# =========================
def step2():
    df = st.session_state["df"]
    if df is None:
        return

    st.header("Step 2 量表设置")

    numeric = df.select_dtypes(include="number").columns.tolist()

    auto = [c for c in numeric if detect_likert(df[c])]

    likert = st.multiselect("Likert题项", numeric, default=auto)
    reverse = st.multiselect("反向题", likert)

    st.session_state["likert_items"] = likert
    st.session_state["reverse_items"] = reverse

    if st.button("应用"):
        df_new = reverse_score(df, reverse)
        st.session_state["df"] = df_new
        st.success("处理完成")


# =========================
# Step 3 KMO
# =========================
def step3():
    df = st.session_state["df"]
    items = st.session_state["likert_items"]

    if not items:
        return

    st.header("Step 3 KMO")

    sub = df[items].dropna()

    kmo_all, kmo_model = calculate_kmo(sub)
    st.write("整体KMO:", round(kmo_model, 3))

    chi, p = calculate_bartlett_sphericity(sub)
    st.write("Bartlett p:", p)

    kmo_df = pd.DataFrame({
        "item": items,
        "KMO": kmo_all
    })

    st.dataframe(kmo_df)


# =========================
# Step 4 EFA
# =========================
def step4():
    df = st.session_state["df"]
    items = st.session_state["likert_items"]

    if not items:
        return

    st.header("Step 4 EFA")

    n = st.slider("因子数", 1, 10, 3)

    if st.button("运行EFA"):
        sub = df[items].dropna()

        fa = FactorAnalyzer(n_factors=n, rotation="oblimin")
        fa.fit(sub)

        loadings = pd.DataFrame(
            fa.loadings_,
            index=items,
            columns=[f"F{i+1}" for i in range(n)]
        )

        st.dataframe(loadings)

        # 自动分配
        mapping = {}
        for item in items:
            factor = loadings.loc[item].abs().idxmax()
            mapping.setdefault(factor, []).append(item)

        st.session_state["constructs"] = mapping


# =========================
# Step 5 构念确认 + 信度
# =========================
def step5():
    df = st.session_state["df"]
    constructs = st.session_state["constructs"]

    if not constructs:
        return

    st.header("Step 5 构念 & 信度")

    new_map = {}

    for c, items in constructs.items():
        sel = st.multiselect(c, items, default=items)
        new_map[c] = sel

        if len(sel) >= 2:
            alpha = cronbach_alpha(df[sel])
            st.write(f"{c} α = {round(alpha,3)}")

    st.session_state["constructs"] = new_map


# =========================
# Step 6 CFA
# =========================
def step6():
    if st.session_state.get("df") is None:
        return

    df = st.session_state["df"].copy()
    constructs = st.session_state.get("constructs", {})

    if not constructs:
        st.warning("请先完成构念定义")
        return

    st.header("Step 6 CFA（验证性因子分析）")
    model = build_cfa_model(constructs)

    if not model.strip():
        st.error("所有构念都只有1个题项，无法进行CFA")
        return

    st.subheader("模型语句（lavaan）")
    st.code(model, language="r")

    # =========================
    # 数据路径（绝对路径）
    # =========================
    base_dir = os.getcwd()
    data_path = os.path.join(base_dir, "cfa_data.csv").replace("\\", "/")

    df.to_csv(data_path, index=False)

    # =========================
    # R 脚本
    # =========================
    script = f"""
library(lavaan)

data <- read.csv("{data_path}")

model <- '
{model}
'

fit <- cfa(model, data=data, estimator="WLSMV")

fitMeasures(fit, c("cfi","tli","rmsea","srmr"))

summary(fit, fit.measures=TRUE, standardized=TRUE)
"""

    # =========================
    # 运行按钮
    # =========================
    if st.button("运行 CFA"):

        out, err = run_r(script)

        if err:
            st.error(err)

        if out:
            st.subheader("CFA输出")
            st.text(out)

            # =========================
            # 简单解析 fit 指标
            # =========================
            try:
                import re

                cfi = float(re.search(r"cfi\\s+([0-9.]+)", out).group(1))
                tli = float(re.search(r"tli\\s+([0-9.]+)", out).group(1))
                rmsea = float(re.search(r"rmsea\\s+([0-9.]+)", out).group(1))
                srmr = float(re.search(r"srmr\\s+([0-9.]+)", out).group(1))

                st.subheader("模型拟合指标")

                st.write(f"CFI = {cfi}")
                st.write(f"TLI = {tli}")
                st.write(f"RMSEA = {rmsea}")
                st.write(f"SRMR = {srmr}")

                # =========================
                # 自动评价
                # =========================
                if cfi > 0.90 and rmsea < 0.08 and srmr < 0.08:
                    st.success("模型拟合良好")
                elif cfi > 0.85:
                    st.warning("模型拟合一般，建议优化")
                else:
                    st.error("模型拟合较差，需要调整模型")

            except:
                st.info("未能自动解析fit指标，请查看上方输出")

# =========================
# Step 7 SEM
# =========================
def step7():
    constructs = list(st.session_state["constructs"].keys())

    st.header("Step 7 路径分析")

    src = st.selectbox("from", constructs)
    tgt = st.selectbox("to", constructs)

    if st.button("添加路径"):
        st.session_state["paths"].append({"from": src, "to": tgt})

    for p in st.session_state["paths"]:
        st.write(p)

    if st.button("运行SEM"):
        df = st.session_state["df"]

        model = build_model(
            st.session_state["constructs"],
            st.session_state["paths"],
            st.session_state["residual_covs"]
        )

        df.to_csv("data.csv", index=False)

        script = f"""
        library(lavaan)
        data <- read.csv("data.csv")

        model <- '
        {model}
        '

        fit <- sem(model, data=data, estimator="WLSMV")
        summary(fit, fit.measures=TRUE, standardized=TRUE)
        """

        out, err = run_r(script)
        st.text(out)
# =========================
# Step 7 MI优化 + 残差相关
# =========================
def step8():
    import os

    df = st.session_state.get("parcel_data", st.session_state["df"])
    constructs = st.session_state.get("parcel_map", st.session_state["constructs"])

    if not constructs:
        return

    st.header("Step 8 模型优化（Modification Index）")

    # =========================
    # 构建模型
    # =========================
    model = build_cfa_model(constructs)

    # 残差相关
    residual_covs = st.session_state.get("residual_covs", [])

    for pair in residual_covs:
        model += f"\n{pair[0]} ~~ {pair[1]}"

    st.subheader("当前模型")
    st.code(model, language="r")

    # =========================
    # 数据路径
    # =========================
    base_dir = os.getcwd()
    data_path = os.path.join(base_dir, "mi_data.csv").replace("\\", "/")
    df.to_csv(data_path, index=False)

    # =========================
    # R脚本（提取MI）
    # =========================
    script = f"""
library(lavaan)

data <- read.csv("{data_path}")

model <- '
{model}
'

fit <- cfa(model, data=data, estimator="WLSMV")

mi <- modindices(fit)
mi <- mi[order(mi$mi, decreasing=TRUE),]

# 只保留残差相关
mi <- subset(mi, op == "~~" & lhs != rhs)

head(mi, 15)
"""

    if st.button("获取MI建议"):
        out, err = run_r(script)

        if err:
            st.error(err)

        if out:
            st.subheader("Top MI 建议")
            st.text(out)

            # =========================
            # 简单解析（可选增强）
            # =========================
            lines = out.split("\n")

            suggestions = []

            for line in lines:
                parts = line.split()
                if len(parts) >= 4:
                    lhs = parts[1]
                    rhs = parts[3]
                    suggestions.append((lhs, rhs))

            # =========================
            # UI 勾选
            # =========================
            st.subheader("选择要添加的残差相关")

            selected = []

            for i, (a, b) in enumerate(suggestions):
                if st.checkbox(f"{a} ~~ {b}", key=f"mi_{i}"):
                    selected.append((a, b))

            if st.button("应用选择"):
                current = st.session_state.get("residual_covs", [])

                for pair in selected:
                    if pair not in current:
                        current.append(pair)

                st.session_state["residual_covs"] = current

                st.success("已添加残差相关，建议重新运行CFA")

# =========================
# 主程序
# =========================
def main():
    init_session()

    st.title("📊 SEM Tool (量表版)")

    step1()
    step2()
    step3()
    step4()
    step5()
    step6()
    step7()
    step8()


if __name__ == "__main__":
    main()