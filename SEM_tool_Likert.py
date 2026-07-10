import os
import re
import uuid
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from scipy.stats import kurtosis, skew

st.set_page_config(page_title="SEM Tool (Questionnaire)", layout="wide")


def init_session():
    defaults = {
        "df_raw": None,
        "df": None,
        "scale_points": 5,
        "likert_items": [],
        "reverse_items": [],
        "constructs": {},
        "parcel_map": {},
        "paths": [],
        "residual_covs": [],
        "mediation_effects": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def safe_name(name):
    name = re.sub(r"\W+", "_", str(name).strip())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = f"V_{uuid.uuid4().hex[:6]}"
    if name[0].isdigit():
        name = f"V_{name}"
    return name


def load_data(file):
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = [safe_name(c) for c in df.columns]
    return df


def run_r_script(script, filename="measure_sem_temp.R"):
    r_path = r"Your path\Rscript.exe"
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    result = subprocess.run([r_path, script_path], capture_output=True, text=True)
    return result.stdout, result.stderr


def reverse_score(df, items, scale_points):
    df = df.copy()
    for col in items:
        df[col] = scale_points + 1 - df[col]
    return df


def remove_invalid_samples(df, items, max_missing_rate, straight_threshold, min_sd):
    if not items:
        return df.copy(), pd.DataFrame()
    item_df = df[items]
    missing_rate = item_df.isna().mean(axis=1)
    max_same_rate = item_df.apply(lambda row: row.value_counts(dropna=True).max() / row.notna().sum() if row.notna().sum() else 1, axis=1)
    row_sd = item_df.std(axis=1, skipna=True)
    invalid = (missing_rate > max_missing_rate) | (max_same_rate >= straight_threshold) | (row_sd < min_sd)
    report = pd.DataFrame({
        "missing_rate": missing_rate,
        "max_same_rate": max_same_rate,
        "response_sd": row_sd,
        "invalid": invalid,
    })
    return df.loc[~invalid].copy(), report


def impute_data(df, items, method, scale_points, m=5):
    df_new = df.copy()
    if not items or method == "不补齐":
        return df_new

    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
    except Exception as exc:
        st.error(f"缺少 sklearn IterativeImputer，无法执行 {method}: {exc}")
        return df_new

    values = df_new[items].astype(float)
    if method == "EM":
        imp = IterativeImputer(random_state=0, max_iter=20, sample_posterior=False)
        filled = imp.fit_transform(values)
    else:
        filled_stack = []
        for seed in range(m):
            imp = IterativeImputer(random_state=seed, max_iter=20, sample_posterior=True)
            filled_stack.append(imp.fit_transform(values))
        filled = np.mean(filled_stack, axis=0)

    filled = np.rint(filled).clip(1, scale_points)
    df_new[items] = filled
    return df_new


def cronbach_alpha(data):
    data = data.dropna()
    if data.shape[1] < 2 or data.shape[0] < 2:
        return np.nan
    item_vars = data.var(axis=0, ddof=1).sum()
    total_var = data.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    k = data.shape[1]
    return (k / (k - 1)) * (1 - item_vars / total_var)


def reliability_table(df, items):
    rows = []
    total_alpha = cronbach_alpha(df[items])
    for item in items:
        remain = [x for x in items if x != item]
        item_total = df[item].corr(df[remain].sum(axis=1)) if remain else np.nan
        rows.append({
            "item": item,
            "item_total_corr": item_total,
            "alpha_if_deleted": cronbach_alpha(df[remain]) if len(remain) >= 2 else np.nan,
        })
    return total_alpha, pd.DataFrame(rows)


def efa_diagnostics(df, items, max_factors=5):
    sub = df[items].dropna()
    if sub.shape[1] < 2 or sub.shape[0] < 3:
        return None
    kmo_all, kmo_model = calculate_kmo(sub)
    chi, bartlett_p = calculate_bartlett_sphericity(sub)
    corr = sub.corr()
    eigenvalues = sorted(np.linalg.eigvalsh(corr), reverse=True)
    suggested_factors = max(1, min(sum(v > 1 for v in eigenvalues), max_factors, sub.shape[1] - 1))
    fa = FactorAnalyzer(n_factors=suggested_factors, rotation="oblimin")
    fa.fit(sub)
    loadings = pd.DataFrame(
        fa.loadings_,
        index=items,
        columns=[f"F{i + 1}" for i in range(suggested_factors)],
    )
    dist = pd.DataFrame({
        "item": items,
        "skew": [skew(sub[c], nan_policy="omit") for c in items],
        "kurtosis": [kurtosis(sub[c], nan_policy="omit") for c in items],
        "KMO": kmo_all,
    })
    return {
        "kmo_model": kmo_model,
        "bartlett_p": bartlett_p,
        "eigenvalues": eigenvalues,
        "suggested_factors": suggested_factors,
        "loadings": loadings,
        "dist": dist,
    }


def suggest_parcels(items, loadings, n_parcels=3):
    primary = loadings.abs().max(axis=1).sort_values(ascending=False)
    parcels = {f"P{i + 1}": [] for i in range(n_parcels)}
    for idx, item in enumerate(primary.index):
        parcels[f"P{idx % n_parcels + 1}"].append(item)
    return parcels


def active_measurement():
    return st.session_state["parcel_map"] if st.session_state["parcel_map"] else st.session_state["constructs"]


def build_measurement_model(measurement):
    lines = []
    for construct, items in measurement.items():
        if len(items) >= 2:
            lines.append(f"{safe_name(construct)} =~ " + " + ".join(safe_name(i) for i in items))
    return "\n".join(lines)


def build_sem_model(measurement, paths, residual_covs, mediation_effects=None):
    lines = [build_measurement_model(measurement)]
    for path in paths:
        lhs = safe_name(path["to"])
        rhs = safe_name(path["from"])
        label = path.get("label", "").strip()
        if label:
            lines.append(f"{lhs} ~ {safe_name(label)}*{rhs}")
        else:
            lines.append(f"{lhs} ~ {rhs}")
    for a, b in residual_covs:
        lines.append(f"{safe_name(a)} ~~ {safe_name(b)}")
    for effect in mediation_effects or []:
        name = safe_name(effect["name"])
        a = safe_name(effect["a"])
        b = safe_name(effect["b"])
        c = safe_name(effect["c"])
        lines.append(f"{name}_indirect := {a}*{b}")
        lines.append(f"{name}_total := {c} + ({a}*{b})")
    return "\n".join(x for x in lines if x.strip())


def fit_advice(cfi, tli, rmsea, srmr):
    if cfi >= 0.90 and tli >= 0.90 and rmsea <= 0.08 and srmr <= 0.08:
        return "拟合较好"
    if cfi >= 0.85 and rmsea <= 0.10:
        return "拟合一般，建议结合载荷和修正指数优化"
    return "拟合偏弱，建议检查构念、题项或残差相关"


def step1_data():
    st.header("Step 1 量表与数据处理")
    file = st.file_uploader("上传问卷数据 CSV / Excel", type=["csv", "xlsx"])
    if file:
        df = load_data(file)
        st.session_state["df_raw"] = df
        st.session_state["df"] = df.copy()
        st.success("数据加载成功")

    df = st.session_state["df"]
    if df is None:
        return

    st.dataframe(df.head())
    st.session_state["scale_points"] = st.radio("量表类型", [5, 7], horizontal=True, format_func=lambda x: f"{x}点里克量表")

    numeric = df.select_dtypes(include="number").columns.tolist()
    auto_items = [
        c for c in numeric
        if df[c].dropna().between(1, st.session_state["scale_points"]).mean() > 0.95
    ]
    likert_items = st.multiselect("选择量表题项", numeric, default=auto_items)
    reverse_items = st.multiselect("反向题目", likert_items)
    st.session_state["likert_items"] = likert_items
    st.session_state["reverse_items"] = reverse_items

    col1, col2, col3 = st.columns(3)
    with col1:
        max_missing = st.slider("无效样本：最大缺失率", 0.0, 1.0, 0.3, 0.05)
    with col2:
        straight = st.slider("无效样本：同一答案占比", 0.5, 1.0, 0.9, 0.05)
    with col3:
        min_sd = st.slider("无效样本：最小行标准差", 0.0, 1.0, 0.1, 0.05)

    impute_method = st.selectbox("缺失值补齐方法", ["不补齐", "EM", "MI"])

    if st.button("应用量表处理"):
        working = st.session_state["df_raw"].copy()
        working = reverse_score(working, reverse_items, st.session_state["scale_points"])
        working, invalid_report = remove_invalid_samples(working, likert_items, max_missing, straight, min_sd)
        working = impute_data(working, likert_items, impute_method, st.session_state["scale_points"])
        st.session_state["df"] = working
        st.session_state["invalid_report"] = invalid_report
        st.success(f"处理完成，保留样本 {len(working)} / {len(st.session_state['df_raw'])}")

    if "invalid_report" in st.session_state:
        with st.expander("无效样本判定报告"):
            st.dataframe(st.session_state["invalid_report"])


def step2_constructs():
    df = st.session_state["df"]
    items = st.session_state["likert_items"]
    if df is None or not items:
        return

    st.header("Step 2 LLM 构念划分与命名")
    question = st.text_area("研究问题 / 理论背景", placeholder="例如：服务质量、满意度如何影响持续使用意愿？")

    if st.button("LLM 生成构念方案"):
        try:
            from LLM_Construct import generate_constructs
            result = generate_constructs(question, items)
            st.session_state["construct_options"] = result
            st.success("已生成构念方案")
        except Exception as exc:
            st.error(f"LLM 构念划分失败：{exc}")

    options = st.session_state.get("construct_options", {})
    if options:
        choice = st.radio("选择构念方案", list(options.keys()))
        selected = options[choice]
        st.json(selected)
        if st.button("应用构念方案"):
            constructs = {
                safe_name(c): [safe_name(v) for v in vars_ if safe_name(v) in df.columns]
                for c, vars_ in selected.get("constructs", {}).items()
                if vars_
            }
            st.session_state["constructs"] = constructs
            st.session_state["parcel_map"] = {}
            st.success("已应用构念方案")
            st.rerun()

    st.subheader("手动调整构念")
    current = st.session_state["constructs"]
    n = st.number_input("构念数量", 1, 100, max(1, len(current) or 3))
    names = list(current.keys())
    new_map = {}
    for i in range(n):
        default_name = names[i] if i < len(names) else f"Construct_{i + 1}"
        cname = safe_name(st.text_input(f"构念 {i + 1} 名称", value=default_name, key=f"scale_construct_{i}"))
        default_items = current.get(default_name, [])
        selected_items = st.multiselect(
            f"{cname} 题项",
            items,
            default=[x for x in default_items if x in items],
            key=f"scale_construct_items_{i}",
        )
        new_map[cname] = selected_items
    st.session_state["constructs"] = new_map


def step3_reliability_efa_parcels():
    df = st.session_state["df"]
    constructs = st.session_state["constructs"]
    if df is None or not constructs:
        return

    st.header("Step 3 信度、EFA、分布与题项打包")
    parcel_map = {}

    for construct, items in constructs.items():
        if len(items) < 2:
            continue
        with st.expander(f"{construct}（{len(items)}题）", expanded=False):
            alpha, rel = reliability_table(df, items)
            st.metric("Cronbach alpha", "NA" if pd.isna(alpha) else round(alpha, 3))
            if pd.notna(alpha):
                if alpha >= 0.8:
                    st.success("信度良好")
                elif alpha >= 0.7:
                    st.warning("信度可接受，建议检查低 item-total 题项")
                else:
                    st.error("信度偏低，建议删减或重构题项")
            st.dataframe(rel.round(3))

            diag = efa_diagnostics(df, items)
            if diag:
                c1, c2, c3 = st.columns(3)
                c1.metric("KMO", round(diag["kmo_model"], 3))
                c2.metric("Bartlett p", round(diag["bartlett_p"], 5))
                c3.metric("建议因子数", diag["suggested_factors"])
                st.write("偏度 / 峰度 / 变量KMO")
                st.dataframe(diag["dist"].round(3))
                st.write("EFA 载荷")
                st.dataframe(diag["loadings"].round(3))

                need_parcel = len(items) >= 6 or diag["suggested_factors"] > 1
                if need_parcel:
                    st.info("该构念题项较多或存在多因子迹象，可考虑题项打包。")
                    use_parcel = st.checkbox(f"对 {construct} 启用题项打包", key=f"use_parcel_{construct}")
                    if use_parcel:
                        n_parcels = st.slider(f"{construct} 打包数量", 2, min(5, len(items)), 3, key=f"n_parcel_{construct}")
                        suggested = suggest_parcels(items, diag["loadings"], n_parcels)
                        st.write("建议打包：")
                        st.json(suggested)
                        selected_parcels = {}
                        for p_name, p_items in suggested.items():
                            selected_parcels[f"{construct}_{p_name}"] = st.multiselect(
                                f"{construct}_{p_name}",
                                items,
                                default=p_items,
                                key=f"parcel_{construct}_{p_name}",
                            )
                        parcel_map[construct] = list(selected_parcels.keys())
                        for parcel_name, parcel_items in selected_parcels.items():
                            if parcel_items:
                                df[parcel_name] = df[parcel_items].mean(axis=1)
                else:
                    st.success("暂不需要题项打包")

    if parcel_map:
        st.session_state["df"] = df
        active = {}
        for construct, items in constructs.items():
            active[construct] = parcel_map.get(construct, items)
        st.session_state["parcel_map"] = active
        st.success("已生成题项打包后的测量模型")
    else:
        st.session_state["parcel_map"] = {}


def step4_validity():
    df = st.session_state["df"]
    measurement = active_measurement()
    if df is None or not measurement:
        return

    st.header("Step 4 量表效度分析")
    rows = []
    scores = {}
    for construct, items in measurement.items():
        if len(items) < 2:
            continue
        sub = df[items].dropna()
        if sub.shape[0] < 3:
            continue
        fa = FactorAnalyzer(n_factors=1, rotation=None)
        fa.fit(sub)
        loadings = np.abs(fa.loadings_.flatten())
        cr = (loadings.sum() ** 2) / ((loadings.sum() ** 2) + np.sum(1 - loadings ** 2))
        ave = np.mean(loadings ** 2)
        rows.append({
            "construct": construct,
            "CR": cr,
            "AVE": ave,
            "建议": "良好" if cr >= 0.7 and ave >= 0.5 else "建议检查低载荷题项或重新划分构念",
        })
        scores[construct] = df[items].mean(axis=1)

    validity = pd.DataFrame(rows)
    st.dataframe(validity.round(3))
    if len(scores) >= 2:
        score_df = pd.DataFrame(scores)
        corr = score_df.corr()
        st.write("构念相关矩阵")
        st.dataframe(corr.round(3))
        st.caption("粗略判断：AVE 应尽量大于该构念与其他构念相关系数平方。")


def step5_cfa_mi():
    df = st.session_state["df"]
    measurement = active_measurement()
    if df is None or not measurement:
        return

    st.header("Step 5 CFA、修正指数与残差相关")
    model = build_measurement_model(measurement)
    for a, b in st.session_state["residual_covs"]:
        model += f"\n{safe_name(a)} ~~ {safe_name(b)}"

    st.code(model, language="r")
    new_cov = st.text_input("添加残差相关：题项1,题项2")
    if st.button("添加残差相关"):
        if "," in new_cov:
            a, b = [x.strip() for x in new_cov.split(",", 1)]
            st.session_state["residual_covs"].append((safe_name(a), safe_name(b)))
            st.rerun()

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measure_cfa_data.csv").replace("\\", "/")
    df.to_csv(data_path, index=False)
    script = f"""
library(lavaan)
data <- read.csv("{data_path}", check.names=FALSE)
model <- '
{model}
'
fit <- cfa(model, data=data, estimator="WLSMV", ordered=names(data))
cat("FIT\\n")
print(round(fitMeasures(fit, c("cfi","tli","rmsea","srmr")), 4))
cat("\\nSUMMARY\\n")
print(summary(fit, fit.measures=TRUE, standardized=TRUE))
cat("\\nMODIFICATION_INDICES\\n")
mi <- modindices(fit, sort.=TRUE, maximum.number=30)
print(mi)
"""
    if st.button("运行 CFA / 输出修正指数"):
        out, err = run_r_script(script, "measure_cfa.R")
        if out:
            st.text(out)
        if err:
            st.error(err)


def step6_paths_sem_mediation():
    df = st.session_state["df"]
    measurement = active_measurement()
    if df is None or not measurement:
        return

    st.header("Step 6 路径推荐、SEM 与中介效应")
    constructs = list(measurement.keys())

    col1, col2 = st.columns(2)
    with col1:
        if st.button("LLM 推荐路径"):
            try:
                from LLM_Path import generate_sem_paths
                paths = generate_sem_paths(constructs, [], [])
                st.session_state["paths"] = paths
                st.success(f"生成 {len(paths)} 条路径")
                st.rerun()
            except Exception as exc:
                st.error(f"LLM 路径推荐失败：{exc}")
    with col2:
        st.caption("路径标签用于中介效应，例如 a、b、cprime。")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        src = st.selectbox("路径起点", constructs, key="path_src")
    with c2:
        tgt = st.selectbox("路径终点", constructs, key="path_tgt")
    with c3:
        label = st.text_input("标签", key="path_label")
    if st.button("添加路径"):
        if src != tgt:
            st.session_state["paths"].append({"from": src, "to": tgt, "label": safe_name(label) if label else ""})
            st.rerun()

    if st.session_state["paths"]:
        st.write("当前路径")
        for i, path in enumerate(st.session_state["paths"]):
            st.write(f"{i + 1}. {path.get('label','')} {path['from']} → {path['to']}")
        if st.button("清空路径"):
            st.session_state["paths"] = []
            st.rerun()

    st.subheader("中介效应检验")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        x = st.selectbox("X", constructs, key="med_x")
    with m2:
        m = st.selectbox("M", constructs, key="med_m")
    with m3:
        y = st.selectbox("Y", constructs, key="med_y")
    with m4:
        med_name = st.text_input("效应名", value="med1")
    if st.button("添加中介模型"):
        st.session_state["paths"].extend([
            {"from": x, "to": m, "label": "a"},
            {"from": m, "to": y, "label": "b"},
            {"from": x, "to": y, "label": "c"},
        ])
        st.session_state["mediation_effects"].append({"name": med_name, "a": "a", "b": "b", "c": "c"})
        st.rerun()

    bootstrap_n = st.number_input("Bootstrap 次数", 500, 10000, 2000, step=500)
    model = build_sem_model(measurement, st.session_state["paths"], st.session_state["residual_covs"], st.session_state["mediation_effects"])
    st.code(model, language="r")

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measure_sem_data.csv").replace("\\", "/")
    df.to_csv(data_path, index=False)
    has_mediation = bool(st.session_state["mediation_effects"])
    estimator = "ML" if has_mediation else "MLR"
    se_part = f'se="bootstrap", bootstrap={int(bootstrap_n)}' if has_mediation else 'se="standard"'
    script = f"""
library(lavaan)
data <- read.csv("{data_path}", check.names=FALSE)
model <- '
{model}
'
fit <- sem(model, data=data, estimator="{estimator}", {se_part})
print(summary(fit, fit.measures=TRUE, standardized=TRUE, rsquare=TRUE, ci=TRUE))
"""
    if st.button("运行 SEM"):
        out, err = run_r_script(script, "measure_sem.R")
        if out:
            st.text(out)
        if err:
            st.error(err)


def main():
    init_session()
    st.title("SEM Tool (问卷量表版)")
    step1_data()
    step2_constructs()
    step3_reliability_efa_parcels()
    step4_validity()
    step5_cfa_mi()
    step6_paths_sem_mediation()


if __name__ == "__main__":
    main()
