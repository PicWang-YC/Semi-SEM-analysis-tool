import subprocess
import streamlit as st
import pandas as pd
from Data_Analyzer import DataAnalyzer
import numpy as np
import sklearn
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import boxcox
from LLM_SEM import generate_sem_paths
from LLM_Construct import generate_constructs
# =========================
# 页面配置
# =========================
st.set_page_config(page_title="SEM Tool", layout="wide")

# =========================
# Session 初始化
# =========================
def init_session():
    defaults = {
        "df": None,
        "step": "data",
        "step2_stage": "delete",
        "constructs": [],
        "paths": [],
        "measurement_model": {},
        "subconstruct_map": {},
        "exogenous_vars": [],
        "endogenous_vars": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =========================
# 工具函数
# =========================
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="latin1")
    else:
        df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()
    return df


def apply_transformations(df, operations, std_method):
    df_new = df.copy()

    for col, op in operations.items():
        if op["fill_mean"]:
            df_new[col] = df_new[col].fillna(df_new[col].mean())
        if op["clip"]:
            Q1 = df_new[col].quantile(0.25)
            Q3 = df_new[col].quantile(0.75)
            IQR = Q3 - Q1
            df_new[col] = df_new[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        if op["log"] and (df_new[col] > 0).all():
            df_new[col] = np.log(df_new[col])
        if op["boxcox"] and (df_new[col] > 0).all():
            df_new[col], _ = boxcox(df_new[col])

    std_cols = [c for c, op in operations.items() if op["standardize"]]
    if std_cols:
        scaler = StandardScaler() if std_method == "Z-score" else MinMaxScaler()
        df_new[std_cols] = scaler.fit_transform(df_new[std_cols])

    return df_new


def build_lavaan_model():
    constructs = st.session_state["constructs"]
    mapping = st.session_state["measurement_model"]
    paths = st.session_state["paths"]
    sub_map = st.session_state["subconstruct_map"]
    ident_method = st.session_state["ident_method"]

    def safe_name(name):
        if any(c in name for c in [' ', '-', '/', '(', ')', '.']):
            return f"`{name}`"
        return name

    lines = []
    fixed_variance_lines = []

    measurement_constructs = list(constructs)
    for con in mapping.values():
        if con and con not in measurement_constructs:
            measurement_constructs.append(con)

    second_order_parents = set(sub_map.keys())

    for c in measurement_constructs:
        if c in second_order_parents:
            continue
        indicators = [v for v, con in mapping.items() if con == c]
        if len(indicators) >= 2:
            if ident_method == "fix_loading":
                first = indicators[0]
                rest = indicators[1:]
                line = f"{safe_name(c)} =~ 1*{safe_name(first)} + " + " + ".join(safe_name(v) for v in rest)
            else:
                first = indicators[0]
                rest = indicators[1:]
                line = f"{safe_name(c)} =~ NA*{safe_name(first)}"
                if rest:
                    line += " + " + " + ".join(safe_name(v) for v in rest)
                fixed_variance_lines.append(f"{safe_name(c)} ~~ 1*{safe_name(c)}")
            lines.append(line)

    for parent, subs in sub_map.items():
        valid_subs = [
            s for s in subs
            if len([v for v, con in mapping.items() if con == s]) >= 2
        ]
        direct_indicators = [v for v, con in mapping.items() if con == parent]
        parent_indicators = valid_subs + direct_indicators
        if len(parent_indicators) >= 2:
            if ident_method == "fix_loading":
                first = parent_indicators[0]
                rest = parent_indicators[1:]
                line = f"{safe_name(parent)} =~ 1*{safe_name(first)}"
                if rest:
                    line += " + " + " + ".join(safe_name(s) for s in rest)
            else:
                first = parent_indicators[0]
                rest = parent_indicators[1:]
                line = f"{safe_name(parent)} =~ NA*{safe_name(first)}"
                if rest:
                    line += " + " + " + ".join(safe_name(s) for s in rest)
                fixed_variance_lines.append(f"{safe_name(parent)} ~~ 1*{safe_name(parent)}")
            lines.append(line)

    lines.extend(fixed_variance_lines)

    for p in paths:
        if p["from"] != p["to"]:
            lines.append(f"{safe_name(p['to'])} ~ {safe_name(p['from'])}")

    # 残差相关
    residual_covs = st.session_state.get("residual_covs", [])
    for pair in residual_covs:
        lines.append(f"{safe_name(pair[0])} ~~ {safe_name(pair[1])}")

    return "\n".join(lines)


def build_single_construct_cfa_model(construct, indicators):
    ident_method = st.session_state.get("ident_method", "fix_loading")

    def safe_name(name):
        if any(c in name for c in [' ', '-', '/', '(', ')', '.']):
            return f"`{name}`"
        return name

    if len(indicators) < 2:
        return ""

    if ident_method == "fix_loading":
        first = indicators[0]
        rest = indicators[1:]
        return f"{safe_name(construct)} =~ 1*{safe_name(first)} + " + " + ".join(safe_name(v) for v in rest)

    first = indicators[0]
    rest = indicators[1:]
    model = f"{safe_name(construct)} =~ NA*{safe_name(first)}"
    if rest:
        model += " + " + " + ".join(safe_name(v) for v in rest)
    model += f"\n{safe_name(construct)} ~~ 1*{safe_name(construct)}"
    return model


def run_r_script(script_path):
    r_path = r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"
    result = subprocess.run(
        [r_path, script_path],
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr


# =========================
# Step 1 上传数据
# =========================
def step_upload():
    st.header("Step 1: 上传数据")

    file = st.file_uploader("上传 CSV / Excel", type=["csv", "xlsx"])

    if file:
        if st.session_state["df"] is None:
            df = load_data(file)
            st.session_state["df"] = df
            st.session_state["step2_stage"] = "delete"
            st.success("上传成功")
        else:
            st.warning("已有数据，是否替换？")
            if st.button("✅ 确认替换数据"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                init_session()
                df = load_data(file)
                st.session_state["df"] = df
                st.session_state["step2_stage"] = "delete"
                st.success("数据已替换")
                st.rerun()

    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.dataframe(df.head())
        st.write(f"数据维度: {df.shape}")


# =========================
# Step 2 数据处理
# =========================
def step_processing():
    if st.session_state["df"] is None:
        return

    st.header("Step 2: 数据处理")

    if "step2_stage" not in st.session_state:
        st.session_state["step2_stage"] = "delete"

    stage = st.session_state["step2_stage"]

    if stage == "process":
        if st.button("🔁 重新选择删除变量"):
            st.session_state["step2_stage"] = "delete"
            st.rerun()

    # ---------- 阶段1：删除变量 ----------
    if stage == "delete":
        st.subheader("Step 2.1 删除变量")

        df = st.session_state["df"]

        delete_cols = st.multiselect(
            "选择要删除的变量",
            df.columns.tolist(),
            key="delete_cols"
        )

        if st.button("确认删除变量"):
            df_new = df.drop(columns=delete_cols, errors="ignore")
            st.session_state["df"] = df_new

            for col in delete_cols:
                for key in [f"map_{col}", f"assign_{col}", f"kmo_del_{col}"]:
                    if key in st.session_state:
                        del st.session_state[key]

            for k in list(st.session_state.keys()):
                if any(s in k for s in ["_fill", "_clip", "_log", "_boxcox", "_std", "global_std", "std_method"]):
                    del st.session_state[k]

            st.session_state["step2_stage"] = "process"
            st.success(f"已删除 {len(delete_cols)} 个变量")
            st.rerun()

    # ---------- 阶段2：数据处理 ----------
    elif stage == "process":
        st.subheader("Step 2.2 数据处理")

        df = st.session_state["df"]
        st.write("当前变量：", df.columns.tolist())

        numeric_cols = [
            col for col in df.select_dtypes(include='number').columns
            if col.lower() not in ["id", "date"]
        ]

        if not numeric_cols:
            st.warning("没有数值变量可处理")
            return

        global_std = st.checkbox("全局标准化", value=True, key="global_std")
        std_method = st.selectbox("标准化方法", ["Z-score", "Min-Max"], key="std_method")

        operations = {}

        for i, col in enumerate(numeric_cols):
            series = df[col]
            missing_count = series.isna().sum()
            missing_rate = missing_count / len(series)
            skew = series.skew()
            kurt = series.kurt()
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum()

            rec_fill   = missing_rate > 0
            rec_clip   = outliers > 0
            rec_log    = skew > 1 and (series > 0).all()
            rec_boxcox = abs(skew) > 1 and not rec_log and (series > 0).all()

            recs = []
            if rec_fill:   recs.append(f"均值填充（缺失率{missing_rate:.0%}）")
            if rec_clip:   recs.append(f"极值截断（{outliers}个异常值）")
            if rec_log:    recs.append("log变换（右偏）")
            if rec_boxcox: recs.append("Box-Cox变换（偏态）")

            with st.expander(
                f"🔹 {col}" + (f"  ⚠️ 推荐: {', '.join(recs)}" if recs else "  ✅ 无需特殊处理")
            ):
                st.write({
                    "缺失值": int(missing_count),
                    "缺失率": f"{missing_rate:.1%}",
                    "偏度": round(skew, 3),
                    "峰度": round(kurt, 3),
                    "异常值数量": int(outliers)
                })

                op = {}
                op["fill_mean"]   = st.checkbox("均值填充",    value=rec_fill,   key=f"{i}_fill")
                op["clip"]        = st.checkbox("极值截断",    value=rec_clip,   key=f"{i}_clip")
                op["log"]         = st.checkbox("log变换",     value=rec_log,    key=f"{i}_log")
                op["boxcox"]      = st.checkbox("Box-Cox变换", value=rec_boxcox, key=f"{i}_boxcox")
                op["standardize"] = st.checkbox("标准化",      value=global_std, key=f"{i}_std")

                operations[col] = op

        if st.button("应用数据处理"):
            df_new = apply_transformations(df, operations, std_method)
            st.session_state["df"] = df_new
            st.success("数据处理完成")
            st.rerun()


# =========================
# Step 3 变量角色设置
# =========================
def step_var_setup():
    if st.session_state.get("step2_stage") != "process":
        return

    st.header("Step 3: 变量角色设置")
    st.caption("请先设置内生变量和外生变量，KMO 检验将自动排除这些变量")

    df = st.session_state["df"]
    all_vars = df.columns.tolist()

    st.subheader("外生变量（自变量）")
    exo = st.multiselect(
        "选择外生变量",
        options=all_vars,
        default=st.session_state.get("exogenous_vars", []),
        key="exo_setup"
    )
    st.session_state["exogenous_vars"] = exo

    st.subheader("内生变量（因变量）")
    endo = st.multiselect(
        "选择内生变量",
        options=all_vars,
        default=st.session_state.get("endogenous_vars", []),
        key="endo_setup"
    )
    st.session_state["endogenous_vars"] = endo

    remaining = [v for v in all_vars if v not in exo and v not in endo]
    st.info(f"将参与 KMO 检验的变量（{len(remaining)} 个）：{remaining}")


# =========================
# Step 4 KMO & Bartlett
# =========================
def step_kmo():
    if st.session_state.get("step2_stage") != "process":
        return

    df = st.session_state["df"]
    if df is None:
        return

    st.header("Step 4: KMO & Bartlett")

    # 排除内生/外生变量
    exo_vars  = st.session_state.get("exogenous_vars", [])
    endo_vars = st.session_state.get("endogenous_vars", [])
    exclude   = exo_vars + endo_vars

    numeric_df = df.select_dtypes(include='number').dropna()
    numeric_df = numeric_df.drop(columns=[c for c in exclude if c in numeric_df.columns])

    # 静默剔除方差为0的列
    zero_var = numeric_df.columns[numeric_df.var() == 0].tolist()
    if zero_var:
        numeric_df = numeric_df.drop(columns=zero_var)
        st.warning(f"以下变量方差为0，已自动剔除（不影响后续分析）：{zero_var}")

    if numeric_df.shape[1] < 2:
        st.error("参与KMO计算的变量数不足2个，请检查变量设置")
        return

    try:
        # 整体 KMO
        kmo_all, kmo_model = calculate_kmo(numeric_df)
        st.subheader("整体 KMO")
        st.write(round(kmo_model, 3))
        if kmo_model >= 0.8:
            st.success("优秀（≥0.8）")
        elif kmo_model >= 0.6:
            st.warning("尚可（0.6~0.8）")
        else:
            st.error("不适合因子分析（<0.6）")

        # Bartlett 检验
        chi, p = calculate_bartlett_sphericity(numeric_df)
        st.subheader("Bartlett 球形检验")
        st.write({"Chi-square": round(chi, 3), "p-value": round(p, 5)})
        if p < 0.05:
            st.success("适合进行因子分析（p < 0.05）")
        else:
            st.warning("不适合进行因子分析（p ≥ 0.05）")

        # 各变量 KMO
        st.subheader("各变量 KMO")
        kmo_df = pd.DataFrame({
            "变量": numeric_df.columns,
            "KMO": kmo_all
        })
        kmo_df["状态"] = kmo_df["KMO"].apply(
            lambda x: "⚠️ 建议删除" if x < 0.6 else ("⚠️ 勉强" if x < 0.7 else "✅ 良好")
        )
        st.dataframe(kmo_df.sort_values("KMO").reset_index(drop=True))

        # 手动删除
        st.subheader("删除变量")
        low_kmo_vars = kmo_df[kmo_df["KMO"] < 0.6]["变量"].tolist()
        if low_kmo_vars:
            st.warning(f"建议删除（KMO < 0.6）：{low_kmo_vars}")
        else:
            st.info("所有变量 KMO ≥ 0.6，无需删除")

        delete_kmo = []
        for _, row in kmo_df.iterrows():
            if st.checkbox(
                f"{row['变量']}（KMO = {round(row['KMO'], 3)}）",
                value=False,
                key=f"kmo_del_{row['变量']}"
            ):
                delete_kmo.append(row["变量"])

        if st.button("应用 KMO 筛选"):
            if delete_kmo:
                df_new = df.drop(columns=delete_kmo)
                st.session_state["df"] = df_new

                for col in delete_kmo:
                    for key in [f"map_{col}", f"assign_{col}", f"kmo_del_{col}"]:
                        if key in st.session_state:
                            del st.session_state[key]

                st.success(f"已删除：{delete_kmo}")
                st.rerun()
            else:
                st.info("未选择任何变量")

    except Exception as e:
        st.error(f"计算失败：{e}")


# =========================
# Step 5 SEM 建模
# =========================
def step_sem():
    if st.session_state.get("step2_stage") != "process":
        return

    st.header("Step 5: SEM 建模")

    df = st.session_state["df"]

    # =========================
    # 🤖 LLM 构念划分（llm)
    # =========================
    st.subheader("🤖 LLM构念划分")

    research_question = st.text_area(
        "请输入你的研究问题（用于自动构念划分）",
        placeholder="例如：驾驶行为如何影响风险？"
    )

    if st.button("生成构念方案"):

        if not research_question:
            st.warning("请先输入研究问题")
        else:
            try:
                from LLM_Construct import generate_constructs

                variables = st.session_state["df"].columns.tolist()

                result = generate_constructs(
                    research_question,
                    variables
                )

                st.session_state["construct_options"] = result
                st.success("已生成3种构念方案")

            except Exception as e:
                st.error(f"生成失败: {e}")

    options = st.session_state.get("construct_options", {})

    if options:
        st.subheader("选择构念方案")

        choice = st.radio(
            "请选择一个方案",
            list(options.keys())
        )

        selected_preview = options[choice]
        if selected_preview.get("subconstructs"):
            st.write("LLM 识别到的潜在子构念：")
            for parent, children in selected_preview["subconstructs"].items():
                st.write(f"  {parent} ← {children}")

    if options and st.button("应用该方案"):

        selected = options[choice]

        construct_dict = selected.get("constructs", {})
        subconstructs = selected.get("subconstructs", {})
        constructs = list(construct_dict.keys())

        for parent, children in subconstructs.items():
            if parent not in constructs:
                constructs.append(parent)
            for child in children:
                if child not in constructs:
                    constructs.append(child)

        mapping = {}
        for c, vars_ in construct_dict.items():
            for v in vars_:
                mapping[v] = c

        st.session_state["constructs"] = constructs
        st.session_state["measurement_model"] = mapping
        st.session_state["subconstruct_map"] = subconstructs
        st.session_state["exogenous_vars"] = selected.get("exogenous", [])
        for i, construct_name in enumerate(constructs):
            st.session_state[f"construct_{i}"] = construct_name

        if subconstructs:
            st.success("已自动填充构念、子构念关系与变量归属（可手动修改）")
        else:
            st.success("已自动填充构念与变量归属（可手动修改）")
        st.rerun()



    # 构念定义 + 变量归属 + EFA 诊断
    st.subheader("Step 5.1 构念定义")
    st.caption("⚠️ 构念名称请勿包含空格或特殊字符，建议用下划线，如 Perceptual_State")

    existing_constructs = st.session_state.get("constructs", [])
    default_construct_count = max(1, len(existing_constructs) or 3)
    n_constructs = st.number_input(
        "构念数量",
        min_value=1,
        max_value=100,
        value=default_construct_count,
        step=1
    )
    construct_names = []
    for i in range(n_constructs):
        default_name = existing_constructs[i] if i < len(existing_constructs) else f"Factor{i+1}"
        name = st.text_input(f"构念 {i+1} 名称", value=default_name, key=f"construct_{i}")
        construct_names.append(name)
    st.session_state["constructs"] = construct_names

    st.write("**变量归属（测量模型）**")

    all_vars = df.columns.tolist()

    if "measurement_model" not in st.session_state:
        st.session_state["measurement_model"] = {}
    mapping = st.session_state["measurement_model"]

    for c in construct_names:
        sub_map_for_assignment = st.session_state.get("subconstruct_map", {})
        child_constructs = sub_map_for_assignment.get(c, [])
        child_assigned_vars = [
            v for v, con in mapping.items()
            if con in child_constructs
        ]
        current = [v for v, con in mapping.items() if con == c]
        selected = st.multiselect(
            f"🔷 {c}",
            options=all_vars,
            default=[v for v in current if v in all_vars],
            key=f"assign_{c}"
        )
        selected = [v for v in selected if v not in child_assigned_vars]
        for v in all_vars:
            if mapping.get(v) == c:
                mapping[v] = None
        for v in selected:
            mapping[v] = c

    st.session_state["measurement_model"] = mapping

    unassigned = [v for v in all_vars if not mapping.get(v)]
    if unassigned:
        st.warning(f"以下变量未分配构念：{unassigned}")
    else:
        st.success("所有变量已分配完毕")

    st.write("**构念内 EFA / KMO / Bartlett 诊断**")
    st.caption("对每个构念分别做探索性因子分析、KMO 和 Bartlett 检验，并根据载荷、KMO 和潜在多因子结构给出删减变量或二阶构念建议。")

    df = st.session_state["df"]
    run_fa = st.button("运行构念内 EFA 诊断")

    if run_fa or st.session_state.get("fa_done", False):

        fa_results = {}
        efa_suggestions = {}

        for c in construct_names:
            indicators = [v for v, con in mapping.items() if con == c]
            if len(indicators) < 2:
                continue

            sub_df = df[indicators].select_dtypes(include='number').dropna()
            valid_indicators = sub_df.columns.tolist()
            if len(valid_indicators) < 2:
                st.warning(f"{c} 可用于 EFA 的数值型指标不足2个")
                continue
            if sub_df.shape[0] < 3:
                st.warning(f"{c} 完整样本数不足，无法稳定运行 EFA/KMO/Bartlett")
                continue

            try:
                from sklearn.decomposition import FactorAnalysis

                kmo_all, kmo_model = calculate_kmo(sub_df)
                bartlett_chi, bartlett_p = calculate_bartlett_sphericity(sub_df)

                corr = sub_df.corr()
                eigenvalues = np.linalg.eigvalsh(corr).tolist()
                eigenvalues = sorted(eigenvalues, reverse=True)
                suggested_factors = sum(ev > 1 for ev in eigenvalues)
                max_factors = max(1, min(len(valid_indicators) - 1, suggested_factors))

                fa = FactorAnalysis(n_components=1, random_state=0)
                fa.fit(sub_df)
                loadings = dict(zip(valid_indicators, fa.components_[0]))

                multi_factor_loadings = None
                if max_factors >= 2:
                    try:
                        from factor_analyzer import FactorAnalyzer
                        efa = FactorAnalyzer(n_factors=max_factors, rotation="varimax")
                        efa.fit(sub_df)
                        multi_factor_loadings = pd.DataFrame(
                            efa.loadings_,
                            index=valid_indicators,
                            columns=[f"Factor{i + 1}" for i in range(max_factors)]
                        )
                    except Exception:
                        multi_factor_loadings = None

                fa_results[c] = {
                    "loadings": loadings,
                    "kmo_model": kmo_model,
                    "kmo_all": dict(zip(valid_indicators, kmo_all)),
                    "bartlett_chi": bartlett_chi,
                    "bartlett_p": bartlett_p,
                    "eigenvalues": eigenvalues,
                    "suggested_factors": suggested_factors,
                    "multi_factor_loadings": multi_factor_loadings
                }
            except Exception as e:
                st.warning(f"{c} EFA/KMO/Bartlett 诊断失败：{e}")

        if fa_results:
            st.session_state["fa_done"] = True
            st.session_state["fa_results"] = fa_results

            for c, result in fa_results.items():
                loadings = result["loadings"]
                low_load = [v for v, l in loadings.items() if abs(l) < 0.4]
                low_kmo = [v for v, k in result["kmo_all"].items() if k < 0.6]
                needs_subconstruct = result["suggested_factors"] >= 2

                with st.expander(f"🔷 {c}（{len(loadings)}个变量）" + (
                f"  ⚠️ 建议检查" if (low_load or low_kmo or needs_subconstruct) else "  ✅ 结构较稳定")):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("整体 KMO", round(result["kmo_model"], 3))
                    col2.metric("Bartlett p", round(result["bartlett_p"], 5))
                    col3.metric("建议因子数", max(1, result["suggested_factors"]))

                    if result["kmo_model"] < 0.6:
                        st.warning("整体 KMO < 0.6：该构念的指标共同性偏弱，建议优先删减低 KMO 或低载荷变量。")
                    if result["bartlett_p"] >= 0.05:
                        st.warning("Bartlett 检验不显著：变量相关性不足，可能不适合做因子分析。")
                    if needs_subconstruct:
                        st.info(f"特征值提示可能存在 {result['suggested_factors']} 个因子，可考虑在 Step 5.2 中设置二阶构念。")

                    load_df = pd.DataFrame({
                        "变量": list(loadings.keys()),
                        "因子载荷": [round(l, 3) for l in loadings.values()],
                        "变量KMO": [round(result["kmo_all"][v], 3) for v in loadings.keys()],
                    })
                    load_df["状态"] = load_df.apply(
                        lambda row: "⚠️ 建议删除/调整" if abs(row["因子载荷"]) < 0.4 or row["变量KMO"] < 0.6
                        else ("⚠️ 偏低" if abs(row["因子载荷"]) < 0.6 or row["变量KMO"] < 0.7 else "✅ 良好"),
                        axis=1
                    )
                    st.dataframe(load_df.sort_values("因子载荷", key=abs, ascending=False).reset_index(drop=True))

                    delete_candidates = sorted(set(low_load + low_kmo))
                    if delete_candidates:
                        st.warning(f"建议从构念 {c} 中优先检查或删除：{delete_candidates}")
                        st.caption("请在上方该构念的变量归属 multiselect 中手动移除变量，然后重新运行 EFA 诊断。")

                    if result["multi_factor_loadings"] is not None:
                        st.write("多因子探索载荷（用于判断是否需要拆分子构念）：")
                        st.dataframe(result["multi_factor_loadings"].round(3))

                        proposed_groups = {}
                        for variable, row in result["multi_factor_loadings"].abs().iterrows():
                            proposed_groups.setdefault(row.idxmax(), []).append(variable)
                        proposed_groups = {k: v for k, v in proposed_groups.items() if len(v) >= 2}
                        if len(proposed_groups) >= 2:
                            efa_suggestions[c] = proposed_groups
                            st.info(f"可考虑将 {c} 拆为子构念：{proposed_groups}")

            st.session_state["efa_subconstruct_suggestions"] = efa_suggestions

    measurement_construct_names = list(construct_names)
    for con in mapping.values():
        if con and con not in measurement_construct_names:
            measurement_construct_names.append(con)

    singleton_constructs = {
        c: [v for v, con in mapping.items() if con == c]
        for c in measurement_construct_names
    }
    singleton_constructs = {c: vars_[0] for c, vars_ in singleton_constructs.items() if len(vars_) == 1}

    if singleton_constructs:
        st.write("**单指标构念处理**")
        st.caption("构念只剩一个变量时，不能作为常规潜变量测量模型。请选择从测量模型删除，或转为内生变量。")
        singleton_actions = {}
        for c, var in singleton_constructs.items():
            singleton_actions[c] = st.radio(
                f"{c} 只包含变量 [{var}]，请选择处理方式",
                ["转为内生变量", "从测量模型删除"],
                key=f"singleton_action_{c}",
                horizontal=True
            )

        if st.button("应用单指标构念处理"):
            current_endo = st.session_state.get("endogenous_vars", [])
            sub_map = st.session_state.get("subconstruct_map", {})
            for c, action in singleton_actions.items():
                var = singleton_constructs[c]
                mapping[var] = None
                if action == "转为内生变量" and var not in current_endo:
                    current_endo.append(var)
                for parent_name, children in list(sub_map.items()):
                    if c in children:
                        sub_map[parent_name] = [child for child in children if child != c]
                    if not sub_map[parent_name]:
                        del sub_map[parent_name]

            st.session_state["endogenous_vars"] = current_endo
            st.session_state["measurement_model"] = mapping
            st.session_state["subconstruct_map"] = sub_map
            st.success("已处理单指标构念")
            st.rerun()

    # 子构念
    st.subheader("Step 5.2 子构念（二阶构念）设置")

    if "subconstruct_map" not in st.session_state:
        st.session_state["subconstruct_map"] = {}
    sub_map = st.session_state["subconstruct_map"]

    efa_suggestions = st.session_state.get("efa_subconstruct_suggestions", {})
    if efa_suggestions:
        st.write("EFA 建议的潜在子构念划分：")
        for parent_name, groups in efa_suggestions.items():
            st.write(f"  {parent_name}: {groups}")

    parent = st.selectbox("父构念", construct_names, key="sub_parent")
    parent_vars = [v for v, con in mapping.items() if con == parent and v in all_vars]

    if parent_vars:
        new_sub_name = st.text_input(
            "新建子构念名称",
            value=f"{parent}_Sub{len(sub_map.get(parent, [])) + 1}",
            key="new_subconstruct_name"
        )
        sub_vars = st.multiselect(
            "该子构念包含的变量",
            parent_vars,
            key="new_subconstruct_vars"
        )

        if st.button("添加子构念"):
            new_sub_name = new_sub_name.strip()
            existing_children = [child for children in sub_map.values() for child in children]

            if not new_sub_name:
                st.warning("请先输入子构念名称")
            elif new_sub_name in construct_names or new_sub_name in existing_children:
                st.warning("该子构念名称已存在，请更换名称")
            elif any(ch in new_sub_name for ch in [' ', '-', '/', '(', ')', '.']):
                st.warning("子构念名称请勿包含空格或特殊字符，建议使用下划线")
            elif len(sub_vars) < 2:
                st.warning("子构念至少建议包含2个变量")
            else:
                for v in sub_vars:
                    mapping[v] = new_sub_name

                sub_map.setdefault(parent, [])
                sub_map[parent].append(new_sub_name)

                st.session_state["measurement_model"] = mapping
                st.session_state["subconstruct_map"] = sub_map
                st.success(f"已创建子构念：{parent} ← {new_sub_name}")
                st.rerun()
    else:
        st.info("该父构念当前没有可分配给新子构念的直接变量。若变量已被拆到子构念中，可继续查看或清空当前子构念结构。")

    if sub_map:
        st.write("当前子构念结构：")
        for k, children in sub_map.items():
            st.write(f"  {k} ← {children}")
            parent_direct_vars = [v for v, con in mapping.items() if con == k]
            if parent_direct_vars:
                st.info(f"{k} 仍有变量未分配到子构念，这些变量会作为父构念的直接观测变量进入 SEM：{parent_direct_vars}")
            for child in children:
                child_vars = [v for v, con in mapping.items() if con == child]
                available_child_vars = [
                    v for v in all_vars
                    if mapping.get(v) in [k, child]
                ]
                updated_child_vars = st.multiselect(
                    f"{child} 的变量归属",
                    available_child_vars,
                    default=[v for v in child_vars if v in available_child_vars],
                    key=f"sub_assign_{k}_{child}"
                )

                if set(updated_child_vars) != set(child_vars):
                    for v in available_child_vars:
                        if mapping.get(v) == child:
                            mapping[v] = k
                    for v in updated_child_vars:
                        mapping[v] = child
                    st.session_state["measurement_model"] = mapping

                if len(updated_child_vars) < 2:
                    st.warning(f"{child} 当前变量少于2个，运行 SEM 时不会作为有效子构念进入二阶模型。")
        if st.button("清空子构念设置"):
            for parent_name, children in sub_map.items():
                for child in children:
                    for v, con in list(mapping.items()):
                        if con == child:
                            mapping[v] = parent_name
            st.session_state["measurement_model"] = mapping
            st.session_state["subconstruct_map"] = {}
            st.rerun()
    else:
        st.info("尚未设置子构念")

    # =========================
    # Step 5.5 CFA 验证
    # =========================
    st.subheader("Step 5.5 CFA 验证")
    st.caption("在进入结构路径设定前，对每个构念单独运行确认性因子分析（CFA），用于检查测量模型拟合和标准化载荷。")

    measurement_construct_names = list(construct_names)
    for con in mapping.values():
        if con and con not in measurement_construct_names:
            measurement_construct_names.append(con)

    cfa_candidates = {
        c: [v for v, con in mapping.items() if con == c and v in df.columns]
        for c in measurement_construct_names
    }
    cfa_candidates = {c: vars_ for c, vars_ in cfa_candidates.items() if len(vars_) >= 2}

    if not cfa_candidates:
        st.info("暂无可运行 CFA 的构念。每个构念至少需要 2 个观测变量。")
    else:
        cfa_target = st.selectbox(
            "选择 CFA 构念",
            ["全部构念"] + list(cfa_candidates.keys()),
            key="cfa_target"
        )
        cfa_estimator = st.selectbox(
            "CFA 估计方法",
            ["ML", "MLR", "GLS"],
            key="cfa_estimator_sel"
        )
        st.caption("ML/MLR 使用 FIML 处理缺失值；GLS 需要完整样本，将自动删除含缺失值的行。")

        short_constructs = [c for c, vars_ in cfa_candidates.items() if len(vars_) == 2]
        if short_constructs:
            st.warning(f"以下构念只有 2 个观测变量，CFA 可能识别不足或拟合指标信息有限：{short_constructs}")

        if st.button("运行 CFA 验证"):
            import os

            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, "sem_data.csv")
            data_path_r = data_path.replace("\\", "/")
            r_script_path = os.path.join(base_dir, "cfa_check.R")

            df.to_csv(data_path, index=False)

            selected_constructs = (
                list(cfa_candidates.keys())
                if cfa_target == "全部构念"
                else [cfa_target]
            )

            cfa_outputs = {}

            for c in selected_constructs:
                indicators = cfa_candidates[c]
                model_str = build_single_construct_cfa_model(c, indicators)

                if not model_str.strip():
                    cfa_outputs[c] = ("", "模型为空，无法运行 CFA。")
                    continue

                cfa_script = f"""
library(lavaan)
data_raw <- read.csv("{data_path_r}", check.names = FALSE)
data <- data_raw[, sapply(data_raw, is.numeric), drop = FALSE]

model <- '
{model_str}
'

estimator_name <- "{cfa_estimator}"

if (estimator_name == "GLS") {{
  data <- na.omit(data)
  fit <- cfa(model, data = data, estimator = estimator_name, std.lv = FALSE)
}} else {{
  fit <- cfa(model, data = data, estimator = estimator_name, std.lv = FALSE, missing = "fiml")
}}

cat("MODEL\\n")
cat(model)
cat("\\n\\nESTIMATOR\\n")
cat(estimator_name)
cat("\\nN_USED\\n")
cat(lavInspect(fit, "nobs"))
cat("\\n\\nFIT_MEASURES\\n")
print(round(fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr")), 4))
cat("\\n\\nSTANDARDIZED_LOADINGS\\n")
std <- standardizedSolution(fit)
print(std[std$op == "=~", c("lhs", "rhs", "est.std", "pvalue")])
cat("\\n\\nSUMMARY\\n")
print(summary(fit, fit.measures = TRUE, standardized = TRUE))
"""

                with open(r_script_path, "w", encoding="utf-8") as f:
                    f.write(cfa_script)

                out, err = run_r_script(r_script_path)
                cfa_outputs[c] = (out, err)

            st.session_state["cfa_outputs"] = cfa_outputs

        if st.session_state.get("cfa_outputs"):
            for c, (out, err) in st.session_state["cfa_outputs"].items():
                with st.expander(f"CFA 结果：{c}", expanded=True):
                    if out:
                        st.text(out)
                    else:
                        st.text("无输出")
                    if err:
                        st.error(err)
#路径设定
    import time
    import uuid

    st.subheader("Step 5.6 路径设定")

    valid_sources = construct_names + st.session_state.get("exogenous_vars", []) + st.session_state.get(
        "endogenous_vars", [])
    valid_targets = construct_names + st.session_state.get("endogenous_vars", [])

    if not valid_sources or not valid_targets:
        st.warning("请先在 Step 3 设置外生/内生变量")

    # =========================
    # ✅ LLM 自动路径（新增）
    # =========================
    if st.button("🤖 LLM推荐路径"):

        constructs = construct_names
        exo = st.session_state.get("exogenous_vars", [])
        endo = st.session_state.get("endogenous_vars", [])

        try:
            from LLM_SEM import generate_sem_paths  # 你的LLM模块

            paths = generate_sem_paths(constructs, exo, endo)

            if not paths:
                st.warning("LLM未生成路径")
            else:
                st.session_state["paths"] = paths
                st.success(f"生成 {len(paths)} 条路径（可手动修改）")
                st.rerun()

        except Exception as e:
            st.error(f"LLM生成失败: {e}")

    # =========================
    # 初始化 paths
    # =========================
    if "paths" not in st.session_state:
        st.session_state["paths"] = []

    # 确保每个path有uid
    for path in st.session_state["paths"]:
        if "uid" not in path:
            path["uid"] = uuid.uuid4().hex

    # =========================
    # 手动添加路径（保留）
    # =========================
    st.write("**添加新路径：**")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        new_src = st.selectbox("起点", valid_sources, key="new_path_src")

    with col2:
        new_tgt = st.selectbox("终点", valid_targets, key="new_path_tgt")

    with col3:
        st.write("")
        st.write("")
        if st.button("➕ 添加"):
            st.session_state["paths"].append({
                "from": new_src,
                "to": new_tgt,
                "uid": uuid.uuid4().hex
            })
            st.rerun()

    # =========================
    # 显示 + 删除（保留）
    # =========================
    if st.session_state["paths"]:
        st.write("**当前路径：**")

        for path in st.session_state["paths"]:
            col1, col2 = st.columns([5, 1])

            with col1:
                st.write(f"{path['from']} → {path['to']}")

            with col2:
                if st.button("🗑️", key=f"del_{path['uid']}"):
                    st.session_state["paths"] = [
                        p for p in st.session_state["paths"]
                        if p["uid"] != path["uid"]
                    ]
                    st.rerun()
    else:
        st.info("暂无路径")
    # =========================
    # 模型设置（保持你原样）
    # =========================
    st.subheader("Step 5.7 模型设置")

    st.session_state["ident_method"] = st.selectbox(
        "识别方式",
        ["fix_loading", "fix_variance"],
        key="ident_method_sel"
    )
    st.caption("fix_loading：固定每个潜变量第一条载荷为 1；fix_variance：释放第一条载荷并固定潜变量方差为 1。")

    estimator = st.selectbox(
        "估计方法",
        ["ML", "MLR", "GLS"],
        key="estimator_sel"
    )
    st.caption("ML/MLR 使用 FIML 处理缺失值；GLS 需要完整样本，将自动删除含缺失值的行。")

    st.subheader("残差相关设置（根据修正指数添加）")
    st.caption("运行'获取修正指数'后，在此手动添加需要的残差相关，格式：变量1,变量2")

    if "residual_covs" not in st.session_state:
        st.session_state["residual_covs"] = []

    col1, col2 = st.columns([3, 1])
    with col1:
        new_cov = st.text_input("添加残差相关（格式：变量1,变量2）", key="new_rescov")
    with col2:
        if st.button("➕ 添加", key="add_path_btn"):
            if "," in new_cov:
                parts = [p.strip() for p in new_cov.split(",")]
                if len(parts) == 2 and parts[0] and parts[1]:
                    pair = tuple(parts)
                    if pair not in st.session_state["residual_covs"]:
                        st.session_state["residual_covs"].append(pair)
                        st.rerun()
                    else:
                        st.warning("已存在")
            else:
                st.warning("格式错误，请用逗号分隔")

    if st.session_state["residual_covs"]:
        st.write("当前残差相关：")
        to_remove = []
        for pair in st.session_state["residual_covs"]:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{pair[0]} ~~ {pair[1]}")
            with col2:
                if st.button("删除", key=f"del_rescov_{pair[0]}_{pair[1]}"):
                    to_remove.append(pair)
        for pair in to_remove:
            st.session_state["residual_covs"].remove(pair)
        if to_remove:
            st.rerun()
    else:
        st.info("暂无残差相关设置")

    # 运行SEM
    if st.button("获取修正指数"):
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, "sem_data.csv").replace("\\", "/")
        r_script_path = os.path.join(base_dir, "mi_check.R")

        model_str = build_lavaan_model()
        estimator = st.session_state.get("estimator_sel", "ML")

        mi_script = f"""
    library(lavaan)
    data_raw <- read.csv("{data_path}", check.names = FALSE)
    data <- data_raw[, sapply(data_raw, is.numeric), drop = FALSE]

    model <- '
    {model_str}
    '

    estimator_name <- "{estimator}"

    if (estimator_name == "GLS") {{
      data <- na.omit(data)
      fit <- suppressWarnings(sem(model, data = data, estimator = estimator_name))
    }} else {{
      fit <- suppressWarnings(sem(model, data = data, estimator = estimator_name, missing = "fiml"))
    }}

    cat("ESTIMATOR\\n")
    cat(estimator_name)
    cat("\\nN_USED\\n")
    cat(lavInspect(fit, "nobs"))
    cat("\\n\\n")

    mi <- suppressWarnings(modindices(fit, sort.=TRUE, maximum.number=20))
    print(mi)
    """

        with open(r_script_path, "w", encoding="utf-8") as f:
            mi_script_clean = mi_script
            f.write(mi_script_clean)

        out, err = run_r_script(r_script_path)

        st.subheader("修正指数（前20）")
        st.text(out if out else "无输出")
        if err:
            st.text(f"信息：{err}")


    if st.button("运行 SEM 分析"):
        import os

        model_str = build_lavaan_model()

        if not model_str.strip():
            st.error("模型为空，请先设置构念和变量归属")
            return

        st.subheader("生成的 lavaan 模型")
        st.code(model_str, language="r")

        # 绝对路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        r_script_path = os.path.join(base_dir, "sem_model.R")
        data_path = os.path.join(base_dir, "sem_data.csv")
        data_path_r = data_path.replace("\\", "/")

        df.to_csv(data_path, index=False)

        r_script = f"""
library(lavaan)
data_raw <- read.csv("{data_path_r}", check.names = FALSE)
data <- data_raw[, sapply(data_raw, is.numeric), drop = FALSE]

model <- '
{model_str}
'

estimator_name <- "{estimator}"

if (estimator_name == "GLS") {{
  data <- na.omit(data)
  fit <- sem(model, data = data, estimator = estimator_name)
}} else {{
  fit <- sem(model, data = data, estimator = estimator_name, missing = "fiml")
}}

cat("ESTIMATOR\\n")
cat(estimator_name)
cat("\\nN_USED\\n")
cat(lavInspect(fit, "nobs"))
cat("\\n\\n")
summary(fit, fit.measures=TRUE, standardized=TRUE)

"""

        with open(r_script_path, "w", encoding="utf-8") as f:
            f.write(r_script)

        out, err = run_r_script(r_script_path)

        st.subheader("输出结果")
        st.text(out if out else "（无输出）")
        if err:
            st.error(err)


# =========================
# 主流程
# =========================
def main():
    init_session()

    st.title("📊 SEM分析工具")

    step_upload()

    if st.session_state["df"] is not None:
        step_processing()
        step_var_setup()
        step_kmo()
        step_sem()


if __name__ == "__main__":
    main()
