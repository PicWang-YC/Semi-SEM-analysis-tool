import streamlit as st
import pandas as pd

st.title("DEBUG TEST")

# =========================
# Step 1
# =========================
uploaded_file = st.file_uploader("上传文件", type=["csv", "xlsx"])

if uploaded_file is not None:

    if "df" not in st.session_state:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state["df"] = df
        st.session_state["step2_stage"] = "delete"

# =========================
# DEBUG
# =========================
st.write("df in session:", "df" in st.session_state)
st.write("stage:", st.session_state.get("step2_stage"))

# =========================
# Step 2
# =========================
df = st.session_state.get("df")

if df is None:
    st.warning("没有数据")
    st.stop()

if "step2_stage" not in st.session_state:
    st.session_state["step2_stage"] = "delete"

stage = st.session_state["step2_stage"]

st.write("当前阶段:", stage)

# =========================
# 删除变量
# =========================
if stage == "delete":

    st.subheader("删除变量")

    cols = df.columns.tolist()
    delete_cols = []

    for col in cols:
        if st.checkbox(col):
            delete_cols.append(col)

    if st.button("删除"):

        st.session_state["df"] = df.drop(columns=delete_cols)
        st.session_state["step2_stage"] = "process"

        st.rerun()

# =========================
# 数据处理
# =========================
elif stage == "process":

    st.subheader("处理阶段")

    st.write("当前列:", st.session_state["df"].columns.tolist())