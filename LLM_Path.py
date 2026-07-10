import uuid
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# =========================
# 初始化 LLM
# =========================
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",   # 可换
        temperature=0
    )


# =========================
# 构建 Chain
# =========================
def build_chain():

    prompt = ChatPromptTemplate.from_template("""
You are designing a structural equation model (SEM).

Variables:
Constructs: {constructs}
Exogenous: {exogenous_vars}
Endogenous: {endogenous_vars}

Rules:
1. The path must be meaningful and interpretable
2. Exogenous → Constructs or Endogenous
3. Constructs → Constructs
4. Constructs → Endogenous
5. Endogenous variables do NOT cause other variables
6. Avoid cycles (no A→B and B→A)
7. Keep the model simple (not too many paths)

Output ONLY JSON:
{{
  "paths": [
    {{"from": "...", "to": "..."}}
  ]
}}
""")

    llm = get_llm()
    parser = JsonOutputParser()

    chain = prompt | llm | parser

    return chain


# =========================
# 路径生成函数（对外调用）
# =========================
def generate_sem_paths(constructs, exogenous_vars, endogenous_vars):

    chain = build_chain()

    result = chain.invoke({
        "constructs": constructs,
        "exogenous_vars": exogenous_vars,
        "endogenous_vars": endogenous_vars
    })

    raw_paths = result.get("paths", [])

    # =========================
    # 清洗 + 去重 + 加UID
    # =========================
    clean_paths = []
    seen = set()

    for p in raw_paths:
        if "from" in p and "to" in p:
            src = p["from"]
            tgt = p["to"]

            if src != tgt:
                key = (src, tgt)
                reverse = (tgt, src)

                if key not in seen and reverse not in seen:
                    seen.add(key)

                    clean_paths.append({
                        "from": src,
                        "to": tgt,
                        "uid": uuid.uuid4().hex
                    })

    # =========================
    # 限制路径数量（防爆）
    # =========================
    MAX_PATHS = 20
    return clean_paths[:MAX_PATHS]