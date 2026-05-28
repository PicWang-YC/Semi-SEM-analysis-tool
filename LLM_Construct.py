import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def generate_constructs(research_question, variables):

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are designing latent constructs for SEM.

Research goal:
{research_question}

Variables:
{variables}

Rules:
1. Group variables into meaningful constructs
2. Construct names must be valid R variable names (no spaces, no special characters)
3. Identify exogenous variables (not part of constructs)
4. Provide 3 different grouping schemes

Output JSON:
{{
  "model_1": {{
    "constructs": {{
      "ConstructA": ["var1", "var2"],
      "ConstructB": ["var3"]
    }},
    "exogenous": ["varX"]
  }},
  "model_2": {{...}},
  "model_3": {{...}}
}}
""")

    chain = prompt | llm | JsonOutputParser()

    result = chain.invoke({
        "research_question": research_question,
        "variables": variables
    })

    return result