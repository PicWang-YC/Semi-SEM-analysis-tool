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
4. After grouping variables into constructs, judge whether any construct may contain meaningful subconstructs
5. Use subconstructs when variables in the same broad construct reflect distinct dimensions, mechanisms, or statistics
6. If a broad construct has subconstructs, assign variables to the subconstructs, keep the broad parent construct with an empty variable list, and map parent -> subconstruct names in "subconstructs"
7. Do not create subconstructs unless there is a clear conceptual reason
8. Provide 3 different grouping schemes

Example:
- Eye diameter mean and median can belong to Pupil_Size
- Eye diameter standard deviation and coefficient of variation can belong to Pupil_Variability
- Pupil_Size and Pupil_Variability can both be subconstructs of Pupil_Change

Output JSON:
{{
  "model_1": {{
    "constructs": {{
      "BroadConstruct": [],
      "SubConstructA": ["var1", "var2"],
      "SubConstructB": ["var3", "var4"],
      "ConstructC": ["var5", "var6"]
    }},
    "subconstructs": {{
      "BroadConstruct": ["SubConstructA", "SubConstructB"]
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
