这是一个基于 **Streamlit** 构建的交互式结构方程模型（SEM）分析工具。它将 Python 的动态交互、大语言模型（LLM）的智能推演能力，与 R 语言（`lavaan` 包）的高精度统计核心结合，实现了全流程、向导式的潜变量建模与路径分析探索。

## ATTENTION 
本工具仅限于科研前期探索路径使用，暂时不能作为正式分析工具

##  核心功能亮点
连续性数据分析（SEM_tool）
1. 智能数据清洗 (Data cleaning)**：支持缺失值自动填充、基于 IQR 的极值截断、Box-Cox/Log 偏态转换，并能根据偏度和异常值指标提供处理推荐。
2. 构念内 EFA 自动化诊断：在正式运行 SEM 前，自动对各个构念进行单因子 FactorAnalysis 诊断，提供 KMO、Bartlett 球形检验及因子载荷矩阵，并智能给出变量删减或拆分子构念的建议。
（可选）二阶潜变量与单指标处理：支持复杂的二阶构念（二阶模型）#建议仅限于成熟理论支持下使用
3. 路径分析和修正系数可视化：添加残差相关优化模型
4. （可选）LLM 智能构念赋能 (AI-Assisted)**：预留了大模型接口（`LLM_Construct` 和 `LLM_Path`），可根据研究问题自动生成 3 种推荐的构念划分方案。提示词自行变更。

里克特量表分析模块（SEM_tool_Likert）
1. 智能数据清洗 (Data cleaning)：可选5点/7点里克特量表，可删除或者补全缺失题项。
2. 构念内 EFA 自动化诊断：在正式运行 SEM 前，自动对各个构念进行单因子 FactorAnalysis 诊断，提供 KMO、Bartlett 球形检验，峰度偏度计算及因子载荷矩阵，并智能给出变量删减或题项打包的建议。
（可选）题项打包功能：支持将复杂题项打包减少题项，使模型更易识别。
3. CFA验证模型稳定性，路径分析和修正系数可视化：添加残差相关优化模型
4. 引入boostrap可进行中介分析
5. （可选）LLM 智能构念赋能 (AI-Assisted)**：预留了大模型接口（`LLM_Construct` 和 `LLM_Path`），可根据研究问题自动生成 3 种推荐的构念划分方案。提示词自行变更。

## 核心python库
streamlit
pandas
numpy
scikit-learn
factor-analyzer
scipy

## R-4.5.1
lavaan
（如果报错，可能是版本不符合，缺少必要的包，Rscript.exe路径和导出路径请在代码中 run_r_script 函数定义中自行修改）

## 📂 项目结构
```text
├── SEM_tool.py          # Streamlit 主程序入口
├── SEM_tool_Likert.py   # Streamlit 主程序入口（里克特量表）
├── Data_cleaning.py     # 负责数据预处理与分析的 DataAnalyzer 类
├── LLM_Construct.py     # 调用大模型生成构念划分方案的模块
├── LLM_Path.py          # 调用大模型生成 SEM 路径关系的模块 
└── .gitignore           # Git 忽略文件配置
