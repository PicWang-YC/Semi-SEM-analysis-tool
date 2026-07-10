import pandas as pd
import numpy as np

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report = {}

    def analyze(self):
        df = self.df

        # =========================
        # 1. 基本信息
        # =========================
        self.report["shape"] = df.shape
        self.report["dtypes"] = df.dtypes.astype(str).to_dict()

        # =========================
        # 2. 缺失值分析
        # =========================
        missing_count = df.isnull().sum()
        missing_ratio = (missing_count / len(df)).round(4)

        self.report["missing"] = {
            "count": missing_count.to_dict(),
            "ratio": missing_ratio.to_dict()
        }

        # =========================
        # 3. 重复值
        # =========================
        self.report["duplicates"] = int(df.duplicated().sum())

        # =========================
        # 4. 数值列统计 + 分布
        # =========================
        numeric_summary = {}
        outliers_info = {}

        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            series = df[col].dropna()

            if len(series) == 0:
                continue

            # 基本统计
            desc = {
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "median": series.median()
            }

            # 📊 分布特征（重点）
            desc["skewness"] = series.skew()
            desc["kurtosis"] = series.kurt()

            numeric_summary[col] = desc

            # =========================
            # 5. 异常值（IQR）
            # =========================
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = ((series < lower) | (series > upper)).sum()

            outliers_info[col] = {
                "count": int(outliers),
                "lower_bound": float(lower),
                "upper_bound": float(upper)
            }

        self.report["numeric_summary"] = numeric_summary
        self.report["outliers"] = outliers_info

        # =========================
        # 6. 分类列统计（可选加分）
        # =========================
        categorical_summary = {}
        cat_cols = df.select_dtypes(include="object").columns

        for col in cat_cols:
            categorical_summary[col] = {
                "unique": df[col].nunique(),
                "top": df[col].mode().iloc[0] if not df[col].mode().empty else None
            }

        self.report["categorical_summary"] = categorical_summary

        return self.report

    def get_report(self):
        return self.report
