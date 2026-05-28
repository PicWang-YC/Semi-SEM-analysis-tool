import pandas as pd
import numpy as np
from Variable_screening import kmo_bartlett_test

if __name__ == "__main__":
    file_path = r"D:/CogTest_DSTest_Data/大分テスト/Carmoving and CarOperation/Screened_Data_Safe_Window5Min2 - 副本.csv"   # 改成你的文件路径
    df = pd.read_csv(file_path)

    # 只保留数值型变量，不然 corr / OLS / VIF 可能报错
    df = df.select_dtypes(include=[np.number]).dropna()

    final_df = kmo_bartlett_test(df)