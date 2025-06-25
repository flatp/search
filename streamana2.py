import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

plt.rcParams['font.family'] = 'MS Gothic'

# データ読み込み
df = pd.read_csv("popstreamlabel2.csv")

# 1. 人気度と知名度のギャップ（両時点）
df["202208ギャップ"] = df["202208知名度"] - df["202208人気度"]
df["202302ギャップ"] = df["202302知名度"] - df["202302人気度"]

df["知名度ギャップ"] = df["202302知名度"] - df["202208知名度"]
df["人気度ギャップ"] = df["202302人気度"] - df["202208人気度"]

# 2. ギャップの変化量
df["ギャップ変化量"] = df["202302ギャップ"] - df["202208ギャップ"]

# 3. ダイレクトアクセスの時系列トレンド
da_cols = [col for col in df.columns if "BR" in col]
df_da = df[["name"] + da_cols].set_index("name")
df_da_T = df_da.T  # 転置して月別分析がしやすいようにする
df_da_T.index = pd.to_datetime(df_da_T.index.str.replace("BR", ""), format="%Y%m")

# 4. 各人物のDA平均変化率
df["BR平均"] = df[da_cols].mean(axis=1)
df["BR変化率"] = (df["202302BR"] - df["202208BR"]) / df["202208BR"] * 100

# 5. 人気・知名度とDA変化率の相関
corr = df[["202208人気度", "202208知名度", "202302人気度", "202302知名度", "BR変化率"]].corr()

# 6. 可視化
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="202302ギャップ", y="BR変化率")
plt.title("ギャップとBR変化率の関係")
plt.xlabel("人気度と知名度のギャップ（2023年2月）")
plt.ylabel("BR変化率（2022年8月→2023年2月）")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="知名度ギャップ", y="BR変化率")
plt.title("知名度変化とBR変化率の関係")
plt.xlabel("知名度変化量")
plt.ylabel("BR変化率（2022年8月→2023年2月）")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="人気度ギャップ", y="BR変化率")
plt.title("人気度変化とBR変化率の関係")
plt.xlabel("人気度変化量")
plt.ylabel("BR変化率（2022年8月→2023年2月）")
plt.grid(True)
plt.tight_layout()
plt.show()

# 時系列推移（例：ランダムな5名を表示）
import random
sample_names = random.sample(list(df["name"]), 20)
df_da_T[sample_names].plot(figsize=(12, 6), title="ランダム20名のBR推移")
plt.ylabel("BR数")
plt.xlabel("月")
plt.grid(True)
plt.tight_layout()
plt.show()

# 相関行列ヒートマップ
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("人気・知名度・BR変化率の相関")
plt.tight_layout()
plt.show()


# DA列（各月のダイレクトアクセス数）を取得
da_cols = [col for col in df.columns if "BR" in col]
x = np.arange(len(da_cols)).reshape(-1, 1)  # 時系列インデックス

# 傾き（トレンド）を算出
trends = []
for _, row in df.iterrows():
    y = row[da_cols].values.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    trend = model.coef_[0][0]  # 傾き（増加 or 減少）
    trends.append(trend)

df["BRトレンド傾き"] = trends
df["BRトレンド分類"] = df["BRトレンド傾き"].apply(lambda v: "上昇" if v > 0 else "下降" if v < 0 else "横ばい")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="BRトレンド傾き", y="知名度ギャップ")
plt.title("BRトレンドと知名度変化の関係")
plt.xlabel("BRトレンド")
plt.ylabel("知名度変化量")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="BRトレンド傾き", y="人気度ギャップ")
plt.title("BRトレンドと人気度変化の関係")
plt.xlabel("BRトレンド")
plt.ylabel("人気度変化量")
plt.grid(True)
plt.tight_layout()
plt.show()

features = [
    "202208人気度", "202208知名度",
    "202302人気度", "202302知名度",
    "202208ギャップ", "202302ギャップ", "ギャップ変化量"
]

# 相関行列の中からトレンド傾きとの相関を抽出
feature_corr = df[features + ["BRトレンド傾き"]].corr()["BRトレンド傾き"].sort_values(ascending=False)
print(feature_corr)