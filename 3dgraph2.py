import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.rcParams["font.family"] = "MS Gothic"

# =====================
# CSV 読み込み
# =====================
df = pd.read_csv("alllist-past.csv")

# =====================
# x, y, z を取得
# =====================
x = df["202208知名度"]
y = df["weightedPV1"]
z = df["aware_diff"] * (101 - df["202208知名度"]) / np.log10(df["202208知名度"] + 1)

# =====================
# 3Dプロット
# =====================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(x, y, z, s=10, alpha=0.6)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("3D scatter plot of alllist-past.csv")

plt.show()
