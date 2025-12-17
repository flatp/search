import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# パラメータ（仮）
# =====================
a = -0.6
b = -0.0008
c = 0.00006

# =====================
# CSV 読み込み（実測データ）
# =====================
df = pd.read_csv("alllist-past.csv")

df["z"] = df["aware_diff"] * (101 - df["202208知名度"]) / np.log10(df["202208知名度"] + 1)

x_data = df["202208知名度"].values
y_data = df["weightedPV1"].values
z_data = df["z"].values

# =====================
# 理論曲面用グリッド
# =====================
x = np.linspace(0, 99.9, 80)          # log(0)回避
y = np.linspace(0, 500000, 80)

X, Y = np.meshgrid(x, y)

Z = (
    a * (100 - X)
    + b * np.log10(100 - X) * Y
    + c * (100 - X) * Y
)

# =====================
# 3Dプロット
# =====================
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

# --- 理論曲面 ---
surf = ax.plot_surface(
    X, Y, Z,
    cmap="viridis",
    alpha=0.6,
    edgecolor="none"
)

# --- 実測点 ---
ax.scatter(
    x_data,
    y_data,
    z_data,
    color="red",
    s=15,
    alpha=0.8,
    label="Observed data"
)

# =====================
# ラベル設定
# =====================
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Observed data and theoretical surface")

ax.legend()

plt.show()
