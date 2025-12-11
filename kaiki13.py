import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

plt.rcParams["font.family"] = "MS Gothic"

# ==============================
# 1. データ読み込み
# ==============================
df = pd.read_csv("alllist-past.csv")

# ==============================
# 2. 特徴量の作成
# ==============================
df["100-202208知名度"] = 100 - df["202208知名度"]
df["log(100-202208知名度)"] = np.log10(df["100-202208知名度"] + 1)
df["log(100-202208知名度)^2"] = df["log(100-202208知名度)"] ** 2
df["log(100-202208知名度)*weightedPV1"] = df["log(100-202208知名度)"] * df["weightedPV1"]
df["(100-202208知名度)*weightedPV1"] = df["100-202208知名度"] * df["weightedPV1"]

ex = [
    "100-202208知名度",
    "log(100-202208知名度)*weightedPV1",
    "(100-202208知名度)*weightedPV1"
]

# ==============================
# 3. 仮の係数 a, b, c を設定
#    （ここを書き換えれば何度でも試せる）
# ==============================
a = -0.6
b = -0.0008
c = 0.00006

# ==============================
# 4. y' = a*x1 + b*x2 + c*x3 を計算
# ==============================
df["y_hat"] = (
    a * df[ex[0]] +
    b * df[ex[1]] +
    c * df[ex[2]]
)

# ==============================
# 5. 実測値 y を計算
# ==============================
df["y"] = df["aware_diff"] * (101 - df["202208知名度"]) / np.log10(df["202208知名度"] + 1)

# ==============================
# 6. 相関係数・R²
# ==============================
corr = df["y"].corr(df["y_hat"])
r2 = r2_score(df["y"], df["y_hat"])

print("相関係数:", corr)
print("R^2:", r2)

# ==============================
# 7. 散布図プロット
# ==============================
plt.figure(figsize=(6, 6))
plt.scatter(df["y"], df["y_hat"], alpha=0.6)

min_val = min(df["y"].min(), df["y_hat"].min())
max_val = max(df["y"].max(), df["y_hat"].max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', linewidth=2, label="y = x")

formula_text = f"y' = {a}×{ex[0]} + {b}×{ex[1]} + {c}×{ex[2]}"

plt.text(
    0.05, 0.95,
    formula_text + f"\nCorr = {corr:.3f}\nR² = {r2:.3f}",
    transform=plt.gca().transAxes,
    fontsize=5.5,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.8)
)

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Actual y")
plt.ylabel("Predicted y'")
plt.title("y vs y'")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"model4.png")
plt.show()