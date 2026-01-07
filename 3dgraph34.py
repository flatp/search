import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "MS Gothic"

# =====================
# 1. CSV 読み込み
# =====================
df = pd.read_csv("alllist-past2.csv")

df["z"] = df["aware_diff_past"]

x_data = df["201902知名度"].values
y_data = df["weightedPV1_past"].values
z_data = df["z"].values

# =====================
# 2. 知名度レンジ & 係数設定
# =====================
bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

abc_dict = {
    (0, 20):  {"a": -21, "b": -0.009, "c": 0.0003, "d":5},
    (20, 40): {"a": -290, "b": -0.003, "c": 0.0001, "d":172},
    (40, 60): {"a": -42,  "b": -0.002, "c": 0.0001, "d":15},
    (60, 80): {"a": -4.6,  "b": 0.000034, "c": 0.000044, "d":-6},
    (80, 100):{"a": -19,  "b": -0.0002, "c": 0.00006, "d":-1},
}

# =====================
# 3. 1つの 3D 座標を作成
# =====================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

# カラーマップをレンジごとに少しずらす
cmaps = ["Blues", "Greens", "Oranges", "Purples", "Reds"]

# =====================
# 4. レンジごとに同一座標へ重ね描き
# =====================
for (low, high), cmap in zip(bins, cmaps):

    a = abc_dict[(low, high)]["a"]
    b = abc_dict[(low, high)]["b"]
    c = abc_dict[(low, high)]["c"]
    d = abc_dict[(low, high)]["d"]

    # --- 実測点 ---
    mask = (df["201902知名度"] >= low) & (df["201902知名度"] < high)
    x_obs = x_data[mask]
    y_obs = y_data[mask]
    z_obs = z_data[mask]

    # --- 理論曲面 ---
    x = np.linspace(low, min(high, 99.9), 60)
    y = np.linspace(0, 2_000_000, 60)
    X, Y = np.meshgrid(x, y)

    Z = (
        a * (100 - X)
        + b * np.log10(101 - X) * Y
        + c * (100 - X) * Y
    ) * np.log10(X + 1) / (201 - X) + d

    ax.plot_surface(
        X, Y, Z,
        cmap=cmap,
        alpha=0.35,
        edgecolor="none"
    )

    ax.scatter(
        x_obs,
        y_obs,
        z_obs,
        s=15,
        alpha=0.8,
        label=f"x ∈ [{low},{high})"
    )

# =====================
# 5. 軸・表示設定
# =====================
ax.set_xlabel("201902知名度 (x)")
ax.set_ylabel("weightedPV1_past (y)")
ax.set_zlabel("aware_diff_past (z)")

ax.set_title("知名度レンジ別 理論曲面 & 実測データ（単一座標）")
ax.set_zlim(-20, 20)

ax.legend()
plt.tight_layout()
plt.show()
