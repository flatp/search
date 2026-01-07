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

# ★ レンジ別 a,b,c（仮）
abc_dict = {
    (0, 20):  {"a": -12, "b": -0.0010, "c": 0.00015},
    (20, 40): {"a": -10, "b": -0.0008, "c": 0.00010},
    (40, 60): {"a": -8,  "b": -0.0006, "c": 0.00008},
    (60, 80): {"a": -6,  "b": -0.0004, "c": 0.00005},
    (80, 100):{"a": -4,  "b": -0.0002, "c": 0.00003},
}

# =====================
# 3. レンジ別 3D 可視化
# =====================
for low, high in bins:
    print("\n" + "=" * 60)
    print(f"201902知名度レンジ: {low}–{high}")
    print("=" * 60)

    a = abc_dict[(low, high)]["a"]
    b = abc_dict[(low, high)]["b"]
    c = abc_dict[(low, high)]["c"]

    # ---------- 対応する実測点 ----------
    mask = (df["201902知名度"] >= low) & (df["201902知名度"] < high)
    x_obs = x_data[mask]
    y_obs = y_data[mask]
    z_obs = z_data[mask]

    print(f"a={a}, b={b}, c={c}, データ数={len(x_obs)}")

    # =====================
    # 理論曲面用グリッド（そのレンジのみ）
    # =====================
    x = np.linspace(low, min(high, 99.9), 80)
    y = np.linspace(0, 2_000_000, 80)

    X, Y = np.meshgrid(x, y)

    Z = (
        a * (100 - X)
        + b * np.log10(101 - X) * Y
        + c * (100 - X) * Y
    ) * np.log10(X + 1) / (201 - X) - 2

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
        x_obs,
        y_obs,
        z_obs,
        color="red",
        s=18,
        alpha=0.8,
        label="Observed data"
    )

    # =====================
    # ラベル・設定
    # =====================
    ax.set_xlabel("201902知名度 (x)")
    ax.set_ylabel("weightedPV1_past (y)")
    ax.set_zlabel("aware_diff_past (z)")

    ax.set_title(
        f"Theoretical surface & data\n"
        f"x ∈ [{low}, {high}),  a={a}, b={b}, c={c}"
    )

    ax.set_zlim(-20, 20)
    ax.legend()

    plt.tight_layout()
    plt.show()
