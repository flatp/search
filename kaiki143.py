import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

plt.rcParams["font.family"] = "MS Gothic"

# ==============================
# 1. データ読み込み
# ==============================
df = pd.read_csv("alllist-past2.csv")

# ==============================
# 2. 特徴量作成
# ==============================
df["100-201902知名度"] = 100 - df["201902知名度"]
df["log(100-201902知名度)"] = np.log10(df["100-201902知名度"] + 1)
df["log(100-201902知名度)^2"] = df["log(100-201902知名度)"] ** 2
df["log(100-201902知名度)*weightedPV1_past"] = (
    df["log(100-201902知名度)"] * df["weightedPV1_past"]
)
df["(100-201902知名度)*weightedPV1_past"] = (
    df["100-201902知名度"] * df["weightedPV1_past"]
)

ex = [
    "100-201902知名度",
    "log(100-201902知名度)*weightedPV1_past",
    "(100-201902知名度)*weightedPV1_past"
]

df["y"] = df["aware_diff_past"]

# ==============================
# 3. 知名度レンジ & 係数設定
# ==============================
bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

# ★ レンジごとに a,b,c を変える
abc_dict = {
    (0, 20):  {"a": -1.2, "b": -0.0010, "c": 0.00015},
    (20, 40): {"a": -1.0, "b": -0.0008, "c": 0.00010},
    (40, 60): {"a": -0.8, "b": -0.0006, "c": 0.00008},
    (60, 80): {"a": -0.6, "b": -0.0004, "c": 0.00005},
    (80, 100):{"a": -0.4, "b": -0.0002, "c": 0.00003},
}

results = []

# ==============================
# 4. レンジ別に y' を計算・評価
# ==============================
for low, high in bins:
    print("\n" + "=" * 60)
    print(f"201902知名度レンジ: {low}–{high}")
    print("=" * 60)

    df_bin = df[
        (df["201902知名度"] >= low) &
        (df["201902知名度"] < high)
    ].copy()

    n = len(df_bin)
    print("データ数:", n)

    if n < 10:
        print("データ不足のためスキップ")
        continue

    a = abc_dict[(low, high)]["a"]
    b = abc_dict[(low, high)]["b"]
    c = abc_dict[(low, high)]["c"]

    # ---------- y' ----------
    df_bin["y_hat"] = (
        a * df_bin[ex[0]] +
        b * df_bin[ex[1]] +
        c * df_bin[ex[2]]
    ) * np.log10(101 - df_bin[ex[0]]) / (df_bin[ex[0]] + 101) - 2

    # ---------- 評価 ----------
    corr = df_bin["y"].corr(df_bin["y_hat"])
    r2 = r2_score(df_bin["y"], df_bin["y_hat"])

    print(f"a={a}, b={b}, c={c}")
    print("相関係数:", corr)
    print("R^2:", r2)

    results.append({
        "range": f"{low}-{high}",
        "n": n,
        "a": a,
        "b": b,
        "c": c,
        "corr": corr,
        "r2": r2
    })

    # ---------- プロット ----------
    plt.figure(figsize=(6, 6))
    plt.scatter(df_bin["y"], df_bin["y_hat"], alpha=0.6)

    min_val = min(df_bin["y"].min(), df_bin["y_hat"].min())
    max_val = max(df_bin["y"].max(), df_bin["y_hat"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "--", linewidth=2)

    formula_text = (
        f"y' = {a}×{ex[0]} + {b}×{ex[1]} + {c}×{ex[2]}"
        f"\nCorr = {corr:.3f}\nR² = {r2:.3f}"
    )

    plt.text(
        0.05, 0.95,
        formula_text,
        transform=plt.gca().transAxes,
        fontsize=6,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Actual y")
    plt.ylabel("Predicted y'")
    plt.title(f"y vs y' ({low}-{high})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"model_abc_{low}_{high}.png")
    plt.show()

# ==============================
# 5. サマリー
# ==============================
summary = pd.DataFrame(results)
print("\n========== SUMMARY ==========")
print(summary)
