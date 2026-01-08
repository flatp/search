import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

plt.rcParams["font.family"] = "MS Gothic"

# ==============================
# 1. データ読み込み
# ==============================
df = pd.read_csv("alllist-past22.csv")

# ==============================
# 2. 特徴量作成
# ==============================
df["100-202208知名度"] = 100 - df["202208知名度"]
df["log(100-202208知名度)"] = np.log10(df["100-202208知名度"] + 1)
df["log(100-202208知名度)^2"] = df["log(100-202208知名度)"] ** 2
df["log(100-202208知名度)*weightedPV1"] = df["log(100-202208知名度)"] * df["weightedPV1"]
df["(100-202208知名度)*weightedPV1"] = df["100-202208知名度"] * df["weightedPV1"]
df["(100-202208知名度)*allPV"] = df["100-202208知名度"] * df["allPV"]
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
# 3. gender 別 係数設定
# ==============================
gender_list = ["男性", "女性"]

abc_dict = {
    "男性": {"a": -3.5, "b": -0.0008, "c": 0.00008, "d": -1.5},
    "女性": {"a": -25, "b": -0.001, "c": 0.00009, "d": 2.4},
}

results = []

# ==============================
# 4. gender 別に y' を計算・評価
# ==============================
for gender in gender_list:
    print("\n" + "=" * 60)
    print(f"gender: {gender}")
    print("=" * 60)

    df_bin = df[df["gender"] == gender].copy()

    n = len(df_bin)
    print("データ数:", n)

    if n < 10:
        print("データ不足のためスキップ")
        continue

    a = abc_dict[gender]["a"]
    b = abc_dict[gender]["b"]
    c = abc_dict[gender]["c"]
    d = abc_dict[gender]["d"]

    # ---------- y' ----------
    df_bin["y_hat"] = (
        a * df_bin[ex[0]] +
        b * df_bin[ex[1]] +
        c * df_bin[ex[2]]
    ) * np.log10(101 - df_bin[ex[0]]) / (df_bin[ex[0]] + 101) + d

    # ---------- 評価 ----------
    corr = df_bin["y"].corr(df_bin["y_hat"])
    r2 = r2_score(df_bin["y"], df_bin["y_hat"])

    print(f"a={a}, b={b}, c={c}, d={d}")
    print("相関係数:", corr)
    print("R^2:", r2)

    results.append({
        "gender": gender,
        "n": n,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
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
        f"gender = {gender}\n"
        f"y' = {a}×{ex[0]} + {b}×{ex[1]} + {c}×{ex[2]} + {d}"
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
    plt.title(f"y vs y' ({gender})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"model_gender_{gender}_2019.png")
    plt.show()

# ==============================
# 5. サマリー
# ==============================
summary = pd.DataFrame(results)
print("\n========== SUMMARY ==========")
print(summary)
