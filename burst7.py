import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

plt.rcParams['font.family'] = 'MS Gothic'

# ==== データ読み込み ====
pageviews = pd.read_csv("pageviews_talents.csv")

# ==== ロング形式に変換 ====
pageviews_long = pageviews.melt(id_vars="article", var_name="date", value_name="views")
pageviews_long["date"] = pd.to_datetime(pageviews_long["date"])
pageviews_long["views"] = pd.to_numeric(pageviews_long["views"], errors="coerce")

# ==== 関数: NumPy版移動平均 ====
def moving_average(x, w=7):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.array([])
    return np.convolve(x, np.ones(w), "same") / w

# ==== バースト判定 ====
names = pageviews["article"].unique()
sample_names = random.sample(list(names), min(10, len(names)))

# ==== 可視化 ====
plt.rcParams["font.family"] = "MS Gothic"
fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()

for i, name in enumerate(sample_names):
    ax = axes[i]
    df = pageviews_long[pageviews_long["article"] == name].sort_values("date").copy()

    views = df["views"].to_numpy(dtype=float)
    ma7 = moving_average(views, w=7)
    burst_mask = views > ma7 * 2  # バースト定義

    # 折れ線（PV）
    ax.plot(df["date"].to_numpy(), views, color="gray", linewidth=1.0, label="PV")
    # 移動平均線（青）
    ax.plot(df["date"].to_numpy(), ma7, color="blue", linewidth=0.5, label="7日移動平均", alpha=0.5)
    # バースト部分（赤点）
    ax.plot(df["date"].to_numpy()[burst_mask],
            views[burst_mask],
            "o", color="red", markersize=4, label="バースト")

    ax.set_title(name, fontsize=10)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, alpha=0.3)

# 凡例を共通に1つだけ表示
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10)

# 余ったサブプロットを非表示
for k in range(i + 1, len(axes)):
    axes[k].axis("off")

fig.suptitle("PV数の推移とバースト検出（ランダム10人）", fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("burst_visualization_10_with_MA.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ 出力完了: burst_visualization_10_with_MA.png に保存されました。")