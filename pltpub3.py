import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

# ==== データ読み込み ====
df = pd.read_csv("alllist3.csv")

# ==== 対象列 ====
y_cols = ["pop_diff", "aware_diff", "202302人気度", "202302知名度"]
x_cols = ["weightedPV1", "weightedPV2", "weightedPV3", "weightedPV4", "weightedPV5"]

# ==== スタイル設定 ====
plt.rcParams["font.family"] = "MS Gothic"
sns.set(style="whitegrid", font="MS Gothic")

# ==== 図の準備 ====
fig, axes = plt.subplots(len(y_cols), len(x_cols), figsize=(20, 18))
fig.suptitle("スコアと人気・知名度指標の相関・回帰分析", fontsize=18, y=1.02)

for i, y in enumerate(y_cols):
    for j, x in enumerate(x_cols):
        ax = axes[i, j]
        sub = df[[x, y]].dropna()

        # 回帰分析
        if len(sub) > 1:
            X = sm.add_constant(sub[x])  # 定数項を追加
            model = sm.OLS(sub[y], X).fit()
            r = np.corrcoef(sub[x], sub[y])[0, 1]
            r2 = model.rsquared
            pval = model.pvalues[x]
        else:
            r, r2, pval = np.nan, np.nan, np.nan

        # 散布図 + 回帰線
        sns.regplot(data=sub, x=x, y=y, ax=ax,
                    scatter_kws={'alpha':0.6},
                    line_kws={'color':'red'})

        # 指標表示
        ax.text(0.05, 0.95,
                f"r = {r:.2f}\nR² = {r2:.2f}\np = {pval:.3g}",
                transform=ax.transAxes,
                fontsize=10, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        # 軸ラベル整形
        if i == len(y_cols) - 1:
            ax.set_xlabel(x, fontsize=11)
        else:
            ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel(y, fontsize=11)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("correlation_weightedPV_regression.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ 出力完了: correlation_weightedPV_regression2.png に保存されました。")
