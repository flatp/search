import pandas as pd
import numpy as np

# ==== パラメータ ====
base_date = pd.Timestamp("2023-02-01")  # 基準日
output_file = "weighted_pageviews5.csv"

# ==== データ読み込み ====
pageviews = pd.read_csv("pageviews_talents2.csv")

# ==== wide形式 → long形式 ====
pageviews_long = pageviews.melt(id_vars="article", var_name="date", value_name="views")
pageviews_long["date"] = pd.to_datetime(pageviews_long["date"])
pageviews_long["views"] = pd.to_numeric(pageviews_long["views"], errors="coerce")

# ==== 基準日との差（日数）====
pageviews_long["days_from_base"] = (base_date - pageviews_long["date"]).dt.days
pageviews_long = pageviews_long[pageviews_long["days_from_base"] >= 0]  # 未来データは除外

# ==== 忘却曲線による重み ====
lambda_ = 0.4
pageviews_long["weight"] = np.exp(-lambda_ * pageviews_long["days_from_base"]) + pageviews_long["days_from_base"] ** -0.3
pageviews_long["weighted_views"] = pageviews_long["views"] * pageviews_long["weight"]

# ==== タレントごとに合計 ====
weighted_sum = (
    pageviews_long.groupby("article")["weighted_views"]
    .sum()
    .reset_index()
    .rename(columns={"article": "name", "weighted_views": "weighted_pv"})
)

# ==== 出力 ====
weighted_sum.to_csv(output_file, index=False)
print(f"✅ 出力完了: {output_file}")
print(weighted_sum.head())
