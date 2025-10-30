import pandas as pd
from datetime import timedelta

# ==== ファイル読み込み ====
news = pd.read_csv("news_all_score_true.csv")
pageviews = pd.read_csv("pageviews_talents.csv")

# ==== 日付整形 ====
news["date"] = pd.to_datetime(news["date"])
news["pub_date"] = news["date"].dt.date

# pageviews: wide→long形式に変換
pageviews_long = pageviews.melt(id_vars="article", var_name="date", value_name="views")
pageviews_long["date"] = pd.to_datetime(pageviews_long["date"])

# ==== publisher単位の基本統計 ====
publisher_summary = news.groupby("publisher").agg(
    news_count=("title", "count"),
    score_sum=("score", "sum"),
    score_mean=("score", "mean")
).reset_index()

# ==== バースト検出 ====
burst_info = []
for name in pageviews["article"].unique():
    df = pageviews_long[pageviews_long["article"] == name].sort_values("date")
    df["ma7"] = df["views"].rolling(window=7, min_periods=1).mean()
    df["burst"] = df["views"] > df["ma7"] * 2
    bursts = df[df["burst"]]
    for _, row in bursts.iterrows():
        burst_info.append({"name": name, "burst_date": row["date"].date(), "views": row["views"]})

burst_df = pd.DataFrame(burst_info)

# ==== バースト当日・前日のニュース ====
burst_with_news = burst_df.copy()
burst_with_news["match_date"] = burst_with_news["burst_date"].apply(lambda x: [x, x - timedelta(days=1)])
burst_with_news = burst_with_news.explode("match_date")

merged = pd.merge(
    burst_with_news,
    news,
    how="left",
    left_on=["name", "match_date"],
    right_on=["name", "pub_date"]
)

# --- publisher別のニュース（重複あり） ---
burst_news = merged.groupby("publisher").agg(
    burst_news_count=("title", "count"),
    burst_score_sum=("score", "sum")
).reset_index()

# --- publisher別のPV・日数（重複なし） ---
# → 同一publisher・同一name・同一burst_dateで一意化してから集計
unique_bursts = merged.drop_duplicates(subset=["publisher", "name", "burst_date"])
publisher_burst_pv = unique_bursts.groupby("publisher").agg(
    burst_pv_sum=("views", "sum"),
    burst_days=("burst_date", "nunique")
).reset_index()

# ==== 結合 ====
publisher_burst_summary = pd.merge(burst_news, publisher_burst_pv, how="outer", on="publisher").fillna(0)
result = pd.merge(publisher_summary, publisher_burst_summary, how="left", on="publisher").fillna(0)

# ==== CSV出力 ====
result.to_csv("publisher_burst_summary.csv", index=False)

print("✅ 出力完了: publisher_burst_summary.csv")
print(result.head())
