import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

plt.rcParams['font.family'] = 'MS Gothic'

# ==== ファイル読み込み ====
pop_all = pd.read_csv('pop_all.csv')
pageviews = pd.read_csv('pageviews_talents.csv')
pop_news = pd.read_csv('pop_news_all.csv')

# ==== 日付整形 ====
# pageviews: 'article' 列がタレント名, 残りの列が 'YYYY-MM-DD' 形式
pageviews_long = pageviews.melt(id_vars='article', var_name='date', value_name='views')
pageviews_long['date'] = pd.to_datetime(pageviews_long['date'])

# pop_news: pubDate を datetime 形式に
pop_news['pubDate'] = pd.to_datetime(pop_news['date'], utc=True)
pop_news['pubDate_date'] = pop_news['pubDate'].dt.tz_convert(None).dt.date

# ==== 人気度・知名度の変化量 ====
pop_all['pop_diff'] = pop_all['202302人気度'] - pop_all['202208人気度']
pop_all['aware_diff'] = pop_all['202302知名度'] - pop_all['202208知名度']

# ==== バースト検出 ====
burst_info = []

for name in pageviews['article'].unique():
    df = pageviews_long[pageviews_long['article'] == name].sort_values('date')
    df['ma7'] = df['views'].rolling(window=7, min_periods=1).mean()
    df['burst'] = (df['views'] > df['ma7'] * 2)  # 2倍以上の急増をバーストとみなす

    bursts = df[df['burst']]
    for _, row in bursts.iterrows():
        burst_info.append({'name': name, 'date': row['date'].date(), 'views': row['views']})

burst_df = pd.DataFrame(burst_info)

# ==== バーストとニュースを結合 ====
burst_with_news = burst_df.copy()
burst_with_news['match_date'] = burst_with_news['date'].apply(lambda x: [x, x - timedelta(days=1)])
burst_with_news = burst_with_news.explode('match_date')

# pop_newsのnameとpop_allのname2を対応付け、さらにarticleとnameを結びつける
name_map = dict(zip(pop_all['name2'], pop_all['name']))  # name2 → name
pop_news['name_mapped'] = pop_news['name'].map(name_map)

# バーストに該当するニュースをマージ
merged = pd.merge(
    burst_with_news,
    pop_news,
    how='left',
    left_on=['name', 'match_date'],
    right_on=['name_mapped', 'pubDate_date']
)

merged['sentiment_score'] = merged['sentiment'].map({'positive': 1, 'negative': 0})

# ==== 感情スコア集計 ====
# バーストごとに感情スコア平均を取る（複数ニュースがあった場合）
grouped = merged.groupby('name_x').agg(
    burst_count=('date_x', 'nunique'),
    sentiment_mean=('sentiment_score', 'mean'),
    news_count=('title', 'count') 
).reset_index()

# ==== 人気度・知名度変化とのマージ ====
final_df = pd.merge(grouped, pop_all, left_on='name_x', right_on='name', how='left')
final_df = final_df[['name', 'burst_count', 'sentiment_mean', 'news_count', 'pop_diff', 'aware_diff']]

# ==== 結果をCSV出力 ====
final_df.to_csv('burst_news_sentiment_pop_change.csv', index=False)

# ==== グラフ表示 ====
plt.figure(figsize=(12, 5))

plt.subplot(2, 2, 1)
sns.regplot(data=final_df, x='burst_count', y='pop_diff')
plt.title('バースト件数 vs 人気度の変化')

plt.subplot(2, 2, 2)
sns.regplot(data=final_df, x='burst_count', y='aware_diff')
plt.title('バースト件数 vs 知名度の変化')

plt.subplot(2, 2, 3)
sns.regplot(data=final_df, x='news_count', y='pop_diff')
plt.title('ニュース件数 vs 人気度の変化')

plt.subplot(2, 2, 4)
sns.regplot(data=final_df, x='news_count', y='aware_diff')
plt.title('ニュース件数 vs 知名度の変化')

plt.tight_layout()
plt.show()

# ==== 相関係数 ====
print("相関係数（バースト件数と人気度・知名度の変化）")
print(final_df[['burst_count', 'news_count', 'pop_diff', 'aware_diff']].corr())


