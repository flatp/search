import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

plt.rcParams['font.family'] = 'MS Gothic'

# ==== ファイル読み込み ====
pop_all = pd.read_csv('pop_all.csv')
pageviews = pd.read_csv('pageviews_talents.csv')
pop_news = pd.read_csv('news_all_score_true.csv')

# ==== 日付整形 ====
# pageviews: 'article' 列がタレント名, 残りの列が 'YYYY-MM-DD' 形式
pageviews_long = pageviews.melt(id_vars='article', var_name='date', value_name='views')
pageviews_long['date'] = pd.to_datetime(pageviews_long['date'])

# pop_news: pubDate を datetime 形式に
pop_news['pubDate'] = pd.to_datetime(pop_news['date'], utc=True)
pop_news['pubDate_date'] = pop_news['pubDate'].dt.tz_convert(None).dt.date

# 2023年1月末までに絞る
end_date = pd.to_datetime('2023-01-31').date()
pop_news = pop_news[pop_news['pubDate_date'] <= end_date]

# pageviews: 日付列を long 形式に変換済み
pageviews_long['date'] = pd.to_datetime(pageviews_long['date'])

# 2023年1月末までに絞る
pageviews_long = pageviews_long[pageviews_long['date'].dt.date <= end_date]

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


# ==== 感情スコア集計 ====
# バーストごとに感情スコア平均を取る（複数ニュースがあった場合）
grouped = merged.groupby('name_x').agg(
    burst_count=('date_x', 'nunique'),
    news_count=('title', 'count') 
).reset_index()

burst_views_sum = burst_df.groupby('name')['views'].sum().reset_index(name='burst_views_sum')
grouped2 = pd.merge(grouped, burst_views_sum, how='left', left_on='name_x', right_on='name')

grouped_keys = set(grouped2['name_x'])

# pop_all のキー（name）リスト
pop_keys = set(pop_all['name'])

# pop_all にだけあるキー
only_in_pop = pop_keys - grouped_keys

# grouped2_dropped のカラムリスト（キー以外）
cols_to_fill = [col for col in grouped2.columns if col != 'name_x']

# 0埋め行の DataFrame を作成
df_fill = pd.DataFrame({
    'name_x': list(only_in_pop),
})

# 各カラムに0を入れる
for col in cols_to_fill:
    df_fill[col] = 0

# grouped2_dropped と 0埋め行を縦に連結
grouped2 = pd.concat([grouped2, df_fill], ignore_index=True)

# バーストの日のニュース score 合計
burst_score_sum = merged.groupby('name_x')['score'].sum().reset_index(name='burst_score_sum')

# 全ニュースの score 合計（pop_news 側で name を name_map で変換済み）
all_score_sum = pop_news.groupby('name_mapped')['score'].sum().reset_index(name='all_score_sum')
all_score_sum.rename(columns={'name_mapped': 'name_x'}, inplace=True)

# grouped2 にマージ
grouped2 = pd.merge(grouped2, burst_score_sum, how='left', on='name_x')
grouped2 = pd.merge(grouped2, all_score_sum, how='left', on='name_x')

# NaN を 0 に変換（スコアが無い場合）
grouped2[['burst_score_sum', 'all_score_sum']] = grouped2[['burst_score_sum', 'all_score_sum']].fillna(0)

# --- バースト日のニュース ---
burst_pos_count = merged[merged['score'] > 0].groupby('name_x')['score'].count().reset_index(name='burst_pos_count')
burst_neg_count = merged[merged['score'] < 0].groupby('name_x')['score'].count().reset_index(name='burst_neg_count')
burst_nonzero_count = merged[merged['score'] != 0].groupby('name_x')['score'] \
    .count().reset_index(name='burst_nonzero_count')

# --- 全ニュース ---
all_pos_count = pop_news[pop_news['score'] > 0].groupby('name_mapped')['score'].count().reset_index(name='all_pos_count')
all_neg_count = pop_news[pop_news['score'] < 0].groupby('name_mapped')['score'].count().reset_index(name='all_neg_count')
all_nonzero_count = pop_news[pop_news['score'] != 0].groupby('name_mapped')['score'] \
    .count().reset_index(name='all_nonzero_count')

# name_mapped → name_x に統一
all_pos_count.rename(columns={'name_mapped': 'name_x'}, inplace=True)
all_neg_count.rename(columns={'name_mapped': 'name_x'}, inplace=True)
all_nonzero_count.rename(columns={'name_mapped': 'name_x'}, inplace=True)

# ==== grouped2 にマージ ====
grouped2 = pd.merge(grouped2, burst_pos_count, how='left', on='name_x')
grouped2 = pd.merge(grouped2, burst_neg_count, how='left', on='name_x')
grouped2 = pd.merge(grouped2, all_pos_count, how='left', on='name_x')
grouped2 = pd.merge(grouped2, all_neg_count, how='left', on='name_x')
grouped2 = pd.merge(grouped2, burst_nonzero_count, how='left', on='name_x')
grouped2 = pd.merge(grouped2, all_nonzero_count, how='left', on='name_x')

# NaN → 0
grouped2[['burst_pos_count', 'burst_neg_count', 'all_pos_count', 'all_neg_count']] = \
    grouped2[['burst_pos_count', 'burst_neg_count', 'all_pos_count', 'all_neg_count']].fillna(0)
grouped2[['burst_nonzero_count', 'all_nonzero_count']] = \
    grouped2[['burst_nonzero_count', 'all_nonzero_count']].fillna(0)

all_news_count = pop_news.groupby('name_mapped')['title'].count().reset_index(name='all_news_count')
all_news_count.rename(columns={'name_mapped': 'name_x'}, inplace=True)
grouped2 = pd.merge(grouped2, all_news_count, how='left', on='name_x')
grouped2[['all_news_count']] = grouped2[['all_news_count']].fillna(0)

grouped2_dropped = grouped2.drop(columns=['name'])

# ==== 人気度・知名度変化とのマージ ====
final_df = pd.merge(grouped2_dropped, pop_all, left_on='name_x', right_on='name', how='left')
final_df = final_df[['name', 'burst_count', 'burst_views_sum', 'news_count', 'all_news_count', 'pop_diff', 'aware_diff', '202208人気度', '202208知名度', '202302人気度', '202302知名度', 'burst_score_sum', 'all_score_sum', 'burst_pos_count', 'burst_neg_count', 'all_pos_count', 'all_neg_count', 'burst_nonzero_count', 'all_nonzero_count']]

# ==== 結果をCSV出力 ====
final_df.to_csv('burst_news_sentiment_pop_change7.csv', index=False)


