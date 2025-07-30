import pandas as pd
import matplotlib.pyplot as plt
import random
import re

plt.rcParams['font.family'] = 'MS Gothic'

# CSVファイル読み込み
df = pd.read_csv("pr_all.csv")  # ← ファイル名を適宜変更してください

# 人気度の列だけを抜き出し、差分（202302人気度 - 202208人気度）を新しい列に
df['pop_diff'] = df['202302人気度'] - df['202208人気度']

# 人気度の基準（202208人気度）でソート
df_sorted = df.sort_values('202208人気度')

# 上位・下位から10人ずつランダム抽出
low_pop = df_sorted.head(50).sample(10, random_state=42)
high_pop = df_sorted.tail(50).sample(10, random_state=42)

# 日付のカラムを取得（nameとpop列以外）
date_cols = [col for col in df.columns if re.match(r'\d{4}-\d{2}-\d{2}', col)]

def plot_group(data, title):
    plt.figure(figsize=(12, 6))
    for _, row in data.iterrows():
        plt.plot(date_cols, row[date_cols].values, label=f"{row['name']} ({row['pop_diff']:+.1f})")
    plt.title(f"{title}（人気度変化も表示）")
    plt.xlabel("日付")
    plt.ylabel("人気度")
    step = 5
    xticks_pos = list(range(0, len(date_cols), step))
    xticks_labels = [date_cols[i] for i in xticks_pos]
    plt.xticks(xticks_pos, xticks_labels, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# 低人気度のタレントのプロット
plot_group(low_pop, "元々人気度が低いタレント（202208基準）")

# 高人気度のタレントのプロット
plot_group(high_pop, "元々人気度が高いタレント（202208基準）")

# 人気度の列だけを抜き出し、差分（202302人気度 - 202208人気度）を新しい列に
df['pop_diff'] = df['202302知名度'] - df['202208知名度']

# 人気度の基準（202208人気度）でソート
df_sorted = df.sort_values('202208知名度')

# 上位・下位から10人ずつランダム抽出
low_pop = df_sorted.head(50).sample(10, random_state=42)
high_pop = df_sorted.tail(50).sample(10, random_state=42)

# 日付のカラムを取得（nameとpop列以外）
date_cols = [col for col in df.columns if re.match(r'\d{4}-\d{2}-\d{2}', col)]

def plot_group(data, title):
    plt.figure(figsize=(12, 6))
    for _, row in data.iterrows():
        plt.plot(date_cols, row[date_cols].values, label=f"{row['name']} ({row['pop_diff']:+.1f})")
    plt.title(f"{title}（知名度変化も表示）")
    plt.xlabel("日付")
    plt.ylabel("知名度")
    step = 5
    xticks_pos = list(range(0, len(date_cols), step))
    xticks_labels = [date_cols[i] for i in xticks_pos]
    plt.xticks(xticks_pos, xticks_labels, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# 低人気度のタレントのプロット
plot_group(low_pop, "元々知名度が低いタレント（202208基準）")

# 高人気度のタレントのプロット
plot_group(high_pop, "元々知名度が高いタレント（202208基準）")
