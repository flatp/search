import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 日本語フォントの設定（必要に応じて）
plt.rcParams['font.family'] = 'MS Gothic'

# データの読み込み
file_path = 'popstreamlabel2.csv'
df = pd.read_csv(file_path)

# 中央値によるグループ分け
df['group'] = df.apply(
    lambda row: 'HH' if row['202302人気度'] >= df['202302人気度'].median() and row['202302知名度'] >= df['202302知名度'].median()
    else 'LH' if row['202302人気度'] >= df['202302人気度'].median()
    else 'HL' if row['202302知名度'] >= df['202302知名度'].median()
    else 'LL', axis=1
)

# DA/BR列の抽出
da_cols = [col for col in df.columns if 'DA' in col]
br_cols = [col for col in df.columns if 'BR' in col]

# 統計特徴量の追加
df['DA_mean'] = df[da_cols].mean(axis=1)
df['DA_sum'] = df[da_cols].sum(axis=1)
df['DA_std'] = df[da_cols].std(axis=1)
df['BR_mean'] = df[br_cols].mean(axis=1)
df['BR_sum'] = df[br_cols].sum(axis=1)
df['BR_std'] = df[br_cols].std(axis=1)

# トレンド特徴量の追加
df['DA_trend'] = (df['202302DA'] - df['202208DA']) / (df['202208DA'] + 1e-6)
df['BR_trend'] = (df['202302BR'] - df['202208BR']) / (df['202208BR'] + 1e-6)

# 特徴量リスト
features = [
    'DA_mean', 'DA_sum', 'DA_std',
    'BR_mean', 'BR_sum', 'BR_std',
    'DA_trend', 'BR_trend'
]

# 特徴量とラベル
X = df[features]
y = df['group']

# 学習・テストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ランダムフォレストモデル
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 予測と評価
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
conf_matrix = confusion_matrix(y_test, y_pred)

# レポート出力
print("分類レポート:\n", report_df)

# 混同行列の可視化
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("混同行列")
plt.xlabel("予測ラベル")
plt.ylabel("実際のラベル")
plt.tight_layout()
plt.show()

