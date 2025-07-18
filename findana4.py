import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'MS Gothic'

# 相関行列のデータを定義
data = {
    'burst_count':     [1.000000, 0.590693, 0.105427, 0.072941, -0.138030, -0.060392],
    'news_count':      [0.590693, 1.000000, 0.079415, 0.172033,  0.091810,  0.093470],
    'pop_diff':        [0.105427, 0.079415, 1.000000, 0.408755,  0.038499, -0.085171],
    'aware_diff':      [0.072941, 0.172033, 0.408755, 1.000000, -0.063002, -0.081862],
    '202302人気度':      [-0.138030, 0.091810, 0.038499, -0.063002, 1.000000, 0.759909],
    '202302知名度':      [-0.060392, 0.093470, -0.085171, -0.081862, 0.759909, 1.000000],
}

# 行列のインデックスとカラムを設定
labels = ['burst_count', 'news_count', 'pop_diff', 'aware_diff', '202302人気度', '202302知名度']
df = pd.DataFrame(data, index=labels, columns=labels)

# ヒートマップの描画
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title('相関係数ヒートマップ')
plt.show()