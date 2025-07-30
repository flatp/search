import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# モデル
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

plt.rcParams['font.family'] = 'MS Gothic'

# === データ読み込み ===
df = pd.read_csv('news_all.csv')

# === 特徴量と目的変数の選定 ===
features = ['ニュース数', 'バースト数', 'バースト元ニュース数']
X = df[features]

# 数値型のうち特徴量でないものを目的変数候補とする
target_candidates = df.select_dtypes(include=[np.number]).columns.difference(features)

# 欠損値除去
df_clean = df[features + list(target_candidates)].dropna()
X_clean = df_clean[features]

# 使用するモデル一覧
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Huber': HuberRegressor(),
    'SVR': SVR(),
    'RandomForest': RandomForestRegressor(random_state=0),
    'GradientBoosting': GradientBoostingRegressor(random_state=0)
}

# スコア格納用（目的変数 × モデル）
results = {}

# 各目的変数について学習・評価
for target in target_candidates:
    y = df_clean[target]
    results[target] = {}
    
    for model_name, model in models.items():
        # スケーリングが必要なモデル（線形・SVM）とそうでないモデルを分ける
        if model_name in ['Linear', 'Ridge', 'Lasso', 'Huber', 'SVR']:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
        else:
            pipe = Pipeline([
                ('model', model)
            ])
        
        r2 = cross_val_score(pipe, X_clean, y, cv=5, scoring='r2').mean()
        results[target][model_name] = r2

# === 可視化（目的変数ごとにモデル別比較）===
for target, model_scores in results.items():
    plt.figure(figsize=(10, 5))
    sorted_scores = dict(sorted(model_scores.items(), key=lambda item: item[1], reverse=True))
    plt.barh(list(sorted_scores.keys()), list(sorted_scores.values()))
    plt.xlabel("R²スコア")
    plt.title(f"目的変数: {target} のモデル別予測精度")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

