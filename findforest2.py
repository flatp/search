import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

plt.rcParams['font.family'] = 'MS Gothic'

# === データ読み込み ===
df = pd.read_csv('news_all.csv')

def forest(features, target):
    df_clean = df[features + [target]].dropna()

    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Huber': HuberRegressor(),
        'SVR': SVR(),
        'RandomForest': RandomForestRegressor(random_state=0),
        'GradientBoosting': GradientBoostingRegressor(random_state=0)
    }

    X = df_clean[features]
    y = df_clean[target]

    results = {}
    
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
        
        r2 = cross_val_score(pipe, X, y, cv=5, scoring='r2').mean()
        results[model_name] = r2

    plt.figure(figsize=(10, 5))
    sorted_scores = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    plt.barh(list(sorted_scores.keys()), list(sorted_scores.values()))
    plt.xlabel("R²スコア")
    plt.title(f"目的変数: {target} のモデル別予測精度")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

forest(['ニュース数', 'バースト数', 'バースト元ニュース数', '202208人気度'], '人気度変化')
forest(['ニュース数', 'バースト数', 'バースト元ニュース数', '202208知名度'], '知名度変化')
