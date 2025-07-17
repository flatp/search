import pandas as pd
import plotly.express as px

# CSVファイルの読み込み
df = pd.read_csv('pop_news_ana_all.csv')

# 対象カラムを指定
fig = px.scatter_3d(
    df,
    x='202302人気度',  # ←横軸
    y='202302知名度',  # ←縦軸
    z='ニュース数',  # ←奥行き
    color='ニュース数',  # 任意（色分けしたい変数）
    title='3次元散布図（Plotly）'
)

# HTMLファイルに保存（任意）
fig.write_html('3d_scatter2.html')

# 画面に表示（Jupyterなど）
fig.show()
