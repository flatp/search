import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CSV読み込み
df = pd.read_csv('pop_datalist.csv')

# 軸データ
x = df.iloc[:, 1]  # B列
y = df.iloc[:, 2]  # C列
z_columns = df.columns[3:6]  # D〜H列

# グラフの準備
fig = plt.figure(figsize=(20, 4))  # 横長の1行5列表示

for i, z_col in enumerate(z_columns):
    ax = fig.add_subplot(1, 5, i + 1, projection='3d')
    ax.scatter(x, y, df[z_col])
    ax.set_title(f'Z: {z_col}')
    ax.set_xlabel('人気度')
    ax.set_ylabel('知名度')
    ax.set_zlabel(z_col)

plt.tight_layout()
plt.show()