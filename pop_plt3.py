import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# CSV読み込み
df = pd.read_csv('pop_datalist.csv')

# B列（2列目）→ x、C列（3列目）→ y
x_all = df.iloc[:, 1]
y_all = df.iloc[:, 2]
z_columns = df.columns[3:8]  # D〜H列

# 描画用の図
fig = plt.figure(figsize=(20, 4))

for i, z_col in enumerate(z_columns):
    z_all = df[z_col]

    # --- IQRで外れ値を除去 ---
    lower = z_all.quantile(0.01)
    upper = z_all.quantile(0.99)
    mask = (z_all >= lower) & (z_all <= upper)

    # フィルタ後のデータ
    x = x_all[mask]
    y = y_all[mask]
    z = z_all[mask]

    # z = np.log(z + 1)

    # 3Dプロット
    ax = fig.add_subplot(1, 5, i + 1, projection='3d')
    ax.scatter(x, y, z)
    ax.set_title(f'Z: {z_col} (外れ値除去)')
    ax.set_xlabel(df.columns[1])
    ax.set_ylabel(df.columns[2])
    ax.set_zlabel(z_col)

plt.tight_layout()
plt.show()
