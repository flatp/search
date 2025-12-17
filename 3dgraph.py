import numpy as np
import matplotlib.pyplot as plt

# =====================
# パラメータ（例：小さめ推奨）
# =====================
a = -0.6
b = -0.0008
c = 0.00006

# =====================
# x, y の範囲
# =====================
x = np.linspace(0, 99.9, 100)        # log(0)回避
y = np.linspace(0, 500000, 100)

X, Y = np.meshgrid(x, y)

# =====================
# z の計算
# =====================
Z = a * (100 - X) + b * np.log10(100 - X) * Y + c * (100 - X) * Y

# =====================
# 3Dプロット
# =====================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

ax.set_xlabel("x (0–100)")
ax.set_ylabel("y (0–500,000)")
ax.set_zlabel("z")
ax.set_title("z = a(100-x) + b log(100-x) y + c(100-x) y")

fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()
