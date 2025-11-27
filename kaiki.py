# =============================
# 0. ライブラリ
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

plt.rcParams["font.family"] = "MS Gothic"

# =============================
# 1. データ読み込み
# =============================
df = pd.read_csv("alllist5.csv")

X = df[["202208知名度", "pub_score2", "weightedPV5"]]
y = df["aware_diff"]

# =============================
# 2. 学習データとテストデータに分割
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 3. シンプルな線形回帰モデル
# =============================
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

print("==== Linear Regression ====")
print("係数:", lin_reg.coef_)
print("切片:", lin_reg.intercept_)
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# =============================
# 4. 非線形性も考慮した2次多項式モデル
# =============================
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)

y_poly_pred = poly_reg.predict(X_poly_test)

print("\n==== Polynomial Regression (degree=2) ====")
print("R2:", r2_score(y_test, y_poly_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_poly_pred)))

# =============================
# 5. 可視化：予測 vs 実測（線形＆多項式）
# =============================
plt.figure(figsize=(10, 5))

# --- 線形 ---
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linear Regression\nPredicted vs Actual")
plt.xlabel("Actual aware_diff")
plt.ylabel("Predicted aware_diff")

# --- 多項式 ---
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_poly_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Polynomial Regression (Degree=2)\nPredicted vs Actual")
plt.xlabel("Actual aware_diff")
plt.ylabel("Predicted aware_diff")

plt.tight_layout()
plt.show()

# =============================
# 6. 各特徴量と目的変数の相関を軽く可視化
# =============================
plt.figure(figsize=(12,4))
for i, col in enumerate(X.columns):
    plt.subplot(1, 3, i+1)
    plt.scatter(df[col], y, alpha=0.5)
    plt.title(f"{col} vs aware_diff")
    plt.xlabel(col)
    plt.ylabel("aware_diff")

plt.tight_layout()
plt.show()
