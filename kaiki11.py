# =====================================
# 0. Libraries
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.tree import DecisionTreeRegressor, plot_tree

plt.rcParams["font.family"] = "MS Gothic"

# =====================================
# 1. Load Data
# =====================================
df = pd.read_csv("alllist-past.csv")

ex = ["202208知名度", "pub_score1", "weightedPV1", "browserank"]

X_raw = df[ex].copy()
y = df["aware_diff"] / np.log10(df["202208知名度"])

# =====================================
# (NEW) 1/x 特徴量を追加
# =====================================
eps = 1e-6  # ゼロ割り防止のための微小値

X = X_raw.copy()
# for col in ex:
#     X[f"{col}_inv"] = 1 / (X_raw[col] + eps) 

# print("追加済み特徴量：", X.columns.tolist())

# =====================================
# 2. Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 3. Linear Regression（基礎モデル）
# =====================================
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

print("========== Linear Regression ==========")
print("Coefficients:", lin_reg.coef_)
print("Intercept:", lin_reg.intercept_)
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# =====================================
# 4. Polynomial Regression (Degree 2) + 1/x
# =====================================
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)
y_poly_pred = poly_reg.predict(X_poly_test)

poly_feature_names = poly.get_feature_names_out(X.columns)

print("\n========== Polynomial Regression (degree=2 + 1/x) ==========")
print("R2:", r2_score(y_test, y_poly_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_poly_pred)))
print("Intercept:", poly_reg.intercept_)

# 係数出力
poly_coef = pd.Series(poly_reg.coef_, index=poly_feature_names)
print("Coefficients:")
print(poly_coef)

# =====================================
# 5. Lasso for Feature Selection (on polynomial terms)
# =====================================
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_poly_train, y_train)
y_lasso_pred = lasso.predict(X_poly_test)

lasso_coef = pd.Series(lasso.coef_, index=poly_feature_names)

print("\n========== Lasso Regression (Poly + 1/x) ==========")
print("R2:", r2_score(y_test, y_lasso_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_lasso_pred)))
print("Intercept:", lasso.intercept_)
print("Coefficients:")
print(lasso_coef[lasso_coef != 0])


plt.figure(figsize=(12, 5))

# Linear
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linear Regression")
plt.xlabel("Actual aware_diff/202208知名度")
plt.ylabel("Predicted")

# Polynomial
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_poly_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Polynomial Regression (deg=2)")
plt.xlabel("Actual aware_diff/202208知名度")
plt.ylabel("Predicted")

# Lasso
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_lasso_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Lasso Regression (on Poly features)")
plt.xlabel("Actual aware_diff/202208知名度")
plt.ylabel("Predicted")

plt.tight_layout()
plt.show()

