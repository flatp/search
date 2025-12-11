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

df["100-202208知名度"] = 100 - df["202208知名度"]
df["log(100-202208知名度)"] = np.log10(df["100-202208知名度"]+1)
df["log(100-202208知名度)^2"] = df["log(100-202208知名度)"] ** 2
df["log(100-202208知名度)*weightedPV1"] = df["log(100-202208知名度)"] * df["weightedPV1"]
df["(100-202208知名度)*weightedPV1"] = df["100-202208知名度"] * df["weightedPV1"]
df["(100-202208知名度)*allPV"] = df["100-202208知名度"] * df["allPV"]

ex = ["100-202208知名度", "log(100-202208知名度)*weightedPV1", "(100-202208知名度)*weightedPV1"]

X_raw = df[ex].copy()
y = df["aware_diff"] * (101 - df["202208知名度"]) / np.log10(df["202208知名度"]+1)
# y = df["aware_diff"] * 50 / df["202208知名度"]

# =====================================
# (NEW) 1/x 特徴量を追加
# =====================================
eps = 1e-6  # ゼロ割り防止のための微小値

X = X_raw.copy()
# X["y-past"] = df["aware_diff_past"] / np.log10(df["201902知名度"]+1)

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
# 5. Lasso for Feature Selection (on polynomial terms)
# =====================================
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_train, y_train)
y_lasso_pred = lasso.predict(X_test)

lasso_coef = pd.Series(lasso.coef_, index=X_train.columns)

print("\n========== Lasso Regression ==========")
print("R2:", r2_score(y_test, y_lasso_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_lasso_pred)))
print("Intercept:", lasso.intercept_)
print("Coefficients:")
print(lasso_coef[lasso_coef != 0])


plt.figure(figsize=(12, 5))

# Linear
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linear Regression")
plt.xlabel("Actual aware_diff/202208知名度")
plt.ylabel("Predicted")

# Lasso
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_lasso_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Lasso Regression")
plt.xlabel("Actual aware_diff/202208知名度")
plt.ylabel("Predicted")

plt.tight_layout()
plt.show()

