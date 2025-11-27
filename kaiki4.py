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
df = pd.read_csv("alllist5.csv")

ex = ["202208知名度", "pub_score2", "weightedPV5", "browserank"]

X = df[ex]
y = df["aware_diff"]

# =====================================
# 2. Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 3. Linear Regression
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
# 4. Polynomial Regression (Degree 2)
# =====================================
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)
y_poly_pred = poly_reg.predict(X_poly_test)

poly_feature_names = poly.get_feature_names_out(X.columns)

print("\n========== Polynomial Regression (degree=2) ==========")
print("R2:", r2_score(y_test, y_poly_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_poly_pred)))
print("Intercept:", poly_reg.intercept_)

# 各特徴量の係数を表示
poly_coef = pd.Series(poly_reg.coef_, index=poly_feature_names)
print("Coefficients:")
print(poly_coef)

# =====================================
# 5. Lasso for Feature Selection (on polynomial terms)
# =====================================
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_poly_train, y_train)
y_lasso_pred = lasso.predict(X_poly_test)

poly_feature_names = poly.get_feature_names_out(X.columns)

print("\n========== Lasso Regression (on Poly features) ==========")
lasso_coef = pd.Series(lasso.coef_, index=poly_feature_names)
print("R2:", r2_score(y_test, y_lasso_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_lasso_pred)))
print("Intercept:", lasso.intercept_)
print("Coefficients:")
print(lasso_coef[lasso_coef != 0])

# =====================================
# 6. Partial Dependence (news × PV interaction)
# =====================================
features = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # 各タプルは (列番号1, 列番号2)

fig, ax = plt.subplots(2, 3, figsize=(18, 8))
PartialDependenceDisplay.from_estimator(
    poly_reg, X_poly_train, features=features,
    feature_names=X.columns, ax=ax
)

# タイトルを個別に設定
ax[0][0].set_title("Partial Dependence: 202208知名度 × pub_score2")
ax[0][1].set_title("Partial Dependence: 202208知名度 × weightedPV5")
ax[0][2].set_title("Partial Dependence: 202208知名度 × browserank")
ax[1][0].set_title("Partial Dependence: pub_score2 × weightedPV5")
ax[1][1].set_title("Partial Dependence: pub_score2 × browserank")
ax[1][2].set_title("Partial Dependence: weightedPV5 × browserank")

plt.tight_layout()
plt.show()

# =====================================
# 7. Small Decision Tree for Human-readable Rules
# =====================================
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns, filled=True, fontsize=8)
plt.title("Decision Tree Rules (max_depth=3)")
plt.show()

# =====================================
# 8. Visualization: Predicted vs Actual
# =====================================
plt.figure(figsize=(12, 5))
plt.title(ex)

# Linear
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linear Regression")
plt.xlabel("Actual aware_diff")
plt.ylabel("Predicted")

# Polynomial
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_poly_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Polynomial Regression (deg=2)")
plt.xlabel("Actual aware_diff")
plt.ylabel("Predicted")

# Lasso
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_lasso_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Lasso Regression (on Poly features)")
plt.xlabel("Actual aware_diff")
plt.ylabel("Predicted")

plt.tight_layout()
plt.show()

# =====================================
# 9. Scatter plots for each variable vs target
# =====================================
plt.figure(figsize=(14, 4))
for i, col in enumerate(X.columns):
    plt.subplot(1, 4, i+1)
    plt.scatter(df[col], y, alpha=0.5)
    plt.title(f"{col} vs aware_diff")
    plt.xlabel(col)
    plt.ylabel("aware_diff")
plt.tight_layout()
plt.show()
