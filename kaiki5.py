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

ex = ["202208知名度", "weightedPV5"]

X = df[ex]
y = df["aware_diff"]

# =====================================
# 2. Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 4. Polynomial Regression (Degree 2)
# =====================================
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)
y_poly_pred = poly_reg.predict(X_poly_test)

poly_feature_names = poly.get_feature_names_out(X.columns)

print("\n========== Polynomial Regression (degree=3) ==========")
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
plt.figure(figsize=(8, 5))
plt.title(ex)

# Polynomial
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_poly_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Polynomial Regression (deg=3)")
plt.xlabel("Actual aware_diff")
plt.ylabel("Predicted")

# Lasso
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_lasso_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Lasso Regression (on Poly features)")
plt.xlabel("Actual aware_diff")
plt.ylabel("Predicted")

plt.tight_layout()
plt.show()

