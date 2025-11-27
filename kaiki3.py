import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

import xgboost as xgb
import shap
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "MS Gothic"

# =============================================
# 1. 読み込み & 前処理
# =============================================
df = pd.read_csv("alllist5.csv")

target = "aware_diff"
y = df[target]

# 説明変数は aware_diff を除くすべて
X = df.drop(columns=[target, "pop_diff", "202302知名度", "202302人気度", "name"])

# 数値のみ残す
X = X.select_dtypes(include=[np.number]).fillna(0)

# =============================================
# 2. RandomForest で粗スクリーニング
# =============================================
rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(30)

print("===== RandomForest 上位特徴量 =====")
print(top_features)

# 上位30特徴のみで再定義
X_top = X[top_features.index]

# =============================================
# 3. LassoCV で絞り込み
# =============================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_top)

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)

lasso_coef = pd.Series(lasso.coef_, index=X_top.columns)
lasso_nonzero = lasso_coef[lasso_coef != 0]

print("===== Lasso 非ゼロ係数 =====")
print(lasso_nonzero.sort_values(ascending=False))

X_lasso = X_top[lasso_nonzero.index]

# =============================================
# 4. XGBoost + SHAP 解析
# =============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_lasso, y, test_size=0.2, random_state=42
)

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
xgb_model.fit(X_train, y_train)

pred_xgb = xgb_model.predict(X_test)
print("===== XGBoost R2 =====", r2_score(y_test, pred_xgb))
print("===== XGBoost RMSE =====", np.sqrt(mean_squared_error(y_test, pred_xgb)))

# SHAP
explainer = shap.TreeExplainer(xgb_model, feature_perturbation='auto')
shap_values = explainer(X_lasso)

plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_lasso)
plt.show()

plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_lasso, plot_type="bar")
plt.show()

# =============================================
# 5. 最終：シンプルモデル（線形・多項式）
# =============================================
pf = PolynomialFeatures(degree=2, include_bias=False)
X_poly = pf.fit_transform(X_lasso)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train2, y_train2)

pred_poly = lr.predict(X_test2)

print("===== Polynomial (degree=2) R2 =====", r2_score(y_test2, pred_poly))
print("===== Polynomial (degree=2) RMSE =====", np.sqrt(mean_squared_error(y_test2, pred_poly)))

# 散布図
plt.figure(figsize=(6,6))
plt.scatter(y_test2, pred_poly, alpha=0.6)
plt.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], "r--")
plt.xlabel("Actual aware_diff")
plt.ylabel("Predicted aware_diff")
plt.title("Polynomial Regression (degree=2)")
plt.show()
