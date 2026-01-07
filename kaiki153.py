# =====================================
# 0. Libraries
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams["font.family"] = "MS Gothic"

# =====================================
# 0.1 Metrics
# =====================================
def mean_absolute_percentage_error(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))

def mean_relative_error(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) / (y_true + eps))

# =====================================
# 1. Load Data
# =====================================
df = pd.read_csv("alllist-past2.csv")

# =====================================
# 2. Feature Engineering
# =====================================
# ---- 202208 ----
df["100-202208知名度"] = 100 - df["202208知名度"]
df["log(100-202208知名度)"] = np.log10(df["100-202208知名度"] + 1)
df["log(100-202208知名度)^2"] = df["log(100-202208知名度)"] ** 2
df["log(100-202208知名度)*weightedPV1"] = (
    df["log(100-202208知名度)"] * df["weightedPV1"]
)
df["(100-202208知名度)*weightedPV1"] = (
    df["100-202208知名度"] * df["weightedPV1"]
)
df["(100-202208知名度)*allPV"] = (
    df["100-202208知名度"] * df["allPV"]
)

# ---- 201902 ----
df["100-201902知名度"] = 100 - df["201902知名度"]
df["log(100-201902知名度)"] = np.log10(df["100-201902知名度"] + 1)
df["log(100-201902知名度)^2"] = df["log(100-201902知名度)"] ** 2
df["log(100-201902知名度)*weightedPV1_past"] = (
    df["log(100-201902知名度)"] * df["weightedPV1_past"]
)
df["(100-201902知名度)*weightedPV1_past"] = (
    df["100-201902知名度"] * df["weightedPV1_past"]
)

# ---- 正則化スケーリング項 ----
df["100-201902知名度::"] = (
    df["100-201902知名度"]
    * np.log10(df["201902知名度"] + 1)
    / (201 - df["201902知名度"])
)

df["log(100-201902知名度)*weightedPV1_past::"] = (
    df["log(100-201902知名度)*weightedPV1_past"]
    * np.log10(df["201902知名度"] + 1)
    / (201 - df["201902知名度"])
)

df["(100-201902知名度)*weightedPV1_past::"] = (
    df["(100-201902知名度)*weightedPV1_past"]
    * np.log10(df["201902知名度"] + 1)
    / (201 - df["201902知名度"])
)

# =====================================
# 3. Features / Target
# =====================================
ex = [
    "100-201902知名度::",
    "log(100-201902知名度)*weightedPV1_past::",
    "(100-201902知名度)*weightedPV1_past::",
]

y_col = "aware_diff_past"

# =====================================
# 4. 201902知名度レンジ別学習
# =====================================
bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
results = []

for low, high in bins:
    print("\n" + "=" * 60)
    print(f"201902知名度レンジ: {low}–{high}")
    print("=" * 60)

    df_bin = df[
        (df["201902知名度"] >= low) &
        (df["201902知名度"] < high)
    ]

    n = len(df_bin)
    print("データ数:", n)

    if n < 20:
        print("データ不足のためスキップ")
        continue

    X = df_bin[ex].copy()
    y = df_bin[y_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------- Linear ----------
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred = lin.predict(X_test)

    lin_r2   = r2_score(y_test, y_pred)
    lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    lin_mape = mean_absolute_percentage_error(y_test, y_pred)
    lin_mre  = mean_relative_error(y_test, y_pred)

    print("\n[Linear Regression]")
    print("R2   :", lin_r2)
    print("RMSE :", lin_rmse)
    print("MAPE :", lin_mape)
    print("MRE  :", lin_mre)
    print("Coef :", lin.coef_)
    print("Intercept:", lin.intercept_)

    # ---------- Lasso ----------
    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_lasso = lasso.predict(X_test)

    lasso_r2   = r2_score(y_test, y_lasso)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, y_lasso))
    lasso_mape = mean_absolute_percentage_error(y_test, y_lasso)
    lasso_mre  = mean_relative_error(y_test, y_lasso)

    coef_lasso = pd.Series(lasso.coef_, index=X.columns)

    print("\n[Lasso Regression]")
    print("R2   :", lasso_r2)
    print("RMSE :", lasso_rmse)
    print("MAPE :", lasso_mape)
    print("MRE  :", lasso_mre)
    print("Intercept:", lasso.intercept_)
    print("Non-zero Coef:")
    print(coef_lasso[coef_lasso != 0])

    # ---------- 保存 ----------
    results.append({
        "range": f"{low}-{high}",
        "n": n,
        "lin_r2": lin_r2,
        "lin_rmse": lin_rmse,
        "lin_mape": lin_mape,
        "lin_mre": lin_mre,
        "lasso_r2": lasso_r2,
        "lasso_rmse": lasso_rmse,
        "lasso_mape": lasso_mape,
        "lasso_mre": lasso_mre,
    })

    # ---------- Plot ----------
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    plt.title(f"Linear ({low}-{high})")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_lasso, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    plt.title(f"Lasso ({low}-{high})")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.tight_layout()
    plt.show()

# =====================================
# 5. Summary
# =====================================
summary = pd.DataFrame(results)
print("\n========== SUMMARY ==========")
print(summary)
