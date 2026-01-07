# =====================================
# 0. Libraries
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams["font.family"] = "MS Gothic"

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
df = pd.read_csv("alllist-past22.csv")

# =====================================
# Feature Engineering（元コードそのまま）
# =====================================
df["100-202208知名度"] = 100 - df["202208知名度"]
df["log(100-202208知名度)"] = np.log10(df["100-202208知名度"] + 1)

df["100-201902知名度"] = 100 - df["201902知名度"]
df["log(100-201902知名度)"] = np.log10(df["100-201902知名度"] + 1)

df["log(100-201902知名度)*weightedPV1_past"] = (
    df["log(100-201902知名度)"] * df["weightedPV1_past"]
)
df["(100-201902知名度)*weightedPV1_past"] = (
    df["100-201902知名度"] * df["weightedPV1_past"]
)

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

features = [
    "100-201902知名度::",
    "log(100-201902知名度)*weightedPV1_past::",
    "(100-201902知名度)*weightedPV1_past::",
]

target = "aware_diff_past"

# =====================================
# 2. gender別にデータ分割
# =====================================
df_male = df[df["gender"] == "男性"]
df_female = df[df["gender"] == "女性"]

def train_and_predict(df_subset):
    X = df_subset[features]
    y = df_subset[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)

    # Lasso
    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)

    metrics = {
        "lin": {
            "Coefficients": lin.coef_,
            "Intercept": lin.intercept_,
            "r2": r2_score(y_test, y_pred_lin),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_lin)),
            "mape": mean_absolute_percentage_error(y_test, y_pred_lin),
        },
        "lasso": {
            "r2": r2_score(y_test, y_pred_lasso),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
            "mape": mean_absolute_percentage_error(y_test, y_pred_lasso),
        },
    }

    return y_test, y_pred_lin, y_pred_lasso, metrics


# =====================================
# 3. 学習・評価
# =====================================
y_m, y_m_lin, y_m_lasso, met_m = train_and_predict(df_male)
y_f, y_f_lin, y_f_lasso, met_f = train_and_predict(df_female)

print("====== 男性モデル ======")
print("Linear :", met_m["lin"])
print("Lasso  :", met_m["lasso"])

print("\n====== 女性モデル ======")
print("Linear :", met_f["lin"])
print("Lasso  :", met_f["lasso"])

# =====================================
# 4. Visualization（男女別モデル比較）
# =====================================
plt.figure(figsize=(12, 5))

# ---- Linear ----
plt.subplot(1, 2, 1)
plt.scatter(y_m, y_m_lin, alpha=0.6, label="男性", c="tab:blue")
plt.scatter(y_f, y_f_lin, alpha=0.6, label="女性", c="tab:orange")
plt.plot(
    [df[target].min(), df[target].max()],
    [df[target].min(), df[target].max()],
    "r--"
)
plt.title("Linear Regression（男女別モデル）")
plt.xlabel("Actual aware_diff")
plt.ylabel("Predicted")
plt.legend()

# ---- Lasso ----
plt.subplot(1, 2, 2)
plt.scatter(y_m, y_m_lasso, alpha=0.6, label="男性", c="tab:blue")
plt.scatter(y_f, y_f_lasso, alpha=0.6, label="女性", c="tab:orange")
plt.plot(
    [df[target].min(), df[target].max()],
    [df[target].min(), df[target].max()],
    "r--"
)
plt.title("Lasso Regression（男女別モデル）")
plt.xlabel("Actual aware_diff")
plt.ylabel("Predicted")
plt.legend()

plt.tight_layout()
plt.show()
