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
# Feature Engineering
# =====================================
df["100-202208知名度"] = 100 - df["202208知名度"]
df["log(100-202208知名度)"] = np.log10(df["100-202208知名度"] + 1)
df["log(100-202208知名度)^2"] = df["log(100-202208知名度)"] ** 2
df["log(100-202208知名度)*weightedPV1"] = df["log(100-202208知名度)"] * df["weightedPV1"]
df["(100-202208知名度)*weightedPV1"] = df["100-202208知名度"] * df["weightedPV1"]
df["(100-202208知名度)*allPV"] = df["100-202208知名度"] * df["allPV"]

df["100-201902知名度"] = 100 - df["201902知名度"]
df["log(100-201902知名度)"] = np.log10(df["100-201902知名度"] + 1)
df["log(100-201902知名度)^2"] = df["log(100-201902知名度)"] ** 2
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

ex = [
    "100-201902知名度::",
    "log(100-201902知名度)*weightedPV1_past::",
    "(100-201902知名度)*weightedPV1_past::",
]

X = df[ex].copy()
y = df["aware_diff_past"]
gender = df["gender"]

# =====================================
# 2. Train-Test Split（gender保持）
# =====================================
X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
    X, y, gender,
    test_size=0.2,
    random_state=42
)

# =====================================
# 3. Linear Regression
# =====================================
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

print("========== Linear Regression ==========")
print("Coefficients:", lin_reg.coef_)
print("Intercept:", lin_reg.intercept_)
print("R2:", r2_score(y_test, y_pred_lin))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lin)))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred_lin))
print("MRE :", mean_relative_error(y_test, y_pred_lin))

# =====================================
# 4. Lasso Regression
# =====================================
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print("\n========== Lasso Regression ==========")
print("R2:", r2_score(y_test, y_pred_lasso))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lasso)))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred_lasso))
print("MRE :", mean_relative_error(y_test, y_pred_lasso))

# =====================================
# 5. Visualization（gender別・1枚）
# =====================================
plt.figure(figsize=(12, 5))

# ---- Linear ----
plt.subplot(1, 2, 1)
for g, color in [("男性", "tab:blue"), ("女性", "tab:orange")]:
    mask = (gender_test == g)
    plt.scatter(
        y_test[mask],
        y_pred_lin[mask],
        alpha=0.6,
        label=g,
        c=color
    )

plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.title("Linear Regression（gender別）")
plt.xlabel("Actual aware_diff")
plt.ylabel("Predicted")
plt.legend()

# ---- Lasso ----
plt.subplot(1, 2, 2)
for g, color in [("男性", "tab:blue"), ("女性", "tab:orange")]:
    mask = (gender_test == g)
    plt.scatter(
        y_test[mask],
        y_pred_lasso[mask],
        alpha=0.6,
        label=g,
        c=color
    )

plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.title("Lasso Regression（gender別）")
plt.xlabel("Actual aware_diff")
plt.ylabel("Predicted")
plt.legend()

plt.tight_layout()
plt.show()
