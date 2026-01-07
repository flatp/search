import pandas as pd
import numpy as np
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =====================
# 設定
# =====================
INPUT_CSV = "alllist_with_content.csv"
MIN_DF = 5
MAX_DF = 0.8
ALPHA = 0.001
RANDOM_STATE = 42

# =====================
# データ読み込み
# =====================
df = pd.read_csv(INPUT_CSV)

# 必要な行だけ残す
df = df.dropna(subset=["content", "aware_diff", "allPV"])
df = df[df["allPV"] > 0]

# =====================
# 目的変数
# =====================
# 1PVあたり awre_diff
df["y"] = df["aware_diff"] * 1000000 / df["allPV"]
df["y_log"] = np.log1p(df["y"])

# =====================
# 日本語トークナイザ
# =====================
tokenizer = Tokenizer()

def tokenize(text):
    tokens = []
    for token in tokenizer.tokenize(str(text)):
        pos = token.part_of_speech.split(",")[0]
        if pos == "名詞":
            tokens.append(token.base_form)
    return tokens

# =====================
# TF-IDF
# =====================
vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    min_df=MIN_DF,
    max_df=MAX_DF
)

X = vectorizer.fit_transform(df["content"])
y = df["y"]

# =====================
# 学習・評価
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

model = Lasso(alpha=ALPHA)
model.fit(X_train, y_train)

# 評価
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("===================================")
print(f"R2 score (test): {r2:.4f}")
print("===================================")

# =====================
# 単語重要度
# =====================
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_

coef_df = pd.DataFrame({
    "word": feature_names,
    "coef": coef
}).sort_values("coef", ascending=False)

# =====================
# 結果出力
# =====================
print("\n--- 1PVあたり awre_diff を押し上げる単語 TOP 30 ---")
print(coef_df.head(30).to_string(index=False))

print("\n--- 1PVあたり awre_diff を下げる単語 TOP 30 ---")
print(coef_df.tail(30).to_string(index=False))

# CSVにも保存
coef_df.to_csv("word_coef_awre_diff_per_PV.csv", index=False, encoding="utf-8-sig")

print("\n単語係数を保存しました: word_coef_awre_diff_per_PV.csv")
