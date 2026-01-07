import pandas as pd
import numpy as np
import re
from janome.tokenizer import Tokenizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =====================
# 設定
# =====================
INPUT_CSV = "alllist_with_content.csv"
RANDOM_STATE = 42

# =====================
# データ読み込み
# =====================
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=["content", "aware_diff", "allPV"])
df = df[df["allPV"] > 0]

# =====================
# 目的変数・制御変数
# =====================
df["log_aware"] = np.log1p(df["aware_diff"])
df["log_pv"] = np.log1p(df["allPV"])

# =====================
# 文章構造特徴量
# =====================
tokenizer = Tokenizer()

def extract_text_features(text):
    text = str(text)

    tokens = list(tokenizer.tokenize(text))
    surfaces = [t.surface for t in tokens]

    noun_count = sum(
        1 for t in tokens if t.part_of_speech.startswith("名詞")
    )
    proper_noun_count = sum(
        1 for t in tokens if "固有名詞" in t.part_of_speech
    )

    year_count = len(re.findall(r"(19|20)\d{2}", text))
    katakana_count = sum(
        1 for s in surfaces if re.fullmatch(r"[ァ-ヴー]+", s)
    )

    return pd.Series({
        "char_len": len(text),
        "token_len": len(tokens),
        "noun_ratio": noun_count / max(len(tokens), 1),
        "proper_noun_ratio": proper_noun_count / max(noun_count, 1),
        "year_count": year_count,
        "katakana_ratio": katakana_count / max(len(tokens), 1),
        "section_count": text.count("\n==")
    })

features = df["content"].apply(extract_text_features)

df = pd.concat([df, features], axis=1)

# =====================
# 説明変数
# =====================
X = df[
    [
        "log_pv",
        "char_len",
        "token_len",
        "noun_ratio",
        "proper_noun_ratio",
        "year_count",
        "katakana_ratio",
        "section_count"
    ]
]

y = df["aware_diff"] * 1000000

# =====================
# 学習・評価
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("===================================")
print(f"R2 score (test): {r2:.4f}")
print("===================================")

# =====================
# 係数の解釈
# =====================
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coef": model.coef_
}).sort_values("coef", ascending=False)

print("\n--- 係数（log awre_diff への影響）---")
print(coef_df.to_string(index=False))

# 保存
coef_df.to_csv(
    "structure_coef_awre_with_pv2.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n係数を保存しました: structure_coef_awre_with_pv2.csv")
