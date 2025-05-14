import pandas as pd
from scipy.stats import spearmanr, pearsonr

def evaluate(filename):
    df1 = pd.read_csv(filename, header=None, names=["name", "score1"])
    df2 = pd.read_csv("pop_score.csv", header=None, names=["name", "score2", "score3"])

    # 人名で内部結合（共通する人のみ対象）
    merged = pd.merge(df1, df2, on="name")

    # スピアマンの順位相関係数を計算
    corr, p_value = spearmanr(merged["score1"], merged["score2"])
    print(filename)
    print("人気度")
    print(f"スピアマンの順位相関係数: {corr:.4f}")
    print(f"p値: {p_value:.4e}")
    corr, p_value = pearsonr(merged["score1"], merged["score2"])
    print(f"ピアソンの相関係数: {corr:.4f}")
    print(f"p値: {p_value:.4e}")
    corr, p_value = spearmanr(merged["score1"], merged["score3"])
    print("知名度")
    print(f"スピアマンの順位相関係数: {corr:.4f}")
    print(f"p値: {p_value:.4e}")
    corr, p_value = pearsonr(merged["score1"], merged["score3"])
    print(f"ピアソンの相関係数: {corr:.4f}")
    print(f"p値: {p_value:.4e}")

evaluate("br_direct.csv")
evaluate("browserank_true.csv")
evaluate("browserank_true_r.csv")