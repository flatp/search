from gnews import GNews
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime
from calendar import monthrange
import csv

with open("pop_news_1-100.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    news0 = list(reader)

with open("pop_news_101-200.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    news1 = list(reader)

with open("pop_news_201-300.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    news2 = list(reader)

with open("pop_news_301-400.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    news3 = list(reader)

with open("pop_news_401-500.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    news4 = list(reader)

with open("pop_news_501-600.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    news5 = list(reader)

with open("pop_news_601-700.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    news6 = list(reader)

with open("pop_news_701-800.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    news7 = list(reader)

with open("pop_news_801-900.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    news8 = list(reader)

with open("pop_news_901-.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    news9 = list(reader)

output = news0 + news1[1:] + news2[1:] + news3[1:] + news4[1:] + news5[1:] + news6[1:] + news7[1:] + news8[1:] + news9[1:]
# ===== CSVに保存 =====
df = pd.DataFrame(output)
df.to_csv("pop_news_all.csv", index=False, encoding='utf-8-sig')

print("✅ 完了しました。'pop_news.csv' を出力しました。")
