from gnews import GNews
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime
from calendar import monthrange
import csv


with open("pop_all.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    csv_list = list(reader)
    csv_list = csv_list[1:22]

with open("pop_news.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    news = list(reader)


output = [["name", "人気度変化","知名度変化", "スコア", "ニュース数", "positive数", "negative数","202208人気度", "202208知名度", "202302人気度", "202302知名度"]]
for row in csv_list:
    pnews = [i for i in news if row[1] == i[0]]
    pnews_p = [i for i in pnews if i[5] == "positive"]
    pnews_n = [i for i in pnews if i[5] == "negative"]
    output.append([row[1], float(row[4])-float(row[2]), float(row[5])-float(row[3]), len(pnews_p)-len(pnews_n), len(pnews), len(pnews_p), len(pnews_n), row[2], row[3], row[4], row[5]])

output_file_path = "pop_news_ana.csv"
with open(output_file_path, 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(output)