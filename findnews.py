from gnews import GNews
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime
from calendar import monthrange
import csv


with open("pop_all.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    csv_list = []
    for row in reader:
        csv_list.append(row[1])
    csv_list = csv_list[1:22]

# ===== 設定 =====
start_date = (2022, 8, 1)
end_date = (2023, 2, 28)
max_results_per_month = 30

# ===== 感情分析モデルセットアップ =====
model_name = "jarvisx17/japanese-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# ===== GNews初期化 =====
google_news = GNews(language='ja', country='JP', max_results=max_results_per_month)

# ===== 月ごとの年月ペア生成 =====
def month_range(start, end):
    result = []
    current = datetime(*start)
    end_dt = datetime(*end)
    while current <= end_dt:
        result.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return result

# ===== ニュース収集＋感情分析 =====
def news_ana(query):
    all_articles = []
    for year, month in month_range(start_date, end_date):
        # 月の初日・末日を設定（同日にならないように）
        google_news.start_date = (year, month, 1)
        last_day = monthrange(year, month)[1]
        google_news.end_date = (year, month, last_day)

        try:
            articles = google_news.get_news(query)
            for a in articles:
                result = sentiment_analyzer(a['title'])[0]
                all_articles.append({
                    'name': query,
                    'date': a['published date'] if a.get('published date') else '',
                    'title': a['title'],
                    'url': a['url'],
                    'publisher': a.get('publisher', {}).get('title', ''),
                    'sentiment': result['label'],
                    'score': round(result['score'], 3)
                })
        except Exception as e:
            print(f"Error in {year}-{month}: {e}")
    print(f"{query} end")
    return all_articles

output = []
for name in csv_list:
    output = output + news_ana(name)
# ===== CSVに保存 =====
df = pd.DataFrame(output)
df.to_csv("pop_news.csv", index=False, encoding='utf-8-sig')

print("✅ 完了しました。'pop_news.csv' を出力しました。")
