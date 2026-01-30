import pandas as pd
from pathlib import Path

from app.feature_engineering import write_dataframe_to_csv
from app.model_class import SentimentAnalyzer
from app.utils import _load_daily_news


def calculate_sentiment_score(stock_name: str, start_date: str, end_date: str):
    model_dir = "finbert"
    sa = SentimentAnalyzer(model_path=model_dir)

    date_range = pd.date_range(start=start_date, end=end_date, freq="D").strftime("%Y-%m-%d")
    rows = []
    for date in date_range:
        articles = _load_daily_news(stock_name, date)
        if not articles:
            print(f"No news record found on {date}")
            continue

        titles = [a.get("title", "") or "" for a in articles]
        contents = [a.get("content", "") or "" for a in articles]

        title_scores = sa.predict(titles)
        content_scores = sa.predict(contents)

        n = len(articles)
        avg = lambda scores, key: sum(s[key] for s in scores) / n

        rows.append({
            "date": date,
            "title_positive": avg(title_scores, "positive"),
            "title_neutral": avg(title_scores, "neutral"),
            "title_negative": avg(title_scores, "negative"),
            "content_positive": avg(content_scores, "positive"),
            "content_neutral": avg(content_scores, "neutral"),
            "content_negative": avg(content_scores, "negative"),
        })

    if not rows:
        return {}

    # Build DataFrame
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Write all columns at once
    csv_path = Path("data") / stock_name / "prediction.csv"
    write_dataframe_to_csv(
        file_path=csv_path,
        csv_col_name=[
            "title_positive", "title_neutral", "title_negative",
            "content_positive", "content_neutral", "content_negative"
        ],
        df=df,
        df_col_name=[
            "title_positive", "title_neutral", "title_negative",
            "content_positive", "content_neutral", "content_negative"
        ],
        col_type=[float] * 6
    )

    # Return dictionary for backwards compatibility / inspection
    return df.to_dict(orient="index")