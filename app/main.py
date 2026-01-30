import datetime as dt
from fastapi import FastAPI

from app.feature_engineering import calculate_rolling_price, derive_equation
from app.sentiment_analysis import calculate_sentiment_score
from app.technical_analysis import get_stock_price
from app.web_scraping import get_stock_news


app = FastAPI(title="Sentiment Analysis API")


@app.get("/")
async def root():
    return {"message": "Hello World!"}


@app.get("/logistic_regression/{stock_name}")
async def logistic_regression(
    stock_name: str,
    start_date: str | None = dt.date.today().isoformat(),
    end_date: str | None = dt.date.today().isoformat(),
    window_size: int | None = 7
):
    stock_name = stock_name.upper()
    start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    window_size = int(window_size) if window_size and window_size > 0 else 7
    return derive_equation(stock_name, start_date, end_date, window_size)


@app.get("/news/{stock_name}")
async def news(
    stock_name: str,
    company_name: str | None = "",
    start_date: str | None = dt.date.today().isoformat(),
    end_date: str | None = dt.date.today().isoformat(),
    daily_max_amount: int | None = 10
):
    stock_name = stock_name.upper()
    company_name = company_name
    start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    return get_stock_news(stock_name, company_name, start_date, end_date, daily_max_amount)


@app.get("/rolling_price/{stock_name}")
async def rolling_price(
    stock_name: str,
    start_date: str | None = dt.date.today().isoformat(),
    end_date: str | None = dt.date.today().isoformat(),
    window_size: int | None = 7
):
    stock_name = stock_name.upper()
    start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    window_size = int(window_size) if window_size and window_size > 0 else 7
    return calculate_rolling_price(stock_name, start_date, end_date, window_size)


@app.get("/sentiment_score/{stock_name}")
async def sentiment_score(
    stock_name: str,
    start_date: str | None = dt.date.today().isoformat(),
    end_date: str | None = dt.date.today().isoformat()
):
    stock_name = stock_name.upper()
    start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    return calculate_sentiment_score(stock_name, start_date, end_date)


@app.get("/stock_price/{stock_name}")
async def stock_price(
    stock_name: str,
    start_date: str | None = dt.date.today().isoformat(),
    end_date: str | None = dt.date.today().isoformat()
):
    stock_name = stock_name.upper()
    start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    return get_stock_price(stock_name, start_date, end_date)
