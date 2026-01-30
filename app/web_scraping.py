from collections import defaultdict
from datetime import datetime, timedelta
import newspaper
import requests
from typing import List, Dict, Optional
from zoneinfo import ZoneInfo

from app.utils import _parse_date_to_ymd, _store_daily_news


def _fetch_gdelt_articles(
    stock_name: str,
    company_name: str,
    start_date: str,
    end_date: str,
    daily_max_amount: int,
    title_only: bool = False,
) -> List[Dict]:
    """Fetch articles from GDELT Document API for a query and date range."""

    base = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Convert to GDELT datetime format YYYYMMDDHHMMSS
    # Align to ET trading hours (09:00 ET start_date -> next day 08:59 ET after end_date)
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    except Exception:
        # if parsing fails, raise ValueError
        raise ValueError("start_date and end_date must be 'YYYY-MM-DD' strings")

    # Always align to NY trading hours (09:00 ET -> next day 08:59 ET).
    ed_next = end_date + timedelta(days=1)
    ny_zone = ZoneInfo("America/New_York")
    utc_zone = ZoneInfo("UTC")
    start_ny = datetime(start_date.year, start_date.month, start_date.day, 9, 0, 0, tzinfo=ny_zone)
    end_ny = datetime(ed_next.year, ed_next.month, ed_next.day, 8, 59, 0, tzinfo=ny_zone)
    start_utc = start_ny.astimezone(utc_zone).strftime("%Y%m%d%H%M%S")
    end_utc = end_ny.astimezone(utc_zone).strftime("%Y%m%d%H%M%S")

    us_filter = "sourcecountry:US"
    query = "(" + f"{stock_name} OR {company_name}".strip() + ")"
    query_and_GEO = f"{query} {us_filter}" if query else us_filter

    params = {
        "query": query_and_GEO,
        "trans": "googtrans",
        "mode": "ArtList",
        "startdatetime": start_utc,
        "enddatetime": end_utc,
        "maxrecords": daily_max_amount,
        "format": "json",
    }
    headers = {
        "Accept": "application/json",
        "User-Agent": "IS5803_SentimentAnalysis/1.0",
    }

    try:
        resp = requests.get(base, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        # Return empty list on network/API error
        return []

    # Basic status/debug handling
    if resp.status_code != 200:
        try:
            # try to surface error details
            err_text = resp.text
        except Exception:
            err_text = "(no response body)"
        print(f"GDELT API request failed: status={resp.status_code}, body={err_text}")
        return []

    data = resp.json()
    articles = data.get("articles") or []
    results: List[Dict] = []
    for a in articles:
        date = _parse_date_to_ymd(a.get("seendate"))
        title = a.get("title")
        url = a.get("url") or a.get("url_mobile")
        source = a.get("domain")

        results.append({
            "date": date,
            "title": title,
            "url": url,
            "source": source,
        })

    if title_only:
        keywords = [stock_name.upper(), company_name.upper()]
        filtered: List[Dict] = []
        for r in results:
            t = r.get('title')
            if not t:
                continue
            tu = t.upper()
            for k in keywords:
                if k in tu:
                    filtered.append(r)
                    break
        results = filtered

    return results


def _scrape_article_contents(articles: List[Dict]) -> List[Dict]:
    """Scrape article contents for a list of articles with 'url' key."""

    for article in articles:
        url = article.get("url")
        if not url:
            continue
        try:
            np = newspaper.Article(url)
            np.download()
            np.parse()
            article["content"] = np.text
        except Exception:
            article["content"] = ""
    
    return articles


def get_stock_news(stock_name: str, company_name: str, start_date: str, end_date: str, daily_max_amount: int) -> Dict[str, Optional[List[Dict]]]:
    """Return news for a stock between start_date and end_date using GDELT."""

    # Request title-only post-filtering so results must contain the stock name or company name in the article title.
    articles = _fetch_gdelt_articles(stock_name, company_name, start_date, end_date, daily_max_amount, title_only=True)
    
    # Scrape full article contents
    articles_with_content = _scrape_article_contents(articles)

    # Store articles of the end_date into a JSON file
    articles_by_date = defaultdict(list)
    for a in articles_with_content:
        article_date = a.get("date")
        articles_by_date[article_date].append(a)
    for d in articles_by_date.keys():
        _store_daily_news(stock_name, d, articles_by_date[d].copy())

    return {
        "stock": stock_name,
        "date": f"{start_date} to {end_date}",
        "news": articles_with_content if articles_with_content else None,
    }