from pathlib import Path
import yfinance as yf

from app.feature_engineering import write_dataframe_to_csv
from app.utils import _shift_date


def get_stock_price(stock_name: str, start_date: str, end_date: str):
    """
    Fetch adjusted closing prices for a stock, write to CSV, and return raw data for inspection.

    - stock_name: ticker symbol
    - start_date, end_date: strings 'YYYY-MM-DD'
    """
    # Fetch historical data
    ticker = yf.Ticker(stock_name)
    data = ticker.history(
        start=start_date,
        end=_shift_date(end_date, days=1),
        auto_adjust=True
    )

    if data.empty:
        price_df = None
    else:
        price_df = data[["Close"]].copy()
        # Write to CSV
        csv_path = Path("data") / stock_name / "prediction.csv"
        write_dataframe_to_csv(
            file_path=csv_path,
            csv_col_name="close_price",
            df=price_df,
            df_col_name="Close",
            col_type=float
        )

    # Return raw data for inspection
    return {
        "stock": stock_name,
        "date_range": f"{start_date} to {end_date}",
        "price": price_df if price_df is not None else None
    }