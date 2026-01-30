import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, log_loss,
    confusion_matrix, precision_recall_curve, auc
)
from typing import List, Optional, Union

from app.utils import _plot_pr_curve, _plot_roc_curve, _shift_date


def _handle_missing_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing dates with previous day's value using forward fill.

    If 'Date' column exists, use it as the index; otherwise use the existing index.
    Returns a DataFrame with complete daily index and forward-filled values.
    """
    if df.empty:
        return df.copy()

    # --- Determine index ---
    if "Date" in df.columns:
        idx = pd.to_datetime(df["Date"], errors="coerce")
    else:
        idx = pd.to_datetime(df.index, errors="coerce")

    if idx.isna().all():
        raise ValueError("All dates are invalid or cannot be converted to datetime")

    df = df.copy()
    df.index = idx
    df.index.name = "date"

    # --- Create complete daily date range ---
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")

    # --- Reindex and forward-fill ---
    df = df.reindex(full_idx)
    df = df.fillna(method="ffill")

    return df


def write_dataframe_to_csv(
    file_path: Union[str, Path],
    csv_col_name: Union[str, List[str]],
    df: pd.DataFrame,
    df_col_name: Union[str, List[str]],
    col_type: Union[type, List[type]]
) -> None:
    """
    Write one or multiple columns from a DataFrame into a CSV keyed by date.
    
    csv_col_name : str or list of str
        Names to write into CSV.
    df_col_name : str or list of str
        Column names from DF to read.
    col_type : type or list of type
        Target type(s) for output columns.
    """

    # --- Normalize inputs to lists ---
    if isinstance(csv_col_name, str):
        csv_cols = [csv_col_name]
    else:
        csv_cols = list(csv_col_name)

    if isinstance(df_col_name, str):
        df_cols = [df_col_name]
    else:
        df_cols = list(df_col_name)

    if isinstance(col_type, type):
        types = [col_type] * len(df_cols)
    else:
        types = list(col_type)

    if len(csv_cols) != len(df_cols) or len(types) != len(df_cols):
        raise ValueError("Lengths of csv_col_name, df_col_name, and col_type must match")

    # --- Ensure output folder exists ---
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # --- Normalize dataframe ---
    df = _handle_missing_value(df)

    # --- Validate df columns ---
    for col in df_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input DataFrame")

    # --- Convert index to YYYY-MM-DD strings ---
    try:
        df.index = df.index.strftime("%Y-%m-%d")
    except Exception:
        df.index = df.index.astype(str)

    # --- Load/create CSV ---
    if not p.exists():
        out = pd.DataFrame(index=pd.Index([], name="date"))
        out.to_csv(p)

    existing = pd.read_csv(p, index_col=0)

    # Normalize date index
    try:
        existing.index = pd.to_datetime(existing.index).strftime("%Y-%m-%d")
    except Exception:
        existing.index = existing.index.astype(str)

    # Ensure all requested CSV columns exist
    for col in csv_cols:
        if col not in existing.columns:
            existing[col] = pd.NA

    # --- Update CSV with df values ---
    for csv_name, df_name, typ in zip(csv_cols, df_cols, types):
        series = df[df_name]

        # Type coercion with fallback
        try:
            series = series.astype(typ)
        except Exception:
            if typ in (float, int):
                series = pd.to_numeric(series, errors="coerce").astype(typ)
            else:
                series = series.astype(str)

        # Insert values; skip NaNs
        for date, value in series.items():
            if pd.isna(value):
                continue
            existing.loc[date, csv_name] = value

    # --- Save CSV ---
    existing = existing.sort_index()
    existing.to_csv(p)


def load_csv_to_dataframe(
    file_path: Union[str, Path],
    parse_dates: bool = True,
    expected_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load a CSV written by `write_dataframe_to_csv` into a normalized DataFrame.

    Behavior:
    - If the CSV does not exist → return empty DataFrame with DatetimeIndex.
    - Normalizes index to DatetimeIndex if parse_dates=True.
    - Optionally ensures the existence of expected columns.

    Parameters:
        file_path: Path to the CSV.
        parse_dates: Convert the index to datetime (default True).
        expected_cols: List of required columns. Missing ones will be created as pd.NA

    Returns:
        A pandas DataFrame with normalized index and all expected columns.
    """

    p = Path(file_path)
    if not p.exists():
        # Create empty DF with datetime index
        idx = pd.to_datetime([], errors="coerce")
        df = pd.DataFrame(index=idx)
        if expected_cols:
            for col in expected_cols:
                df[col] = pd.NA
        return df

    # Read existing CSV
    try:
        df = pd.read_csv(p, index_col=0)
    except Exception as e:
        print(f"Failed to load CSV {p}: {e}")
        return pd.DataFrame()

    # Normalize index to datetime
    if parse_dates:
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            # fallback: keep strings
            df.index = df.index.astype(str)

    # Ensure expected columns exist
    if expected_cols:
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA

    return df


def calculate_rolling_price(
    stock_name: str,
    start_date: str,
    end_date: str,
    window_size: int
) -> dict:
    """
    Calculate rolling average closing prices for a stock.

    - Uses slice-first optimization to avoid processing the entire dataset.
    - Assumes CSV contains continuous daily data with 'close_price' column.
    - Does not use min_periods; the first (window_size - 1) days will be NaN.
    """

    csv_path = Path("data") / stock_name / "prediction.csv"

    price_df = load_csv_to_dataframe(
        file_path=csv_path,
        parse_dates=True,
        expected_cols=["close_price"]
    )

    if price_df is None or price_df.empty:
        return {
            "stock": stock_name,
            "date_range": f"{start_date} to {end_date}",
            "rolling_price": None,
            "trend_flag": None
        }

    # Ensure datetime index and numeric values
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()
    price_df["close_price"] = pd.to_numeric(price_df["close_price"], errors="coerce")

    # Compute extended_start_date to cover full rolling window
    extended_start_date = _shift_date(start_date, 1 - window_size)
    extended_start_dt = pd.to_datetime(extended_start_date)
    end_dt = pd.to_datetime(end_date)
    start_dt = pd.to_datetime(start_date)

    # Slice only the relevant range for efficiency
    slice_mask = (price_df.index >= extended_start_dt) & (price_df.index <= end_dt)
    sliced_df = price_df.loc[slice_mask].copy()

    if sliced_df.empty:
        return {
            "stock": stock_name,
            "date_range": f"{start_date} to {end_date}",
            "rolling_price": None,
            "trend_flag": None
        }

    # Compute rolling average
    rolling_col_name = f"rolling_price_{window_size}"
    sliced_df[rolling_col_name] = sliced_df["close_price"].rolling(window=window_size).mean()

    # Compute rise/drop trend flag
    def _trend_flag(row):
        if pd.isna(row[rolling_col_name]) or pd.isna(row["close_price"]):
            return pd.NA
        return 1 if row["close_price"] > row[rolling_col_name] else 0

    trend_col_name = f"trend_flag_{window_size}"
    sliced_df[trend_col_name] = sliced_df.apply(_trend_flag, axis=1)

    # Slice back to user-requested window
    final_mask = (sliced_df.index >= start_dt) & (sliced_df.index <= end_dt)
    final_df = sliced_df.loc[final_mask, [rolling_col_name, trend_col_name]].copy()

    # Write to CSV
    write_dataframe_to_csv(
        file_path=csv_path,
        csv_col_name=[rolling_col_name, trend_col_name],
        df=final_df,
        df_col_name=[rolling_col_name, trend_col_name],
        col_type=[float, int]
    )

    return {
        "stock": stock_name,
        "date_range": f"{start_date} to {end_date}",
        "rolling_price": final_df[rolling_col_name].to_dict(),
        "trend_flag": final_df[trend_col_name].to_dict()
    }


def derive_equation(stock_name: str, start_date: str, end_date: str, window_size: int) -> dict:
    """
    Train logistic regression to predict trend_flag.
    Produces per-fold metrics + aggregated metrics.
    Stores metrics and plots under stock's data folder.
    """
    csv_path = Path("data") / stock_name / "prediction.csv"
    data_dir = Path("data") / stock_name
    data_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        "title_positive", "title_neutral", "title_negative",
        "content_positive", "content_neutral", "content_negative"
    ]
    target_col = f"trend_flag_{window_size}"

    df = load_csv_to_dataframe(csv_path, parse_dates=True, expected_cols=feature_cols + [target_col])
    if df is None or df.empty:
        return {"stock": stock_name, "date_range": f"{start_date} to {end_date}", "equation": None, "metrics": None}

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df[feature_cols + [target_col]] = df[feature_cols + [target_col]].apply(pd.to_numeric, errors="coerce")

    # Slice to requested date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df.index >= start_dt) & (df.index <= end_dt)].copy()
    if df.empty:
        return {"stock": stock_name, "date_range": f"{start_date} to {end_date}", "equation": None, "metrics": None}

    X = df[feature_cols].values
    y = df[target_col].values

    # old: tscv = TimeSeriesSplit(n_splits=5)
    tscv = TimeSeriesSplit(n_splits=3)
    fold_metrics = []
    y_true_all, y_prob_all = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # old: model = LogisticRegression(max_iter=300)
        model = LogisticRegression(max_iter=300, class_weight='balanced')
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        roc_auc = roc_auc_score(y_test, y_prob)
        logloss = log_loss(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        pr_auc = auc(*precision_recall_curve(y_test, y_prob)[:2][::-1])  # recall first, then precision

        fold_metrics.append({
            "fold": fold_idx,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "log_loss": logloss,
            "confusion_matrix": cm.tolist()
        })

        y_true_all.extend(y_test)
        y_prob_all.extend(y_prob)

    # Aggregate metrics
    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    y_pred_all = (y_prob_all >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average="binary")
    roc_auc = roc_auc_score(y_true_all, y_prob_all)
    logloss = log_loss(y_true_all, y_prob_all)
    cm = confusion_matrix(y_true_all, y_pred_all)
    pr_auc = auc(*precision_recall_curve(y_true_all, y_prob_all)[:2][::-1])

    agg_metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": logloss,
        "confusion_matrix": cm.tolist()
    }

    # Plot only aggregate curves
    _plot_roc_curve(y_true_all, y_prob_all, data_dir / "roc_curve.png", roc_auc)
    _plot_pr_curve(y_true_all, y_prob_all, data_dir / "pr_curve.png", pr_auc)

    # Retrain model on full data for equation
    # old: final_model = LogisticRegression(max_iter=300)
    final_model = LogisticRegression(max_iter=300, class_weight='balanced')
    final_model.fit(X, y)
    equation = "y = sigmoid(" + " + ".join([f"{coef:.4f}*{feat}" for coef, feat in zip(final_model.coef_[0], feature_cols)]) + f" + {final_model.intercept_[0]:.4f})"

    # Save metrics
    with open(data_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("Equation:\n")
        f.write(f"{equation}\n\n")
        f.write("Per-fold metrics:\n")
        f.write(f"{fold_metrics}\n\n")
        f.write("Aggregated metrics:\n")
        f.write(f"{agg_metrics}\n")

    return {
        "stock": stock_name,
        "date_range": f"{start_date} to {end_date}",
        "equation": equation,
        "metrics": {
            "per_fold": fold_metrics,
            "aggregated": agg_metrics
        }
    }