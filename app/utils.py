from datetime import datetime, timedelta, timezone
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve
from typing import List, Optional, Union
from zoneinfo import ZoneInfo

# Cache default timezone to avoid repeated construction
try:
    DEFAULT_TZ = ZoneInfo("America/Seattle")
except Exception:
    DEFAULT_TZ = timezone(timedelta(hours=-8))


def _shift_date(date: str, days: int = 0) -> str:
    """
    Shift a date by a given number of days.

    Parameters:
        date (str): Input date in 'YYYY-MM-DD' format.
        days (int): Number of days to shift.
                   Positive → move forward.
                   Negative → move backward.

    Returns:
        str: Shifted date in 'YYYY-MM-DD' format.

    Raises:
        ValueError: If the input date is invalid.
    """
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        shifted = dt + timedelta(days=days)
        return shifted.strftime("%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Invalid date string '{date}': {e}")


def _parse_date_to_ymd(date_raw: Union[str, int, float, None]) -> Optional[str]:
    """
    Coerce various date formats to 'YYYY-MM-DD' using DEFAULT_TZ.
    
    Supports:
    - epoch seconds (int or float)
    - ISO strings
    - common date formats like '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%Y%m%d%H%M%S', '%Y-%m-%d'
    """
    if date_raw is None:
        return None

    # Epoch time
    if isinstance(date_raw, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(date_raw), tz=timezone.utc).astimezone(DEFAULT_TZ)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    # String parsing
    if isinstance(date_raw, str):
        try:
            dt = datetime.fromisoformat(date_raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=DEFAULT_TZ)
            else:
                dt = dt.astimezone(DEFAULT_TZ)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y%m%d%H%M%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(date_raw, fmt).replace(tzinfo=DEFAULT_TZ)
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    continue

    # Failed to parse
    return None


def _store_daily_news(stock_name: str, date_str: str, daily_news: list) -> None:
    """
    Store news of a specific day into a JSON file.
    
    - stock_name: ticker or identifier
    - date_str: 'YYYY-MM-DD'
    - daily_news: list of news dicts
    """
    if not isinstance(daily_news, list):
        raise TypeError(f"daily_news must be a list, got {type(daily_news)}")

    folder_path = Path("data") / stock_name / "news"
    file_path = folder_path / f"{date_str}.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(daily_news, f, ensure_ascii=False, indent=4)


def _load_daily_news(stock_name: str, date_str: str) -> List[dict]:
    """
    Load daily news from JSON file.

    - stock_name: ticker or identifier
    - date_str: 'YYYY-MM-DD'
    Returns a list of news dicts; empty list if file does not exist or fails to parse.
    """
    file_path = Path("data") / stock_name / "news" / f"{date_str}.json"
    if not file_path.exists():
        # File missing, return empty list
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        else:
            # Unexpected format
            return []
    except Exception as e:
        # Parsing error
        print(f"Failed to load {file_path}: {e}")
        return []
    

def _plot_roc_curve(y_true, y_score, save_path, auc_value):
    """
    Plot aggregated ROC curve and save to file.
    
    y_true: true binary labels
    y_score: predicted probabilities for positive class
    save_path: Path object or string to save figure
    auc_value: ROC-AUC score (float) for labeling
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Aggregated ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()


def _plot_pr_curve(y_true, y_score, save_path, pr_auc_value):
    """
    Plot aggregated Precision-Recall curve and save to file.
    
    y_true: true binary labels
    y_score: predicted probabilities for positive class
    save_path: Path object or string to save figure
    pr_auc_value: PR-AUC score (float) for labeling
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
    
    plt.figure(figsize=(6, 6))
    plt.plot(recall_vals, precision_vals, label=f"PR curve (AUC = {pr_auc_value:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Aggregated Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()