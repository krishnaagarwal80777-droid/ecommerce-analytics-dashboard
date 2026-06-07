
import pandas as pd

def compute_revenue_trends(sessions: pd.DataFrame) -> pd.DataFrame:
    purchases = sessions[sessions["event"] == "purchase"].copy()
    purchases["date"] = pd.to_datetime(purchases["date"])
    purchases["week"] = purchases["date"].dt.to_period("W").astype(str)

    weekly = purchases.groupby("week").agg(
        gmv         = ("revenue", "sum"),
        orders      = ("revenue", "count"),
        unique_users= ("user_id", "nunique")
    ).reset_index()

    weekly["aov"] = (weekly["gmv"] / weekly["orders"]).round(2)  
    return weekly