
import pandas as pd

def compute_funnel(sessions: pd.DataFrame) -> dict:
    
    visitors  = sessions[sessions["event"] == "add_to_cart"]["user_id"].nunique()
    add_cart  = sessions[sessions["event"] == "add_to_cart"]["user_id"].nunique()
    purchases = sessions[sessions["event"] == "purchase"]["user_id"].nunique()
    visits    = sessions[sessions["event"] == "visit"]["user_id"].nunique()

    return {
        "stages": ["Visited", "Added to Cart", "Purchased"],
        "counts": [visits, add_cart, purchases],
        "rates" : [
            100.0,
            round(add_cart  / visits   * 100, 1),
            round(purchases / add_cart * 100, 1),
        ]
    }

def compute_funnel_by_channel(users: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
   
    merged = sessions.merge(users[["user_id", "channel"]], on="user_id")
    result = []
    for channel, grp in merged.groupby("channel"):
        visitors  = grp[grp["event"] == "visit"]["user_id"].nunique()
        purchasers = grp[grp["event"] == "purchase"]["user_id"].nunique()
        result.append({
            "channel"        : channel,
            "visitors"       : visitors,
            "purchasers"     : purchasers,
            "conversion_rate": round(purchasers / visitors * 100, 2) if visitors else 0
        })
    return pd.DataFrame(result).sort_values("conversion_rate", ascending=False)