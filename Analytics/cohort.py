
import pandas as pd
import numpy as np

def compute_cohort_retention(users: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
   
    purchases = sessions[sessions["event"] == "purchase"].copy()
    purchases["date"] = pd.to_datetime(purchases["date"])
    users["signup_date"] = pd.to_datetime(users["signup_date"])

    
    users["cohort_week"] = users["signup_date"].dt.to_period("W")

    merged = purchases.merge(users[["user_id", "cohort_week", "signup_date"]], on="user_id")
    merged["order_week"]   = merged["date"].dt.to_period("W")
    merged["weeks_since"]  = (
        merged["order_week"].apply(lambda x: x.start_time) -
        merged["cohort_week"].apply(lambda x: x.start_time)
    ).dt.days // 7

    
    merged = merged[merged["weeks_since"].between(0, 12)]

    cohort_sizes = users.groupby("cohort_week")["user_id"].nunique()
    cohort_data  = merged.groupby(["cohort_week", "weeks_since"])["user_id"].nunique().reset_index()
    cohort_data["cohort_size"] = cohort_data["cohort_week"].map(cohort_sizes)
    cohort_data["retention"]   = (cohort_data["user_id"] / cohort_data["cohort_size"] * 100).round(1)

    matrix = cohort_data.pivot(index="cohort_week", columns="weeks_since", values="retention")
    matrix.index = matrix.index.astype(str)
    return matrix.fillna(0)