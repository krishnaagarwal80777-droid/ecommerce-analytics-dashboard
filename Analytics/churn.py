
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


OBS_END   = pd.Timestamp("2025-09-30")   
PRED_END  = pd.Timestamp("2025-12-31")   

def compute_rfm(users: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    purchases = sessions[sessions["event"] == "purchase"].copy()
    purchases["date"] = pd.to_datetime(purchases["date"])

  
    obs = purchases[purchases["date"] <= OBS_END]

    rfm = obs.groupby("user_id").agg(
        last_order_obs = ("date",    "max"),
        frequency      = ("date",    "count"),
        monetary       = ("revenue", "sum"),
        avg_basket     = ("revenue", "mean"),
    ).reset_index()

    rfm["recency_obs"] = (OBS_END - rfm["last_order_obs"]).dt.days

   
    future = purchases[
        (purchases["date"] > OBS_END) &
        (purchases["date"] <= PRED_END)
    ]
    active_in_future = set(future["user_id"].unique())
    rfm["churned"] = (~rfm["user_id"].isin(active_in_future)).astype(int)

    
    rfm["r_score"] = pd.qcut(rfm["recency_obs"],
                              q=3, labels=[3,2,1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"),
                              q=3, labels=[1,2,3]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"),
                              q=3, labels=[1,2,3]).astype(int)

    def segment(row):
        if row["r_score"] == 3 and row["f_score"] >= 2: return "Champion"
        if row["r_score"] >= 2 and row["f_score"] >= 2: return "Loyal"
        if row["r_score"] == 3:                          return "Recent"
        if row["r_score"] == 2:                          return "At Risk"
        return "Lost"

    rfm["segment"] = rfm.apply(segment, axis=1)
    return rfm


def train_churn_model(rfm: pd.DataFrame):
    features = ["recency_obs", "frequency", "monetary", "avg_basket", "r_score", "f_score", "m_score"]
    X = rfm[features]
    y = rfm["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    report    = classification_report(y_test, model.predict(X_test), output_dict=True)
    importance = pd.DataFrame({
        "feature"   : features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return model, importance, report