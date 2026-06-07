
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


CHANNEL_CART_RATE = {
    "organic"     : 0.38,
    "paid_search" : 0.32,
    "social"      : 0.28,
    "email"       : 0.42,  
    "referral"    : 0.45,   
}
PURCHASE_RATE = 0.45   # of those who add to cart

def generate_ecommerce_data(n_users=2000, seed=42):
    np.random.seed(seed)

    start_date = datetime(2025, 1, 1)
    end_date   = datetime(2025, 12, 31)
    date_range = (end_date - start_date).days

    users = pd.DataFrame({
        "user_id"    : range(1, n_users + 1),
        "signup_date": [start_date + timedelta(days=int(d))
                        for d in np.random.randint(0, date_range // 2, n_users)],
        "channel"    : np.random.choice(
            ["organic", "paid_search", "social", "email", "referral"],
            n_users, p=[0.30, 0.25, 0.20, 0.15, 0.10]
        ),
        "country"    : np.random.choice(
            ["IN", "US", "UK", "DE", "SG"],
            n_users, p=[0.40, 0.25, 0.15, 0.10, 0.10]
        ),
    })

    events = []
    for _, user in users.iterrows():
        cart_rate = CHANNEL_CART_RATE[user["channel"]]
        n_sessions = np.random.randint(1, 9)

        for _ in range(n_sessions):
            days_offset  = np.random.randint(0, date_range)
            session_date = start_date + timedelta(days=int(days_offset))

            events.append({
                "user_id": user["user_id"],
                "date"   : session_date,
                "event"  : "visit"
            })
            if np.random.random() < cart_rate:
                events.append({
                    "user_id": user["user_id"],
                    "date"   : session_date,
                    "event"  : "add_to_cart"
                })
                if np.random.random() < PURCHASE_RATE:
                    revenue = round(np.random.lognormal(mean=4.0, sigma=0.8), 2)
                    events.append({
                        "user_id": user["user_id"],
                        "date"   : session_date,
                        "event"  : "purchase",
                        "revenue": revenue
                    })

        
        if np.random.random() < 0.40:
            for _ in range(np.random.randint(1, 4)):
                days_offset  = np.random.randint(273, date_range)
                session_date = start_date + timedelta(days=int(days_offset))
                events.append({
                    "user_id": user["user_id"],
                    "date"   : session_date,
                    "event"  : "visit"
                })
                if np.random.random() < cart_rate:
                    events.append({
                        "user_id": user["user_id"],
                        "date"   : session_date,
                        "event"  : "add_to_cart"
                    })
                    if np.random.random() < PURCHASE_RATE:
                        revenue = round(np.random.lognormal(mean=4.0, sigma=0.8), 2)
                        events.append({
                            "user_id": user["user_id"],
                            "date"   : session_date,
                            "event"  : "purchase",
                            "revenue": revenue
                        })

    sessions = pd.DataFrame(events)
    sessions["revenue"] = sessions.get("revenue", pd.Series(dtype=float))
    sessions["date"]    = pd.to_datetime(sessions["date"])
    return users, sessions


if __name__ == "__main__":
    users, sessions = generate_ecommerce_data()
    users.to_csv("data/users.csv", index=False)
    sessions.to_csv("data/sessions.csv", index=False)
    print(f"Users    : {len(users):,}")
    print(f"Sessions : {len(sessions):,}")
    print(f"Purchases: {(sessions['event'] == 'purchase').sum():,}")

   