import pandas as pd
from churn import compute_rfm,train_churn_model


users = pd.read_csv("data/users.csv")
sessions = pd.read_csv("data/sessions.csv")


rfm = compute_rfm(users, sessions)

print("Segments:")
print(rfm["segment"].value_counts().to_string())

print(f"\nChurn rate: {rfm['churned'].mean():.1%}")


model, imp, report = train_churn_model(rfm)

print("\nFeature importance:")
print(imp.to_string(index=False))

print(f"\nModel accuracy: {report['accuracy']:.1%}")