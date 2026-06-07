# E-Commerce Product Analytics Dashboard

An end-to-end analytics dashboard built with **Python, Dash, Plotly, Pandas, and Scikit-Learn** to analyze customer behavior, retention, revenue, and churn in an e-commerce environment.

##  Features

* **Conversion Funnel Analysis** – Track user drop-off from visit to purchase with acquisition channel breakdowns.
* **Cohort Retention** – Weekly retention heatmaps to measure user engagement over time.
* **RFM Segmentation** – Classify customers into segments such as Champions, Loyal, At Risk, and Lost.
* **Churn Prediction** – Random Forest model to identify customers likely to churn, including feature importance analysis (Gradient Boosting coming soon)
* **Revenue Analytics** – Monitor weekly GMV, order volume, and Average Order Value (AOV).

##  Tech Stack

**Python • Dash • Plotly • Pandas • NumPy • Scikit-Learn**

##  Project Structure

```text
├── data/
│   └── generate_data.py
├── analytics/
│   ├── funnel.py
│   ├── cohort.py
│   ├── churn.py
│   └── revenue.py
├── app.py
└── requirements.txt
```

##  Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Generate Sample Data

```bash
python data/generate_data.py
```

### Launch Dashboard

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8050
```

## Key Insights

* Identify conversion bottlenecks in the purchase journey
* Measure user retention and engagement trends
* Discover high-value customer segments
* Predict churn risk using machine learning
* Track revenue and growth metrics

## Future Enhancements

* Gradient Boosting / XGBoost churn models
* Customer Lifetime Value (CLV) prediction
* Marketing attribution analysis
* Cloud deployment and real-time data pipelines



If you found this project useful, consider starring the repository.
