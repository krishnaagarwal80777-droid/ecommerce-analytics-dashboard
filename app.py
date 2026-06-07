
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from Analytics.funnel  import compute_funnel, compute_funnel_by_channel
from Analytics.cohort  import compute_cohort_retention
from Analytics.churn   import compute_rfm, train_churn_model
from Analytics.revenue import compute_revenue_trends

# ── Load data ──────────────────────────────────────────────────────────────
users    = pd.read_csv("data/users.csv")
sessions = pd.read_csv("data/sessions.csv")

funnel        = compute_funnel(sessions)
funnel_ch     = compute_funnel_by_channel(users, sessions)
cohort_matrix = compute_cohort_retention(users, sessions)
rfm           = compute_rfm(users, sessions)
_, feat_imp, report = train_churn_model(rfm)
revenue       = compute_revenue_trends(sessions)

# ── KPI cards ──────────────────────────────────────────────────────────────
total_users   = len(users)
total_revenue = sessions[sessions["event"] == "purchase"]["revenue"].sum()
churn_rate    = rfm["churned"].mean()
conversion    = funnel["counts"][2] / funnel["counts"][0] * 100

def kpi_card(title, value, color):
    return dbc.Card(dbc.CardBody([
        html.P(title, className="text-muted mb-1", style={"fontSize": "0.85rem"}),
        html.H4(value, style={"color": color, "fontWeight": "700"})
    ]), className="shadow-sm rounded-3")

# ── Figures ─────────────────────────────────────────────────────────────────
def fig_funnel():
    colors = ["#5B8DEF", "#F4845F", "#2ECC71"]
    fig = go.Figure(go.Funnel(
        y      = funnel["stages"],
        x      = funnel["counts"],
        textinfo="value+percent previous",
        marker = {"color": colors}
    ))
    fig.update_layout(title="Conversion Funnel (unique users)",
                      template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20))
    return fig

def fig_funnel_by_channel():
    fig = px.bar(funnel_ch, x="channel", y="conversion_rate",
                 color="conversion_rate", color_continuous_scale="Blues",
                 labels={"conversion_rate": "Conversion rate (%)", "channel": "Channel"},
                 title="Conversion rate by acquisition channel")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      margin=dict(t=50, b=20))
    return fig

def fig_cohort():
    fig = go.Figure(go.Heatmap(
        z          = cohort_matrix.values,
        x          = [f"Week {c}" for c in cohort_matrix.columns],
        y          = cohort_matrix.index.astype(str),
        colorscale = "Blues",
        text       = cohort_matrix.values.round(1),
        texttemplate="%{text}%",
    ))
    fig.update_layout(title="Weekly cohort retention (%)",
                      template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20))
    return fig

def fig_churn_segments():
    seg_counts = rfm["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]
    colors_map = {
        "Champion": "#2ECC71", "Loyal": "#5B8DEF",
        "Recent"  : "#F1C40F", "At Risk": "#E67E22", "Lost": "#E74C3C"
    }
    fig = px.pie(seg_counts, names="segment", values="count",
                 color="segment", color_discrete_map=colors_map,
                 title="RFM customer segments", hole=0.4)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=50, b=20))
    return fig

def fig_feature_importance():
    fig = px.bar(feat_imp, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale="Blues",
                 title="Churn model — feature importance")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", yaxis={"categoryorder":"total ascending"},
                      showlegend=False, margin=dict(t=50, b=20))
    return fig

def fig_revenue():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=revenue["week"], y=revenue["gmv"],
                             mode="lines+markers", name="GMV",
                             line=dict(color="#5B8DEF", width=2)))
    fig.add_trace(go.Bar(x=revenue["week"], y=revenue["orders"],
                         name="Orders", marker_color="#F4845F", opacity=0.5,
                         yaxis="y2"))
    fig.update_layout(
        title="Weekly revenue (GMV) & order volume",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis =dict(title="GMV ($)"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=60, b=20)
    )
    return fig

# ── Layout ──────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "E-commerce Analytics"

app.layout = dbc.Container([

    html.H2("📊 E-commerce Product Analytics Dashboard",
            className="text-center my-4 fw-bold"),

    # KPI row
    dbc.Row([
        dbc.Col(kpi_card("Total Users",    f"{total_users:,}",          "#5B8DEF"), md=3),
        dbc.Col(kpi_card("Total Revenue",  f"${total_revenue:,.0f}",    "#2ECC71"), md=3),
        dbc.Col(kpi_card("Churn Rate",     f"{churn_rate:.1%}",         "#E74C3C"), md=3),
        dbc.Col(kpi_card("Conversion Rate",f"{conversion:.1f}%",        "#F1C40F"), md=3),
    ], className="mb-4 g-3"),

    # Funnel row
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_funnel()),            md=6),
        dbc.Col(dcc.Graph(figure=fig_funnel_by_channel()), md=6),
    ], className="mb-4"),

    # Revenue
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_revenue()), md=12),
    ], className="mb-4"),

    # Cohort
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_cohort()), md=12),
    ], className="mb-4"),

    # Churn
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_churn_segments()),    md=6),
        dbc.Col(dcc.Graph(figure=fig_feature_importance()),md=6),
    ], className="mb-4"),

    html.Footer("Built by Krishna Agarwal · E-commerce Analytics Dashboard",
                className="text-center text-muted py-3")

], fluid=True)

if __name__ == "__main__":
    app.run(debug=True)