"""
visualizations.py — Reusable Plotly chart builders for GharValue AI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Colour palette (consistent across charts)
# ---------------------------------------------------------------------------
PRIMARY = "#1e3a5f"
SECONDARY = "#2dd4bf"
ACCENT = "#f59e0b"
BACKGROUND = "#ffffff"
GRID_COLOR = "#e5e7eb"

SEQUENTIAL = px.colors.sequential.Teal
QUALITATIVE = px.colors.qualitative.Set2

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=BACKGROUND,
    plot_bgcolor=BACKGROUND,
    font=dict(family="Segoe UI, system-ui, sans-serif", size=12, color="#1a1a2e"),
    margin=dict(l=40, r=30, t=50, b=40),
    colorway=QUALITATIVE,
)


# ---------------------------------------------------------------------------
# Market Insights charts
# ---------------------------------------------------------------------------

def price_by_city(df: pd.DataFrame) -> go.Figure:
    """Median price per city, sorted descending."""
    if "city" not in df.columns or "price" not in df.columns:
        return _empty_figure("No city/price data available.")

    agg = (
        df.groupby("city")["price"]
        .agg(["median", "count"])
        .rename(columns={"median": "Median Price", "count": "Count"})
        .query("Count >= 5")
        .sort_values("Median Price", ascending=True)
        .tail(25)
        .reset_index()
    )
    if agg.empty:
        return _empty_figure("Not enough city data (min 5 records per city).")

    fig = px.bar(
        agg,
        x="Median Price",
        y="city",
        orientation="h",
        title="Median Property Price by City",
        labels={"Median Price": "Median Price (₹)", "city": "City"},
        color="Median Price",
        color_continuous_scale=SEQUENTIAL,
        hover_data={"Count": True},
        text=agg["Median Price"].apply(lambda v: _short_inr(v)),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_coloraxes(showscale=False)
    return fig


def price_per_sqft(df: pd.DataFrame) -> go.Figure:
    """Price per sq ft distribution by city/state."""
    if "price" not in df.columns or "area" not in df.columns:
        return _empty_figure("Price and area columns required.")

    df2 = df.copy()
    df2 = df2[(df2["area"] > 0) & (df2["price"] > 0)].copy()
    df2["price_sqft"] = df2["price"] / df2["area"]
    df2 = df2[df2["price_sqft"].between(
        df2["price_sqft"].quantile(0.02),
        df2["price_sqft"].quantile(0.98),
    )]

    group_col = "city" if "city" in df2.columns else None
    if group_col is None:
        return _empty_figure("City column not found.")

    agg = (
        df2.groupby(group_col)["price_sqft"]
        .agg(["median", "count"])
        .rename(columns={"median": "Median ₹/sqft", "count": "Count"})
        .query("Count >= 5")
        .sort_values("Median ₹/sqft", ascending=False)
        .head(20)
        .reset_index()
    )
    if agg.empty:
        return _empty_figure("Not enough data.")

    fig = px.bar(
        agg,
        x=group_col,
        y="Median ₹/sqft",
        title="Median Price per Sq Ft by City",
        color="Median ₹/sqft",
        color_continuous_scale=SEQUENTIAL,
        labels={"Median ₹/sqft": "₹ per Sq Ft"},
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_coloraxes(showscale=False)
    return fig


def bhk_distribution(df: pd.DataFrame) -> go.Figure:
    """BHK distribution as a pie chart."""
    if "bedrooms" not in df.columns:
        return _empty_figure("Bedrooms column not found.")

    counts = (
        df["bedrooms"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    counts.columns = ["BHK", "Count"]
    counts["Label"] = counts["BHK"].astype(str) + " BHK"

    fig = px.pie(
        counts,
        values="Count",
        names="Label",
        title="Property Distribution by BHK",
        color_discrete_sequence=QUALITATIVE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def area_vs_price(df: pd.DataFrame, color_col: str | None = None) -> go.Figure:
    """Scatter: built-up area vs price with optional colour grouping."""
    if "area" not in df.columns or "price" not in df.columns:
        return _empty_figure("Area and price columns required.")

    df2 = df.copy()
    df2 = df2[(df2["area"] > 0) & (df2["price"] > 0)]
    df2 = df2[
        df2["area"].between(df2["area"].quantile(0.01), df2["area"].quantile(0.99)) &
        df2["price"].between(df2["price"].quantile(0.01), df2["price"].quantile(0.99))
    ]

    # Downsample for performance
    if len(df2) > 2000:
        df2 = df2.sample(2000, random_state=42)

    if color_col and color_col in df2.columns:
        fig = px.scatter(
            df2,
            x="area",
            y="price",
            color=color_col,
            title="Built-up Area vs Price",
            labels={"area": "Area (sq ft)", "price": "Price (₹)"},
            opacity=0.6,
            color_discrete_sequence=QUALITATIVE,
        )
    else:
        fig = px.scatter(
            df2,
            x="area",
            y="price",
            title="Built-up Area vs Price",
            labels={"area": "Area (sq ft)", "price": "Price (₹)"},
            opacity=0.6,
            color_discrete_sequence=[PRIMARY],
        )

    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def price_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of property prices."""
    if "price" not in df.columns:
        return _empty_figure("Price column not found.")

    prices = df["price"].dropna()
    prices = prices[prices.between(prices.quantile(0.01), prices.quantile(0.99))]

    fig = px.histogram(
        prices,
        nbins=50,
        title="Property Price Distribution",
        labels={"value": "Price (₹)", "count": "Count"},
        color_discrete_sequence=[PRIMARY],
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def property_type_comparison(df: pd.DataFrame) -> go.Figure:
    """Box plot of price by property type."""
    if "property_type" not in df.columns or "price" not in df.columns:
        return _empty_figure("Property type / price data not available.")

    df2 = df.dropna(subset=["property_type", "price"]).copy()
    df2 = df2[df2["price"].between(
        df2["price"].quantile(0.01),
        df2["price"].quantile(0.99),
    )]

    counts = df2["property_type"].value_counts()
    valid_types = counts[counts >= 5].index
    df2 = df2[df2["property_type"].isin(valid_types)]

    if df2.empty:
        return _empty_figure("Not enough data per property type (min 5 records).")

    fig = px.box(
        df2,
        x="property_type",
        y="price",
        title="Price Distribution by Property Type",
        labels={"property_type": "Property Type", "price": "Price (₹)"},
        color="property_type",
        color_discrete_sequence=QUALITATIVE,
    )
    fig.update_layout(**LAYOUT_DEFAULTS, showlegend=False)
    return fig


def furnishing_price_comparison(df: pd.DataFrame) -> go.Figure:
    """Bar chart: average price by furnishing status."""
    if "furnishing" not in df.columns or "price" not in df.columns:
        return _empty_figure("Furnishing / price data not available.")

    agg = (
        df.groupby("furnishing")["price"]
        .median()
        .reset_index()
        .rename(columns={"price": "Median Price"})
    )
    fig = px.bar(
        agg,
        x="furnishing",
        y="Median Price",
        title="Median Price by Furnishing Status",
        labels={"furnishing": "Furnishing", "Median Price": "Median Price (₹)"},
        color="furnishing",
        color_discrete_sequence=QUALITATIVE,
    )
    fig.update_layout(**LAYOUT_DEFAULTS, showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Model Evaluation charts
# ---------------------------------------------------------------------------

def actual_vs_predicted(y_true, y_pred, model_name: str = "") -> go.Figure:
    """Scatter: actual vs predicted with perfect-fit reference line."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    lim_min = min(y_true.min(), y_pred.min()) * 0.95
    lim_max = max(y_true.max(), y_pred.max()) * 1.05

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        marker=dict(color=PRIMARY, opacity=0.5, size=5),
        name="Predictions",
    ))
    fig.add_trace(go.Scatter(
        x=[lim_min, lim_max],
        y=[lim_min, lim_max],
        mode="lines",
        line=dict(color=ACCENT, dash="dash", width=2),
        name="Perfect Fit",
    ))
    fig.update_layout(
        title=f"Actual vs Predicted — {model_name}",
        xaxis_title="Actual Price (₹)",
        yaxis_title="Predicted Price (₹)",
        **LAYOUT_DEFAULTS,
    )
    return fig


def residual_histogram(y_true, y_pred, model_name: str = "") -> go.Figure:
    """Histogram of residuals."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig = px.histogram(
        residuals,
        nbins=50,
        title=f"Residual Distribution — {model_name}",
        labels={"value": "Residual (₹)", "count": "Count"},
        color_discrete_sequence=[SECONDARY],
    )
    fig.add_vline(x=0, line_dash="dash", line_color=ACCENT)
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def feature_importance_chart(importance_df: pd.DataFrame, model_name: str = "") -> go.Figure:
    """Horizontal bar chart of feature importances."""
    top = importance_df.head(15)
    fig = px.bar(
        top,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Feature Importance — {model_name}",
        color="Importance",
        color_continuous_scale=SEQUENTIAL,
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_coloraxes(showscale=False)
    return fig


def model_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing MAE and RMSE across models."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["MAE (₹) — Lower is Better", "R² Score — Higher is Better"],
    )

    models = comparison_df["Model"].tolist()
    colours = QUALITATIVE[:len(models)]

    fig.add_trace(
        go.Bar(x=models, y=comparison_df["MAE (₹)"].tolist(),
               marker_color=colours, showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=models, y=comparison_df["R² Score"].tolist(),
               marker_color=colours, showlegend=False),
        row=1, col=2,
    )
    fig.update_layout(title="Model Comparison", **LAYOUT_DEFAULTS)
    return fig


# ---------------------------------------------------------------------------
# 3D Analytics
# ---------------------------------------------------------------------------

def scatter_3d(df: pd.DataFrame) -> go.Figure:
    """
    3D scatter: Area × Bedrooms × Price, coloured by property_type or city.
    """
    required = {"area", "bedrooms", "price"}
    if not required.issubset(df.columns):
        return _empty_figure("Required columns: area, bedrooms, price.")

    df2 = df.dropna(subset=list(required)).copy()
    df2 = df2[(df2["area"] > 0) & (df2["price"] > 0) & (df2["bedrooms"] > 0)]
    df2 = df2[
        df2["area"].between(df2["area"].quantile(0.01), df2["area"].quantile(0.99)) &
        df2["price"].between(df2["price"].quantile(0.01), df2["price"].quantile(0.99))
    ]

    if len(df2) < 10:
        return _empty_figure(f"Only {len(df2)} valid records; need at least 10.")

    # Downsample for performance
    if len(df2) > 3000:
        df2 = df2.sample(3000, random_state=42)

    color_col = None
    for candidate in ["property_type", "state", "city"]:
        if candidate in df2.columns and df2[candidate].nunique() <= 15:
            color_col = candidate
            break

    hover = ["city"] if "city" in df2.columns else []

    fig = px.scatter_3d(
        df2,
        x="area",
        y="bedrooms",
        z="price",
        color=color_col,
        opacity=0.65,
        size_max=8,
        title="3D Property Analytics: Area × Bedrooms × Price",
        labels={
            "area": "Built-up Area (sq ft)",
            "bedrooms": "Bedrooms (BHK)",
            "price": "Price (₹)",
        },
        hover_data=hover,
        color_discrete_sequence=QUALITATIVE,
    )
    fig.update_layout(
        paper_bgcolor=BACKGROUND,
        font=dict(family="Segoe UI, system-ui, sans-serif", size=11),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(title=color_col.replace("_", " ").title() if color_col else ""),
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#57606a"),
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _short_inr(v: float) -> str:
    if v >= 1e7:
        return f"₹{v/1e7:.1f}Cr"
    if v >= 1e5:
        return f"₹{v/1e5:.1f}L"
    return f"₹{v:,.0f}"
