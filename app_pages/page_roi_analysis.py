import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os


@st.cache_data
def load_kpis():
    cleaned = "inputs/datasets/cleaned/marketing_kpis_cleaned.csv"
    raw = "inputs/datasets/raw/Marketing.csv"

    if os.path.exists(cleaned):
        df = pd.read_csv(cleaned, parse_dates=["c_date"])
        if "month" not in df.columns and "c_date" in df.columns:
            df["month"] = pd.to_datetime(df["c_date"]).dt.to_period("M").astype(str)
        if "roi" not in df.columns and {"revenue", "mark_spent"}.issubset(df.columns):
            df["roi"] = (
                (df["revenue"] - df["mark_spent"]) /
                df["mark_spent"].replace(0, np.nan)
            ).round(2)
        df["roi"] = df["roi"].fillna(0)
        return df

    if not os.path.exists(raw):
        return None

    df = pd.read_csv(raw)
    df["campaign_name"] = df["campaign_name"].astype(str).str.lower().str.strip()
    df["category"] = df["category"].astype(str).str.lower().str.strip()
    df["c_date"] = pd.to_datetime(df["c_date"])
    df["month"] = df["c_date"].dt.to_period("M").astype(str)
    df["roi"] = (
        (df["revenue"] - df["mark_spent"]) /
        df["mark_spent"].replace(0, np.nan)
    ).round(2)
    df["roi"] = df["roi"].fillna(0)

    return df


def page_roi_analysis_body():
    st.title("Campaign ROI Analysis")
    st.subheader("Business Requirement 3 — Which campaigns deliver the best return?")

    st.markdown("""
    This page analyses marketing spend efficiency across campaign
    categories using the Digital Marketing KPIs dataset.

    **ROI formula:** `(Revenue − Marketing Spend) / Marketing Spend`
    """)

    df = load_kpis()

    if df is None:
        st.error("Dataset not found. Please ensure `Marketing.csv` is available.")
        return

    # ── Summary KPIs ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Overall Campaign Performance")

    total_spend = df["mark_spent"].sum()
    total_revenue = df["revenue"].sum()
    overall_roi = (total_revenue - total_spend) / total_spend if total_spend != 0 else 0
    total_orders = df["orders"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spend", f"${total_spend:,.0f}")
    col2.metric("Total Revenue", f"${total_revenue:,.0f}")
    col3.metric("Overall ROI", f"{overall_roi:.1%}")
    col4.metric("Total Orders", f"{total_orders:,}")

    # ── Scatter: Spend vs Revenue ───────────────────────────────────
    st.markdown("---")
    st.subheader("1. Marketing Spend vs Revenue by Category")
    st.markdown("""
    Each point represents one campaign-day. Points above the break-even line
    represent profitable campaigns, while those below indicate inefficient spend.
    """)

    fig1 = px.scatter(
        df,
        x="mark_spent",
        y="revenue",
        color="category",
        hover_data=["campaign_name", "c_date", "orders"],
        title="Marketing Spend vs Revenue by Category",
        labels={
            "mark_spent": "Marketing Spend (USD)",
            "revenue": "Revenue (USD)",
            "category": "Category",
        },
        opacity=0.7,
    )

    max_val = max(df["mark_spent"].max(), df["revenue"].max())
    fig1.add_scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        line=dict(color="gray", dash="dash"),
        name="Break-even",
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.success("""
    **Interpretation:**  
    Points above the break-even line represent profitable campaigns.
    The scatter plot reveals which campaign categories consistently generate
    higher revenue relative to marketing spend.
    """)

    # ── Bar: ROI by Category ────────────────────────────────────────
    st.markdown("---")
    st.subheader("2. Average ROI by Campaign Category")

    roi_by_cat = df.groupby("category")["roi"].mean().reset_index()
    roi_by_cat.columns = ["Category", "Average ROI"]
    roi_by_cat = roi_by_cat.sort_values("Average ROI", ascending=True)

    fig2 = px.bar(
        roi_by_cat,
        x="Average ROI",
        y="Category",
        orientation="h",
        color="Average ROI",
        color_continuous_scale="RdYlGn",
        title="Average ROI by Campaign Category",
        text=roi_by_cat["Average ROI"].map("{:.1%}".format),
    )
    fig2.update_layout(coloraxis_showscale=False)

    st.plotly_chart(fig2, use_container_width=True)

    best_cat = roi_by_cat.loc[roi_by_cat["Average ROI"].idxmax(), "Category"]
    best_roi = roi_by_cat["Average ROI"].max()

    st.success(f"""
    **Interpretation:**  
    `{best_cat.title()}` delivers the highest average ROI of **{best_roi:.1%}**.
    This suggests that ConvertIQ should prioritise budget allocation toward this category.
    """)

    # ── Line: Revenue Trend ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("3. Monthly Revenue Trend by Category")

    categories = sorted(df["category"].dropna().unique().tolist())
    selected_cats = st.multiselect(
        "Filter by campaign category:",
        options=categories,
        default=categories,
    )

    monthly = df.groupby(["month", "category"])["revenue"].sum().reset_index()
    monthly_filtered = monthly[monthly["category"].isin(selected_cats)]

    fig3 = px.line(
        monthly_filtered,
        x="month",
        y="revenue",
        color="category",
        markers=True,
        title="Monthly Revenue by Campaign Category",
        labels={
            "month": "Month",
            "revenue": "Revenue (USD)",
            "category": "Category",
        },
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.success("""
    **Interpretation:**  
    The line chart reveals seasonality and growth trends by category.
    Identifying months with revenue dips can help ConvertIQ optimise
    budget timing and campaign planning.
    """)

    # ── Conclusions ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Key Findings — BR3 Summary")

    st.info(f"""
    **Campaign ROI Analysis Conclusions:**

    1. Highest-performing category: **{best_cat.title()}** with an average ROI of **{best_roi:.1%}**
    2. Marketing effectiveness varies significantly across campaign types
    3. Seasonal revenue patterns suggest opportunities for optimised budget allocation
    4. **Recommendation:** Increase investment in {best_cat.title()} campaigns while reviewing lower-performing categories to improve efficiency

    From a business perspective, ROI-driven optimisation can significantly improve
    marketing profitability and resource allocation.
    """)
