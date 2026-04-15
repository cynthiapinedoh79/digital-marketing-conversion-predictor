import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


@st.cache_data
def load_data():
    path = "inputs/datasets/cleaned/marketing_campaign_cleaned.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    df = pd.read_csv(
        "inputs/datasets/raw/digital_marketing_campaign_dataset.csv")
    return df.drop(columns=[
        'CustomerID', 'AdvertisingPlatform',
        'AdvertisingTool', 'ConversionRate'], errors='ignore')


def page_data_analysis_body():
    st.title("Customer Behaviour Analysis")
    st.subheader("Business Requirement 1 — What variables correlate "
                 "with conversion?")

    st.markdown("""
    This page presents the findings from the exploratory data analysis
    conducted on the Digital Marketing Campaign dataset.
    """)

    df = load_data()

    # ── Section 1: Correlation Heatmap ──────────────────────────────
    st.markdown("---")
    st.subheader("1. Correlation Heatmap")
    st.markdown("""
    Spearman rank correlations between all numeric features and
    the Conversion target. Higher absolute values indicate stronger
    association.
    """)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr_matrix = df[numeric_cols].corr(method='spearman')

    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        text_auto='.2f',
        title='Spearman Correlation Matrix'
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.success("""
    **Interpretation:** The strongest correlations with Conversion are
    TimeOnSite, EmailClicks, EmailOpens, AdSpend and ClickThroughRate.
    No single variable dominates — the ML model needs to combine all
    features to achieve meaningful predictive performance.
    """)

    # ── Section 2: Box Plots ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("2. Feature Distribution by Conversion Outcome")

    top_features = [
        'TimeOnSite', 'EmailClicks', 'EmailOpens',
        'PreviousPurchases', 'LoyaltyPoints']
    selected = st.selectbox("Select a feature to explore:", top_features)

    df_plot = df.copy()
    df_plot['Outcome'] = df_plot['Conversion'].map(
        {0: 'Not Converted', 1: 'Converted'})

    fig2 = px.box(
        df_plot, x='Outcome', y=selected,
        color='Outcome',
        color_discrete_map={
            'Not Converted': '#d62728', 'Converted': '#2ca02c'},
        title=f'{selected} by Conversion Outcome',
        points='outliers'
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    mean_conv = df.loc[df['Conversion'] == 1, selected].mean()
    mean_not = df.loc[df['Conversion'] == 0, selected].mean()
    diff_pct = ((mean_conv - mean_not) / mean_not) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Converted — Mean", f"{mean_conv:.2f}")
    col2.metric("Not Converted — Mean", f"{mean_not:.2f}")
    col3.metric("Difference", f"{diff_pct:+.1f}%")

    st.success(f"""
    **Interpretation:** Converted leads show a **{diff_pct:+.1f}%**
    difference in {selected} compared to non-converted leads.
    """)

    # ── Section 3: Scatter Plot ──────────────────────────────────────
    st.markdown("---")
    st.subheader("3. Time on Site vs Pages Per Visit")

    fig3 = px.scatter(
        df_plot, x='TimeOnSite', y='PagesPerVisit',
        color='Outcome',
        color_discrete_map={
            'Not Converted': '#d62728', 'Converted': '#2ca02c'},
        opacity=0.4,
        title='Time on Site vs Pages Per Visit by Conversion Outcome',
        labels={
            'TimeOnSite': 'Time on Site (min)',
            'PagesPerVisit': 'Pages Per Visit'}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.success("""
    **Interpretation:** The scatter plot shows that converted and
    non-converted leads overlap across both dimensions, confirming
    that no single pair of features perfectly separates the two classes.
    The dominance of converted leads (green) reflects the 87.65% class
    imbalance in the dataset.
    """)

    # ── Section 4: Bar Charts ────────────────────────────────────────
    st.markdown("---")
    st.subheader("4. Conversion Rate by Channel and Campaign Type")

    col_a, col_b = st.columns(2)

    with col_a:
        conv_ch = df.groupby('CampaignChannel')[
            'Conversion'].mean().reset_index()
        conv_ch = conv_ch.sort_values('Conversion', ascending=True)
        fig4 = px.bar(
            conv_ch, x='Conversion', y='CampaignChannel',
            orientation='h', color='Conversion',
            color_continuous_scale='Blues',
            title='Conversion Rate by Channel',
            text=conv_ch['Conversion'].map('{:.1%}'.format)
        )
        fig4.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        conv_type = df.groupby('CampaignType')[
            'Conversion'].mean().reset_index()
        conv_type = conv_type.sort_values('Conversion', ascending=True)
        fig5 = px.bar(
            conv_type, x='Conversion', y='CampaignType',
            orientation='h', color='Conversion',
            color_continuous_scale='Greens',
            title='Conversion Rate by Campaign Type',
            text=conv_type['Conversion'].map('{:.1%}'.format)
        )
        fig5.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)

    st.success("""
    **Interpretation:** Conversion rates vary by both channel and
    campaign type. Campaigns of type Conversion achieve 93.4% —
    significantly higher than Awareness (85.6%). This confirms that
    campaign intent is a stronger predictor than channel choice.
    """)

    # ── Summary ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Key Findings — BR1 Summary")
    st.info("""
    The top 5 variables most associated with Conversion are:
    **CampaignType**, **EmailClicks**, **PreviousPurchases**,
    **TimeOnSite** and **AdSpend**.

    Campaign channel does not show a statistically significant effect
    on conversion rate (chi-square test: p = 0.59).
    """)
