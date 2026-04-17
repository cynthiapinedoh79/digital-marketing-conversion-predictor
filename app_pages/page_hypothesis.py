import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from scipy.stats import pointbiserialr, chi2_contingency


@st.cache_data
def load_data():
    cleaned_path = "inputs/datasets/cleaned/marketing_campaign_cleaned.csv"
    raw_path = "inputs/datasets/raw/digital_marketing_campaign_dataset.csv"

    if os.path.exists(cleaned_path):
        return pd.read_csv(cleaned_path)

    if not os.path.exists(raw_path):
        return None

    df = pd.read_csv(raw_path)
    return df.drop(
        columns=[
            "CustomerID",
            "AdvertisingPlatform",
            "AdvertisingTool",
            "ConversionRate",
        ],
        errors="ignore",
    )


def page_hypothesis_body():
    st.title("Project Hypotheses")
    st.subheader("What did we assume, and were we right?")

    st.markdown("""
    Before conducting any analysis, three hypotheses were formulated
    based on domain knowledge of digital marketing behaviour.
    Each hypothesis was validated using statistical methods.
    """)

    df = load_data()

    if df is None:
        st.error("Dataset not found. Please ensure the project dataset is available.")
        return

    required_cols = [
        "Conversion",
        "TimeOnSite",
        "PagesPerVisit",
        "CampaignChannel",
        "AdSpend",
        "EmailClicks",
        "PreviousPurchases",
        "LoyaltyPoints",
        "WebsiteVisits",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return

    # Ensure target is numeric/binary
    df = df.copy()
    df["Conversion"] = pd.to_numeric(df["Conversion"], errors="coerce")
    df = df.dropna(subset=["Conversion"])
    df["Conversion"] = df["Conversion"].astype(int)

    # ── HYPOTHESIS 1 ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hypothesis 1 — Engagement Depth Predicts Conversion")

    with st.expander("Statement and validation method", expanded=True):
        st.markdown("""
        **Statement:** Leads that spend more time on site and view more
        pages per visit are significantly more likely to convert.

        **Validation method:** Point-biserial correlation between
        TimeOnSite / PagesPerVisit and the binary Conversion target.
        Significance threshold: alpha = 0.05.
        """)

    r_time, p_time = pointbiserialr(df["Conversion"], df["TimeOnSite"])
    r_pages, p_pages = pointbiserialr(df["Conversion"], df["PagesPerVisit"])
    mean_time_conv = df.loc[df["Conversion"] == 1, "TimeOnSite"].mean()
    mean_time_not = df.loc[df["Conversion"] == 0, "TimeOnSite"].mean()

    if mean_time_not != 0:
        diff_pct = ((mean_time_conv - mean_time_not) / mean_time_not) * 100
    else:
        diff_pct = 0

    col1, col2 = st.columns(2)

    with col1:
        results_h1 = pd.DataFrame({
            "Feature": ["TimeOnSite", "PagesPerVisit"],
            "Correlation r": [f"{r_time:.4f}", f"{r_pages:.4f}"],
            "p-value": [f"{p_time:.2e}", f"{p_pages:.2e}"],
            "Significant": [
                "Yes" if p_time < 0.05 else "No",
                "Yes" if p_pages < 0.05 else "No",
            ],
        })
        st.dataframe(results_h1, use_container_width=True, hide_index=True)
        st.metric(
            "Avg TimeOnSite — Converted",
            f"{mean_time_conv:.2f} min",
            delta=f"{diff_pct:+.1f}% vs Not Converted",
        )

    with col2:
        df_plot = df.copy()
        df_plot["Outcome"] = df_plot["Conversion"].map(
            {0: "Not Converted", 1: "Converted"}
        )
        fig = px.violin(
            df_plot,
            x="Outcome",
            y="TimeOnSite",
            color="Outcome",
            box=True,
            points=False,
            color_discrete_map={
                "Not Converted": "#d62728",
                "Converted": "#2ca02c",
            },
            title="TimeOnSite by Conversion",
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    if p_time < 0.05 and p_pages < 0.05:
        st.success(f"""
        ✅ **Hypothesis 1 — CONFIRMED**

        TimeOnSite (r={r_time:.3f}, p={p_time:.2e}) and PagesPerVisit
        (r={r_pages:.3f}, p={p_pages:.2e}) show statistically significant
        positive correlations with conversion. Converted leads spend
        **{diff_pct:.1f}% more time on site**.

        **Action:** Invest in multi-step engagement content to increase
        dwell time and pages-per-visit across campaigns.
        """)
    else:
        st.warning("""
        ❌ **Hypothesis 1 — NOT CONFIRMED**

        The expected engagement variables were not both statistically significant.
        """)

    # ── HYPOTHESIS 2 ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hypothesis 2 — Campaign Channel Affects Conversion")

    with st.expander("Statement and validation method", expanded=True):
        st.markdown("""
        **Statement:** The campaign channel (Email, SEO, PPC,
        Social Media, Referral) has a statistically significant
        effect on conversion rate.

        **Validation method:** Chi-square test of independence
        between CampaignChannel and Conversion.
        Significance threshold: alpha = 0.05.
        """)

    contingency = pd.crosstab(df["CampaignChannel"], df["Conversion"])
    chi2, p_chi, dof, _ = chi2_contingency(contingency)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | Chi-square | {chi2:.3f} |
        | p-value | {p_chi:.4e} |
        | Degrees of freedom | {dof} |
        | Significant | {'Yes' if p_chi < 0.05 else 'No'} |
        """)
        conv_ch = df.groupby("CampaignChannel")["Conversion"].agg(["mean", "count"]).reset_index()
        conv_ch.columns = ["Channel", "Conv. Rate", "N"]
        conv_ch["Conv. Rate"] = conv_ch["Conv. Rate"].map("{:.2%}".format)
        st.dataframe(conv_ch, use_container_width=True, hide_index=True)

    with col4:
        conv_rates = df.groupby("CampaignChannel")["Conversion"].mean().reset_index()
        fig2 = px.bar(
            conv_rates,
            x="CampaignChannel",
            y="Conversion",
            color="Conversion",
            color_continuous_scale="RdYlGn",
            title="Conversion Rate by Channel",
            labels={
                "Conversion": "Conv. Rate",
                "CampaignChannel": "Channel",
            },
            text=conv_rates["Conversion"].map("{:.1%}".format),
        )
        fig2.update_layout(coloraxis_showscale=False, showlegend=False, height=300)
        st.plotly_chart(fig2, use_container_width=True)

    if p_chi < 0.05:
        st.success(f"""
        ✅ **Hypothesis 2 — CONFIRMED**

        Chi-square test: chi2={chi2:.2f}, p={p_chi:.2e} —
        statistically significant association confirmed.
        """)
    else:
        st.warning(f"""
        ❌ **Hypothesis 2 — REJECTED**

        Chi-square test: chi2={chi2:.2f}, p={p_chi:.2e} —
        no statistically significant association between
        CampaignChannel and conversion rate (p > 0.05).

        While small visual differences exist between channels,
        they are not statistically significant.
        """)

    # ── HYPOTHESIS 3 ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hypothesis 3 — Ad Spend Alone is a Weak Predictor")

    with st.expander("Statement and validation method", expanded=True):
        st.markdown("""
        **Statement:** AdSpend alone is not a reliable predictor
        of conversion — its predictive power is significantly lower
        than behavioural engagement features.

        **Validation method:** Point-biserial correlation comparison
        between AdSpend and engagement features vs Conversion.
        """)

    features_compare = [
        "AdSpend",
        "EmailClicks",
        "PreviousPurchases",
        "LoyaltyPoints",
        "TimeOnSite",
        "WebsiteVisits",
    ]

    corr_results = []
    for feat in features_compare:
        r, p = pointbiserialr(df["Conversion"], df[feat])
        corr_results.append({
            "Feature": feat,
            "|Correlation|": round(abs(r), 4),
            "p-value": f"{p:.2e}",
            "Significant": "Yes" if p < 0.05 else "No",
        })

    corr_df = pd.DataFrame(corr_results).sort_values("|Correlation|", ascending=False)

    col5, col6 = st.columns(2)
    with col5:
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

    with col6:
        fig3 = px.bar(
            corr_df,
            x="|Correlation|",
            y="Feature",
            orientation="h",
            color="|Correlation|",
            color_continuous_scale="Blues",
            title="Absolute Correlation with Conversion",
        )
        fig3.update_layout(coloraxis_showscale=False, height=300)
        st.plotly_chart(fig3, use_container_width=True)

    adspend_corr = corr_df.loc[
        corr_df["Feature"] == "AdSpend", "|Correlation|"
    ].values[0]
    max_other = corr_df.loc[
        corr_df["Feature"] != "AdSpend", "|Correlation|"
    ].max()
    best_feat = corr_df.loc[
        corr_df["Feature"] != "AdSpend"
    ].iloc[0]["Feature"]

    if adspend_corr < max_other:
        st.success(f"""
        ✅ **Hypothesis 3 — CONFIRMED**

        AdSpend correlation: {adspend_corr:.4f} vs
        {best_feat}: {max_other:.4f}.

        Raw spend is a weaker standalone predictor than behavioural features.

        **Action:** Focus on qualifying engaged leads rather than
        increasing raw spend alone.
        """)
    else:
        st.warning(f"""
        ❌ **Hypothesis 3 — NOT CONFIRMED**

        AdSpend performed similarly to or better than the strongest behavioural feature.
        """)

    # ── Summary Table ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hypotheses Summary")

    summary = pd.DataFrame({
        "Hypothesis": [
            "H1 — Engagement depth predicts conversion",
            "H2 — Campaign channel affects conversion rate",
            "H3 — AdSpend alone is a weak predictor",
        ],
        "Method": [
            "Point-biserial correlation",
            "Chi-square test",
            "Correlation comparison",
        ],
        "Result": [
            "Confirmed ✅" if p_time < 0.05 and p_pages < 0.05 else "Not confirmed ❌",
            "Rejected ❌" if p_chi >= 0.05 else "Confirmed ✅",
            "Confirmed ✅" if adspend_corr < max_other else "Not confirmed ❌",
        ],
        "Key finding": [
            f"TimeOnSite r={r_time:.3f}, PagesPerVisit r={r_pages:.3f}",
            f"chi2={chi2:.2f}, p={p_chi:.2e}",
            f"AdSpend={adspend_corr:.4f} vs {best_feat}={max_other:.4f}",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.info(f"""
    **Key Business Conclusions from Hypothesis Testing:**

    * **H1 CONFIRMED** — Engagement depth is a statistically significant
      predictor of conversion. Converted leads spend {diff_pct:.1f}% more
      time on site. ConvertIQ should prioritise content strategies that
      increase dwell time and page exploration over broad reach campaigns.

    * **H2 REJECTED** — Campaign channel alone does not determine conversion
      outcome (p={p_chi:.3f} > 0.05). Budget reallocation decisions should
      not be based on channel selection alone — engagement quality matters more
      than the acquisition channel.

    * **H3 CONFIRMED** — Advertising spend is a weaker predictor than
      behavioural signals. A lead showing high engagement but moderate spend
      exposure is more likely to convert than one with high spend but low
      interaction. ConvertIQ should use engagement-based lead scoring rather
      than spend-based targeting.
    """)
