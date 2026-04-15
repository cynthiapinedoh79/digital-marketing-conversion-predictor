import streamlit as st


def page_summary_body():
    st.title("Digital Marketing Conversion Predictor")
    st.subheader("Project Summary")

    st.markdown("""
    ---
    ### Business Context

    **ConvertIQ** is a fictional digital marketing agency that runs campaigns
    across five channels for a diverse client base. The agency lacks a
    data-driven method to identify which leads are most likely to convert,
    causing their sales team to spend equal time on high and low probability
    prospects.

    This dashboard presents a machine learning solution that addresses
    three business requirements:
    """)

    st.info("""
    * **BR1** — Understand which customer and campaign attributes
      correlate with conversion
    * **BR2** — Predict whether a given lead will convert
    * **BR3** — Analyse marketing spend efficiency across campaign categories
    """)

    st.markdown("---")
    st.subheader("Dataset Description")

    st.markdown("""
    **Dataset 1 — Digital Marketing Campaign**
    * Source: [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset)
    * 8,000 rows and 16 features (after cleaning)
    * Each row represents a unique lead exposed to a marketing campaign
    """)

    with st.expander("View variable definitions"):
        st.markdown("""
        | Variable | Type | Description |
        |---|---|---|
        | Age | Numeric | Customer age in years |
        | Gender | Categorical | Male / Female |
        | Income | Numeric | Annual income (USD) |
        | CampaignChannel | Categorical | Email / SEO / PPC / Social Media / Referral |
        | CampaignType | Categorical | Awareness / Consideration / Conversion / Retention |
        | AdSpend | Numeric | Campaign advertising spend (USD) |
        | ClickThroughRate | Numeric | Ratio of clicks to impressions |
        | WebsiteVisits | Numeric | Number of website visits |
        | PagesPerVisit | Numeric | Average pages viewed per visit |
        | TimeOnSite | Numeric | Average time on site (minutes) |
        | SocialShares | Numeric | Social media shares |
        | EmailOpens | Numeric | Number of email opens |
        | EmailClicks | Numeric | Number of email clicks |
        | PreviousPurchases | Numeric | Number of prior purchases |
        | LoyaltyPoints | Numeric | Loyalty programme points |
        | **Conversion** | **Target** | **1 = Converted, 0 = Not Converted** |
        """)

    st.markdown("""
    **Dataset 2 — Digital Marketing KPIs**
    * Source: [Kaggle](https://www.kaggle.com/datasets/sinderpreet/analyze-the-marketing-spending)
    * 308 rows — daily campaign performance metrics
    * Used for Campaign ROI Analysis (BR3)

    ---
    ### Key Terms

    * **Conversion**: A lead who completes the desired action
    * **CTR**: Click-Through Rate — proportion of impressions that result in a click
    * **ROI**: Return on Investment — `(Revenue - Spend) / Spend`
    * **Recall**: Proportion of actual converters correctly identified
    * **F1-Score**: Harmonic mean of precision and recall
    * **SMOTE**: Synthetic Minority Oversampling Technique — addresses class imbalance
    """)
