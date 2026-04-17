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
    * **BR1 — Customer Behaviour Analysis:**
      Understand which customer attributes and campaign engagement metrics
      correlate most strongly with conversion, with supporting data visualisations.

    * **BR2 — Conversion Prediction:**
      Predict whether a given lead will convert based on their demographic
      profile and behavioural engagement data, so the sales team can
      prioritise outreach efficiently.

    * **BR3 — Campaign ROI Intelligence:**
      Analyse marketing spend efficiency across campaign categories
      to support budget allocation decisions.
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
        """)

    with st.expander("View variable definitions"):
        st.markdown("""
        | Variable | Type | Description |
        |---|---|---|
        | c_date | Date | Date of campaign activity |
        | campaign_name | Categorical | Name of the campaign |
        | category | Categorical | Campaign category (social, search, etc.) |
        | impressions | Numeric | Total ad impressions |
        | mark_spent | Numeric | Marketing spend in USD |
        | clicks | Numeric | Total clicks |
        | leads | Numeric | Number of leads generated |
        | orders | Numeric | Number of orders placed |
        | revenue | Numeric | Revenue generated in USD |
        | **roi** | **Derived** | **(revenue - mark_spent) / mark_spent × 100** |
        | **month** | **Derived** | **Month extracted from c_date** |
        """)

    st.markdown("""

    ---
                
    ### Key Terms

    * **Conversion**: A lead who completes the desired action
    * **CTR**: Click-Through Rate — proportion of impressions that result in a click
    * **ROI**: Return on Investment — `(Revenue - Spend) / Spend`
    * **Recall**: Proportion of actual converters correctly identified
    * **F1-Score**: Harmonic mean of precision and recall
    * **SMOTE**: Synthetic Minority Oversampling Technique — addresses class imbalance
    """)

    st.markdown("---")
    st.subheader("📌 Key Business Conclusions")

    st.success("""
    **What drives conversion at ConvertIQ:**

    * Leads that spend more time on site and click more emails are
      significantly more likely to convert — prioritise engagement signals
      over demographic data when qualifying leads.

    * Campaign channel alone does not determine conversion outcome.
      Email, SEO, PPC, Social Media and Referral channels produce
      statistically equivalent conversion rates (p = 0.594).
      Do not reallocate budget based on channel alone.

    * Advertising spend is a weak standalone predictor (r = 0.12).
      A lead with high engagement but low ad exposure is more likely
      to convert than one with high spend but low interaction.
    """)

    st.info("""
    **How to use this dashboard:**

    * 📊 Go to **Customer Behaviour Analysis** to explore which features
      correlate most with conversion
    * 🤖 Go to **Conversion Predictor** to input a lead profile and get
      an instant prediction
    * 💰 Go to **Campaign ROI Analysis** to evaluate which campaigns
      deliver the best return
    * 📈 Go to **Model Performance** to review how well the ML model performs
    """)

    st.markdown(
        "For a complete overview, please refer to the [Project README](https://github.com/cynthiapinedoh79/digital-marketing-conversion-predictor)."
    )


   
