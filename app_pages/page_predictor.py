import os

import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_pipeline():
    path = "outputs/ml_pipeline/v1/classification_pipeline.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None


def page_predictor_body():
    st.title("Conversion Predictor")
    st.subheader("Business Requirement 2 — Will this lead convert?")

    st.markdown("""
    Enter the profile of a lead below and click **Predict Conversion**
    to receive an instant prediction from the trained ML pipeline.
    """)

    # Small responsive styling for result cards and spacing
    st.markdown("""
    <style>
    .prediction-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .prediction-card.success {
        background-color: #d4edda;
    }
    .prediction-card.error {
        background-color: #f8d7da;
    }
    .prediction-title {
        font-size: 26px;
        font-weight: 700;
        white-space: nowrap;
        line-height: 1.2;
    }
    .prediction-subtext {
        margin-top: 0.75rem;
    }
    @media (max-width: 1024px) {
        .prediction-title {
            font-size: 22px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    pipeline = load_pipeline()

    if pipeline is None:
        st.warning("""
        **Model not yet trained.**
        Please run `jupyter_notebooks/05_Modelling.ipynb` first to
        generate the pipeline file at
        `outputs/ml_pipeline/v1/classification_pipeline.pkl`.
        """)
        return

    st.markdown("---")
    st.subheader("Lead Profile Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age = st.slider("Age", min_value=18, max_value=70, value=35)
        gender = st.selectbox("Gender", options=["Female", "Male"])
        income = st.slider(
            "Annual Income (USD)",
            min_value=20000,
            max_value=150000,
            value=60000,
            step=1000,
        )

    with col2:
        st.markdown("**Campaign Details**")
        campaign_channel = st.selectbox(
            "Campaign Channel",
            options=["Email", "SEO", "PPC", "Social Media", "Referral"],
        )
        campaign_type = st.selectbox(
            "Campaign Type",
            options=["Awareness", "Consideration", "Conversion", "Retention"],
        )
        ad_spend = st.slider(
            "Ad Spend (USD)",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=50.0,
        )
        ctr = st.slider(
            "Click-Through Rate",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
        )

    with col3:
        st.markdown("**Engagement Metrics**")
        website_visits = st.slider(
            "Website Visits",
            min_value=0,
            max_value=50,
            value=5,
        )
        pages_per_visit = st.slider(
            "Pages Per Visit",
            min_value=1.0,
            max_value=20.0,
            value=3.0,
            step=0.5,
        )
        time_on_site = st.slider(
            "Time on Site (min)",
            min_value=0.0,
            max_value=60.0,
            value=5.0,
            step=0.5,
        )
        social_shares = st.slider(
            "Social Shares",
            min_value=0,
            max_value=100,
            value=5,
        )
        email_opens = st.slider(
            "Email Opens",
            min_value=0,
            max_value=30,
            value=3,
        )
        email_clicks = st.slider(
            "Email Clicks",
            min_value=0,
            max_value=20,
            value=1,
        )
        previous_purchases = st.slider(
            "Previous Purchases",
            min_value=0,
            max_value=10,
            value=1,
        )
        loyalty_points = st.slider(
            "Loyalty Points",
            min_value=0,
            max_value=5000,
            value=500,
        )

    st.markdown("---")

    if st.button("Predict Conversion", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Income": [income],
            "CampaignChannel": [campaign_channel],
            "CampaignType": [campaign_type],
            "AdSpend": [ad_spend],
            "ClickThroughRate": [ctr],
            "WebsiteVisits": [website_visits],
            "PagesPerVisit": [pages_per_visit],
            "TimeOnSite": [time_on_site],
            "SocialShares": [social_shares],
            "EmailOpens": [email_opens],
            "EmailClicks": [email_clicks],
            "PreviousPurchases": [previous_purchases],
            "LoyaltyPoints": [loyalty_points],
        })

        try:
            probability = float(pipeline.predict_proba(input_data)[0][1])

            st.markdown("---")
            st.subheader("Prediction Result")

            # Equal-width columns are more reliable across tablet sizes
            col_pred, col_prob = st.columns(2)

            with col_pred:
                if probability >= 0.5:
                    st.markdown("""
                    <div class="prediction-card success">
                        <div class="prediction-title">✔️ CONVERTED</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(
                        "<div class='prediction-subtext'>This lead is predicted to <strong>convert</strong>.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("""
                    <div class="prediction-card error">
                        <div class="prediction-title">✘ NOT CONVERTED</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(
                        "<div class='prediction-subtext'>This lead is predicted <strong>not to convert</strong>.</div>",
                        unsafe_allow_html=True,
                    )

            with col_prob:
                st.markdown("**Conversion Probability**")
                st.progress(probability)
                st.markdown(f"**{probability:.1%}** probability")

                if probability >= 0.80:
                    st.success("High confidence — prioritise this lead.")
                elif probability >= 0.60:
                    st.warning("Moderate confidence — worth following up.")
                else:
                    st.info("Low probability — consider nurturing first.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.markdown("""
    **How this works:** The model uses a Random Forest Classifier trained
    on 6,400 lead profiles with SMOTE to address class imbalance.
    See the **Model Performance** page for full evaluation metrics.
    """)
