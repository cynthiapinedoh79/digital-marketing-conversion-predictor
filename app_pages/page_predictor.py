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
    st.markdown("# Conversion Predictor")
    st.subheader("Business Requirement 2 — Will this lead convert?")

    st.markdown("Enter lead details and click **Predict Conversion** to estimate conversion probability.")
    st.caption("Supports data-driven decision making in marketing campaigns.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Prioritise leads by predicting conversion probability based on engagement and campaign data.")

    with st.expander("ℹ️ How to interpret input values"):
        st.markdown("""
    **Engagement metrics** (Email Opens, Clicks, Website Visits) represent cumulative interactions per lead.

    These values are not tied to a fixed timeframe. Instead, they reflect engagement across the typical lifecycle of a lead.

    **Business tip:** Compare leads within a consistent campaign window (e.g. first 2–4 weeks).
    """)

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
    st.markdown("## 🧾 Lead Profile Input")

    st.caption("📊 Inputs represent cumulative engagement per lead across a typical campaign cycle.")
    st.caption("💡 Compare leads using consistent campaign stages (e.g. first 2–4 weeks).")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Demographics")
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
        st.markdown("#### 📈 Campaign Details")
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
        st.markdown("#### 📊 Engagement Metrics")
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

    if st.button("🚀 Predict Conversion", type="primary", use_container_width=True):
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
            st.subheader(" 📊 Prediction Result")

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
                st.markdown("#### Conversion Probability")
                percentage = probability * 100

                if probability >= 0.75:
                    score_color = "#065f46"
                    bar_gradient = "linear-gradient(90deg, #10b981, #059669)"
                    bar_shadow = " 0 0 4px rgba(16, 185, 129, 0.25)"

                    likelihood_label = "🟢 High likelihood to convert"
                    lead_label = "🔥 High-Value Lead"
                    lead_bg = "#d4edda"
                    lead_text = "#155724"

                    quick_bg = "#d4edda"
                    quick_text = "#155724"

                    action_bg = "#ecfdf5"
                    action_border = "#10b981"
                    action_text = "#065f46"

                elif probability >= 0.40:
                    score_color = "#92400e"
                    bar_gradient = "linear-gradient(90deg, #f59e0b, #d97706)"
                    bar_shadow = "0 0 4px rgba(217, 119, 6, 0.25)"

                    likelihood_label = "🟡 Moderate likelihood to convert"
                    lead_label = "⚡ Potential Lead"
                    lead_bg = "#fef3c7"
                    lead_text = "#92400e"

                    quick_bg = "#fffbeb"
                    quick_text = "#92400e"

                    action_bg = "#fffbeb"
                    action_border = "#f59e0b"
                    action_text = "#92400e"

                else:
                    score_color = "#dc3545"
                    bar_gradient = "linear-gradient(90deg, #ef4444, #dc2626)"
                    bar_shadow = "0 0 4px rgba(220, 38, 38, 0.25)"

                    likelihood_label = "🔴 Low likelihood to convert"
                    lead_label = "🧊 Cold Lead"
                    lead_bg = "#e2e3e5"
                    lead_text = "#4b5563"

                    quick_bg = "#fef2f2"
                    quick_text = "#991b1b"

                    action_bg = "#fef2f2"
                    action_border = "#ef4444"
                    action_text = "#991b1b"

                st.markdown(f"""
                <div style='
                    font-size:52px;
                    font-weight:800;
                    text-align:center;
                    margin-bottom:6px;
                    color:{score_color};
                '>
                    {percentage:.0f}%
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="
                    text-align:center;
                    font-size:14px;
                    color:#475569;  /* 👈 gris profesional */
                    margin-top:6px;
                    margin-bottom:12px;
                    font-weight:500;
                ">
                    {probability:.2%} probability
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="
                    width:100%;
                    background-color:#e9ecef;
                    border-radius:10px;
                    height:14px;
                    margin:6px 0 12px 0;
                ">
                    <div style="
                        width:{percentage}%;
                        background: {bar_gradient};
                        height:14px;
                        border-radius:10px;
                        transition: width 0.5s ease-in-out;
                        box-shadow: {bar_shadow};
                    "></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    f"<div style='text-align:center; font-size:20px; font-weight:700; color:{score_color}; margin-bottom:12px;'>{likelihood_label}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(f"""
                <div style='
                    background-color:{lead_bg};
                    color:{lead_text};
                    padding:8px 14px;
                    border-radius:20px;
                    display:block;
                    width:fit-content;
                    margin:0 auto 14px auto;
                    font-weight:600;
                '>
                    {lead_label}
                </div>
                """, unsafe_allow_html=True)

                if probability < 0.40:
                    msg = "Low probability — consider nurturing first."
                elif probability < 0.75:
                    msg = "Moderate probability — reinforce engagement."
                else:
                    msg = "High probability — prioritise conversion."

                st.markdown(f"""
                <div style='
                    background-color:{quick_bg};
                    padding:14px;
                    border-radius:10px;
                    margin:10px 0 15px 0;
                    text-align:center;
                    font-weight:600;
                    color:{quick_text};
                '>
                    {msg}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 📌 Interpretation")
            st.markdown("""
            <div style='
                background-color:#f9fafb;
                padding:22px;
                border-radius:16px;
                margin-top:20px;
                margin-bottom:20px;
                box-shadow:0 4px 12px rgba(0,0,0,0.05);
                border:1px solid #e5e7eb;
            '>
                <h4 style='color:#111827; margin-bottom:8px;'>🧠 Model Insight</h4>
                <p style='color:#374151; font-size:15px; line-height:1.6; margin:0;'>
                    This prediction combines engagement metrics, campaign interaction,
                    and historical conversion patterns to support data-driven decision-making.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # -------------------------------
            # 📱 SOCIAL MEDIA INSIGHT
            # -------------------------------
            
            if campaign_channel == "Social Media":
                st.markdown("---")
                st.markdown("### 📱 Social Media Performance Insight")

                social_raw = pd.DataFrame({
                    "campaign_name": [
                        "facebook_tier1", "facebook_tier2",
                        "facebook_retargeting", "facebook_lal",
                        "instagram_tier1", "instagram_tier2"
                    ],
                    "platform": [
                        "Facebook", "Facebook", "Facebook", "Facebook",
                        "Instagram", "Instagram"
                    ],
                    "orders": [474, 688, 108, 294, 758, 313],
                    "revenue": [2396412, 3463306, 536919, 300233, 4544124, 670460],
                    "mark_spent": [2564793, 4693870, 266466, 2641939, 2565277, 1066154]
                })

                platform_df = social_raw.groupby("platform").agg(
                    Orders=("orders", "sum"),
                    Revenue=("revenue", "sum"),
                    Spent=("mark_spent", "sum")
                ).reset_index()

                platform_df["ROI (%)"] = (
                    (platform_df["Revenue"] - platform_df["Spent"])
                    / platform_df["Spent"].replace(0, pd.NA) * 100
                ).round(1)

                import plotly.express as px

                col_s1, col_s2 = st.columns(2)

                with col_s1:
                    fig1 = px.bar(
                        platform_df, x="platform", y="Orders",
                        title="Orders by Platform",
                        color="platform",
                    )
                    fig1.update_layout(
                        showlegend=False, bargap=0.25,
                        xaxis=dict(tickfont=dict(size=14, color="black")),
                        yaxis=dict(tickfont=dict(size=12, color="black"))
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with col_s2:
                    fig2 = px.bar(
                        platform_df, x="platform", y="ROI (%)",
                        title="ROI (%) by Platform",
                        color="platform",
                    )
                    fig2.update_layout(
                        showlegend=False, bargap=0.25,
                        xaxis=dict(tickfont=dict(size=14, color="black")),
                        yaxis=dict(tickfont=dict(size=12, color="black"))
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.dataframe(
                    platform_df[["platform", "Orders", "Revenue", "Spent", "ROI (%)"]],
                    use_container_width=True
                )

                ig_roi = platform_df.loc[platform_df["platform"] == "Instagram", "ROI (%)"].values[0]
                fb_roi = platform_df.loc[platform_df["platform"] == "Facebook", "ROI (%)"].values[0]

                if ig_roi > fb_roi:
                    st.markdown(f"""
                    Based on the supporting KPI dataset, **Instagram** currently outperforms Facebook
                    for social media conversion efficiency, with a **{ig_roi:.1f}% ROI** compared to
                    **{fb_roi:.1f}%** for Facebook. Instagram also generates a higher order volume,
                    suggesting it may be the stronger platform for conversion-focused social campaigns.
                    """)
                else:
                    st.markdown(f"""
                    Based on the supporting KPI dataset, **Facebook** currently outperforms Instagram
                    for social media conversion efficiency, with a **{fb_roi:.1f}% ROI** compared to
                    **{ig_roi:.1f}%** for Instagram. This suggests that budget allocation could be
                    prioritised toward higher-performing Facebook social campaigns.
                    """)

            # -------------------------------
            # 🎯 DYNAMIC ACTIONS (PLAYBOOK)
            # -------------------------------

            actions = []

            # Solo generar acciones si hay señales débiles
            if probability < 0.75:

                if time_on_site < 5:
                    actions.append("Increase engagement through landing page optimisation or better content.")

                if email_clicks == 0:
                    actions.append("Send targeted email campaigns to stimulate interest.")

                if pages_per_visit < 3:
                    actions.append("Improve website navigation or highlight key offers more clearly.")

                if previous_purchases == 0:
                    actions.append("Build trust with testimonials, reviews, or introductory offers.")

                if ctr < 0.05:
                    actions.append("Refine ad creatives or targeting strategy to improve click-through rate.")

            # Build dynamic factor list based on actual inputs
            positive_factors = []
            risk_factors = []

            if time_on_site >= 10:
                positive_factors.append("Higher time on site suggests stronger engagement.")
            else:
                risk_factors.append("Lower time on site suggests weaker engagement.")

            if pages_per_visit >= 4:
                positive_factors.append("More pages per visit indicates deeper exploration of the offer.")
            else:
                risk_factors.append("Fewer pages per visit suggests limited exploration.")

            if email_clicks >= 2:
                positive_factors.append("Email clicks indicate active interest in campaign content.")
            else:
                risk_factors.append("Limited email clicks reduce evidence of active campaign interest.")

            if previous_purchases >= 1:
                positive_factors.append("Previous purchases may reflect existing brand trust.")
            else:
                risk_factors.append("No previous purchases means there is less evidence of prior loyalty.")

            if loyalty_points >= 1000:
                positive_factors.append("Higher loyalty points suggest a stronger brand relationship.")

            if ctr >= 0.10:
                positive_factors.append("A higher click-through rate reflects stronger campaign responsiveness.")
            else:
                risk_factors.append("Lower click-through rate suggests weaker campaign responsiveness.")

            if ad_spend >= 1000:
                positive_factors.append("Higher ad spend may have increased campaign exposure.")

            # Dynamic interpretation by probability band
            if probability < 0.40:
                st.markdown("""
                This lead shows a **low likelihood of conversion**. The overall profile indicates
                weaker engagement and lower alignment with the behaviour patterns most commonly
                associated with converted leads.
                """)

                st.markdown("### Key Factors Considered")
                for factor in risk_factors[:4]:
                    st.markdown(f"- {factor}")
                if not risk_factors:
                    st.markdown("- Engagement signals are currently limited or mixed.")

                action_bg = "#fef2f2"
                action_border = "#ef4444"
                action_text = "#991b1b"
                action_msg = (
                    "Focus on nurturing first. This profile lacks strong engagement signals. "
                    "Prioritise trust-building strategies before attempting conversion."
                )

                if actions:
                    st.markdown("""
                    <div style='margin-top:40px;'>
                        <h4 style='margin-bottom:10px; font-weight:700;'>🎯 Suggested Next Best Actions</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    for action in actions[:3]:
                        st.markdown(f"- {action}")

            elif probability < 0.75:
                st.markdown("""
                This lead shows a **moderate likelihood of conversion**. There are clear signs
                of interest, but the profile still shows a mix of encouraging and weaker signals.
                Conversion may improve with more targeted follow-up.
                """)

                st.markdown("### Key Factors Considered")
                shown_factors = positive_factors[:2] + risk_factors[:2]
                if shown_factors:
                    for factor in shown_factors:
                        st.markdown(f"- {factor}")
                else:
                    st.markdown("- This lead shows a mixed profile with moderate engagement.")

                action_bg = "#fffbeb"
                action_border = "#f59e0b"
                action_text = "#92400e"
                action_msg = (
                    "This lead shows potential. Reinforce value proposition and remove friction "
                    "through personalised messaging or targeted campaigns."
                )

                if actions:
                    st.markdown("""
                    <div style='margin-top:40px;'>
                        <h4 style='margin-bottom:10px; font-weight:700;'>🎯 Suggested Next Best Actions</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    for action in actions[:3]:
                        st.markdown(f"- {action}")

            else:
                st.markdown("""
                This lead shows a **high likelihood of conversion**. The profile aligns strongly
                with engagement and behaviour patterns historically associated with converted leads.
                """)

                st.markdown("### Key Factors Considered")
                for factor in positive_factors[:4]:
                    st.markdown(f"- {factor}")
                if not positive_factors:
                    st.markdown("- This lead shows several strong engagement indicators overall.")

                action_bg = "#ecfdf5"
                action_border = "#10b981"
                action_text = "#065f46"
                action_msg = (
                    "This is a high-value lead. Prioritise immediate conversion actions such as "
                    "direct offers, urgency triggers, and fast follow-up."
                )

                if actions:
                    st.markdown("""
                    <div style='margin-top:40px;'>
                        <h4 style='margin-bottom:10px; font-weight:700;'>🎯 Suggested Next Best Actions</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    for action in actions[:3]:
                        st.markdown(f"- {action}")

            st.markdown("### Recommended Action")
            st.markdown(f"""
            <div style='
                background-color:{action_bg};
                padding:18px;
                border-radius:14px;
                margin-top:10px;
                border-left:6px solid {action_border};
                line-height:1.7;
            '>
                <span style='color:{action_text}; font-size:15px;'>
                    {action_msg}
                </span>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.markdown("""
    **How this works:** The model uses a Random Forest Classifier trained
    on 6,400 lead profiles with SMOTE to address class imbalance.
    See the **Model Performance** page for full evaluation metrics.
    """)
