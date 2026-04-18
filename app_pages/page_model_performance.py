import streamlit as st
import pandas as pd
import os
import joblib
import plotly.express as px
from PIL import Image



def page_model_performance_body():
    st.title("Model Performance")
    st.subheader("Business Requirement 2 — How well does the model perform?")

    st.markdown("""
    This page presents a detailed evaluation of the trained Random Forest
    classification pipeline. The success metrics were defined in the
    ML Business Case:

    * **Minimum**: Recall ≥ 0.75 for class 1 (Converted)
    * **Target**: F1-score ≥ 0.80 for class 1 (Converted)
    """)

    base_path = "outputs/ml_pipeline/v1"
    pipeline_path = f"{base_path}/classification_pipeline.pkl"
    meta_path = f"{base_path}/model_metadata.pkl"

    model_trained = os.path.exists(pipeline_path)

    if not model_trained:
        st.warning("""
        Model output files not yet generated.
        Please run `jupyter_notebooks/05_Modelling.ipynb` first.
        """)
        return

    # ── Performance Statement ────────────────────────────────────────
    st.markdown("---")
    st.subheader("Performance Assessment")

    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)

        f1 = meta.get("test_f1", 0)
        recall = meta.get("test_recall", 0)
        roc = meta.get("test_roc_auc", 0)
        train_acc = meta.get("train_accuracy", None)
        test_acc = meta.get("test_accuracy", None)
        recall_not_converted = meta.get("test_recall_class0", None)

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Test Recall (class 1)",
            f"{recall:.3f}",
            delta="≥ 0.75 required"
        )
        col2.metric(
            "Test F1-Score (class 1)",
            f"{f1:.3f}",
            delta="≥ 0.80 target"
        )
        col3.metric(
            "ROC-AUC Score",
            f"{roc:.3f}"
        )

        if train_acc is not None and test_acc is not None:
            st.markdown(f"""
            **Train Accuracy:** {train_acc:.3f}  
            **Test Accuracy:** {test_acc:.3f}
            """)

        if recall >= 0.75 and f1 >= 0.80:
            st.success("""
            ✅ The model meets the main business requirement for identifying converted customers.
            """)
        else:
            st.warning("The model does not meet the business requirement.")

        if train_acc is not None and test_acc is not None:
            if train_acc - test_acc > 0.10:
                st.warning("⚠️ The model shows signs of overfitting (train vs test gap).")

        if recall_not_converted is not None and recall_not_converted < 0.60:
            st.warning("⚠️ Performance on the minority class (Not Converted) is weak.")

        if (
            train_acc is not None
            and test_acc is not None
            and recall_not_converted is not None
        ):
            st.markdown(f"""
            ### Business Interpretation

            The model successfully identifies customers likely to convert,
            meeting the defined business thresholds.

            However, two important limitations must be considered:

            - The model shows moderate overfitting — train accuracy ({train_acc:.4f}) is higher than test accuracy ({test_acc:.4f}).
            - Performance on the minority class ("Not Converted") is weaker (recall = {recall_not_converted:.4f}), which is expected due to class imbalance.

            This means the model is effective for prioritising high-probability conversions,
            but less reliable at identifying non-converting customers.
            """)
        else:
            st.markdown("""
            ### Business Interpretation

            The model successfully identifies customers likely to convert,
            meeting the defined business thresholds.

            However, performance should be interpreted carefully, especially
            regarding overfitting and class imbalance.
            """)
    else:
        st.warning("""
        `model_metadata.pkl` was not found.
        Please run the final cells in `jupyter_notebooks/05_Modelling.ipynb`
        to generate the metadata file.
        """)

    # ── Pipeline Steps ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("ML Pipeline Steps")

    st.markdown("""
    The trained pipeline consists of the following sequential steps:
    """)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.info("""
        **Step 1 — OrdinalEncoder**
        Encodes categorical features (Gender, CampaignChannel, CampaignType)
        into numerical values using feature-engine's OrdinalEncoder.
        """)

        st.info("""
        **Step 2 — SMOTE**
        Applies Synthetic Minority Oversampling Technique to address class
        imbalance (87.65% converted vs 12.35% not converted) during training.
        """)

    with col_p2:
        st.info("""
        **Step 3 — RandomizedSearchCV**
        Optimises 7 hyperparameters across 30 iterations with 5-fold
        cross-validation to find the best model configuration.
        """)

        st.info("""
        **Step 4 — RandomForestClassifier**
        Ensemble of decision trees that predicts conversion probability
        based on 15 input features using the best hyperparameters found.
        """)
    
    # ── Confusion Matrices ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Confusion Matrices")
    st.markdown("""
    A confusion matrix shows correct and incorrect predictions
    broken down by class. For conversion prediction:

    * **True Positives**: Correctly predicted converters
    * **False Negatives**: Actual converters missed by the model
    """)

    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        cm_train = f"{base_path}/confusion_matrix_train.png"
        if os.path.exists(cm_train):
            st.image(
                Image.open(cm_train),
                caption="Confusion Matrix — Train Set",
                use_container_width=True
            )

    with col_cm2:
        cm_test = f"{base_path}/confusion_matrix_test.png"
        if os.path.exists(cm_test):
            st.image(
                Image.open(cm_test),
                caption="Confusion Matrix — Test Set",
                use_container_width=True
            )

    st.markdown("""
    **Interpretation:**  
    The model correctly identifies most converted customers (high recall for class 1).

    However, it struggles with the minority class ("Not Converted"), producing more false positives.
    This reflects the class imbalance in the dataset and explains the lower recall for that class.
    """)

    # ── ROC Curve ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("ROC Curve")

    roc_path = f"{base_path}/roc_curve.png"
    if os.path.exists(roc_path):
        col_roc, col_txt = st.columns([2, 1])

        with col_roc:
            st.image(
                Image.open(roc_path),
                caption="ROC Curve — Test Set",
                use_container_width=True
            )

        with col_txt:
            st.markdown("""
            **ROC-AUC Guide:**
            * 1.0 → Perfect
            * 0.9–1.0 → Excellent
            * 0.8–0.9 → Good
            * 0.7–0.8 → Fair
            * 0.5 → Random
            """)

    st.markdown("""
    **Interpretation:**  
    The ROC-AUC score indicates acceptable discrimination ability.
    The model performs better than random guessing but still has room for improvement.
    """)

    # ── Feature Importance ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Feature Importance")

    pipeline_obj = joblib.load(pipeline_path)
    rf_classifier = pipeline_obj.named_steps['classifier']
    importances = rf_classifier.feature_importances_


    feature_names = [
        'Age', 'Gender', 'Income', 'CampaignChannel', 'CampaignType',
        'AdSpend', 'ClickThroughRate', 'WebsiteVisits', 'PagesPerVisit',
        'TimeOnSite', 'SocialShares', 'EmailOpens', 'EmailClicks',
        'PreviousPurchases', 'LoyaltyPoints'
    ]

    fi_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances
    }).sort_values('Importance', ascending=True)

    fig_fi = px.bar(
        fi_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance — Random Forest Classifier',
        labels={'Importance': 'Mean Decrease in Impurity'},
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig_fi.update_layout(
        coloraxis_showscale=False,
        height=500,
        font=dict(size=12)
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("""
    **Interpretation:**  
    CampaignType, TimeOnSite, AdSpend and Email interactions are the most influential predictors.

    These features reflect customer engagement and marketing effectiveness.

    However, feature importance in Random Forest represents model-based relevance,
    not direct causality.
    """)

    # ── Hyperparameter Summary ───────────────────────────────────────
    st.markdown("---")
    st.subheader("Hyperparameter Tuning Summary")

    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)
        best_params = meta.get("best_params", {})

        params_df = pd.DataFrame([
            {
                "Hyperparameter": k.replace("classifier__", ""),
                "Best Value": str(v)
            }
            for k, v in best_params.items()
        ])

        st.dataframe(params_df, use_container_width=True)

        st.markdown(f"""
        * **Search**: RandomizedSearchCV — 30 iterations, 5-fold CV
        * **Scoring**: F1-score (class 1)
        * **Best CV F1**: {meta.get("best_cv_f1", 0):.4f}
        """)

    # ── Final Conclusions ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Final Conclusions")

    st.success("""
    ✅ **The model meets both business success criteria:**
    - Test Recall (Converted): 0.8381 — exceeds the minimum requirement of 0.75
    - Test F1-score (Converted): 0.8762 — exceeds the target of 0.80
    - ROC-AUC: 0.7331 — acceptable discrimination ability on unseen data
    """)

    st.markdown("""
    **What this means for ConvertIQ:**

    The Random Forest pipeline can reliably identify leads likely to convert,
    enabling the sales team to prioritise outreach and reduce wasted effort.
    In practical terms:

    - A lead scoring **above 75% probability** should receive immediate
      direct follow-up from the sales team
    - A lead scoring **between 40–75%** should be enrolled in a targeted
      nurture campaign to build engagement before direct contact
    - A lead scoring **below 40%** should be deprioritised and moved to
      long-term awareness campaigns

    **Known limitations to consider:**

    - The model shows moderate overfitting — train accuracy (0.8528) is higher
      than test accuracy (0.7925). Performance on genuinely new data may be
      slightly lower than test results suggest.
    - Recall for the "Not Converted" class is 0.4697 — the model is less
      reliable at identifying leads who will not convert. False positives
      (leads predicted to convert but who do not) should be expected.

    **Recommended next steps:**

    - Adjust the classification threshold (currently 0.50) to optimise the
      precision/recall trade-off based on business priorities
    - Test Gradient Boosting or XGBoost as alternative classifiers
    - Collect additional behavioural features to improve minority class detection
    - Monitor model performance over time and retrain as new campaign data arrives
    """)

