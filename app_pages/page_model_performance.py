import streamlit as st
import pandas as pd
import os
import joblib
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

    fi_path = f"{base_path}/feature_importance.png"
    if os.path.exists(fi_path):
        st.image(
            Image.open(fi_path),
            caption="Feature Importances — Random Forest",
            use_container_width=True
        )

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

    st.markdown("""
    The Random Forest model achieved the main business objective of identifying customers likely to convert,
    meeting the required recall and F1-score thresholds for the positive class.

    However, the evaluation revealed two important limitations:

    - The model shows overfitting, with higher performance on the training set than on the test set.
    - Performance on the minority class ("Not Converted") remains limited due to class imbalance.

    Overall, the model is suitable as an initial predictive solution for business decision support,
    but further optimisation is recommended before production deployment.
    """)
