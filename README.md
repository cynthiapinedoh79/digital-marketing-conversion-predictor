# Digital Marketing Conversion Predictor

## Table of Contents
1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and Validation](#hypothesis-and-validation)
4. [Rationale to Map Business Requirements](#rationale-to-map-business-requirements)
5. [ML Business Case](#ml-business-case)
6. [Dashboard Design](#dashboard-design)
7. [Unfixed Bugs](#unfixed-bugs)
8. [Deployment](#deployment)
9. [Main Libraries](#main-libraries)
10. [Credits](#credits)

---

## Dataset Content

The project uses two datasets sourced from Kaggle:

**Dataset 1 — Digital Marketing Campaign Dataset**
Source: [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset)
- 8,000 rows and 20 columns (16 after cleaning)
- Each row represents a unique lead exposed to a digital marketing campaign
- Target variable: `Conversion` (1 = converted, 0 = not converted)

| Variable | Type | Description |
|---|---|---|
| Age | Numeric | Customer age in years |
| Gender | Categorical | Male / Female |
| Income | Numeric | Annual income in USD |
| CampaignChannel | Categorical | Email / SEO / PPC / Social Media / Referral |
| CampaignType | Categorical | Awareness / Consideration / Conversion / Retention |
| AdSpend | Numeric | Campaign advertising spend in USD |
| ClickThroughRate | Numeric | Ratio of clicks to impressions |
| WebsiteVisits | Numeric | Number of website visits |
| PagesPerVisit | Numeric | Average pages viewed per visit |
| TimeOnSite | Numeric | Average time on site in minutes |
| SocialShares | Numeric | Number of social media shares |
| EmailOpens | Numeric | Number of email opens |
| EmailClicks | Numeric | Number of email clicks |
| PreviousPurchases | Numeric | Number of prior purchases |
| LoyaltyPoints | Numeric | Loyalty programme points |
| **Conversion** | **Target** | **1 = Converted, 0 = Not Converted** |

**Note:** `AdvertisingPlatform` and `AdvertisingTool` were dropped —
both contained only confidential placeholder values with no predictive value.
`ConversionRate` was dropped to prevent data leakage.

**Dataset 2 — Digital Marketing KPIs**
Source: [Kaggle](https://www.kaggle.com/datasets/sinderpreet/analyze-the-marketing-spending)
- 308 rows and 11 columns (13 after cleaning)
- Daily campaign performance metrics per campaign
- Used for supplementary ROI analysis (BR3)

---

## Business Requirements

A fictional digital marketing agency, **ConvertIQ**, needs a data-driven
solution to identify which leads are most likely to convert.

**BR1 — Customer Behaviour Analysis:**
The client wants to understand which customer attributes and campaign
engagement metrics correlate most strongly with conversion, with supporting
data visualisations.

**BR2 — Conversion Prediction:**
The client wants to predict whether a given lead will convert, based on
their demographic profile and behavioural engagement data, so their sales
team can prioritise outreach.

**BR3 — Campaign ROI Intelligence:**
The client wants to analyse marketing spend efficiency across campaign
categories to support budget allocation decisions.

---

## Hypothesis and Validation

### Hypothesis 1 — Engagement Depth Predicts Conversion

**Statement:** Leads that spend more time on site and view more pages
per visit are significantly more likely to convert than low-engagement leads.

**Validation method:** Point-biserial correlation between `TimeOnSite`,
`PagesPerVisit`, and the binary `Conversion` target (α = 0.05).

**Result:** ✅ **CONFIRMED**

- TimeOnSite: r = 0.1296, p = 2.58e-31  
- PagesPerVisit: r = 0.1028, p = 2.93e-20  
- Converted leads spend **47% more time on site** on average  

**Interpretation:**  
Both engagement variables show statistically significant positive
relationships with conversion, although the effect size is moderate.
This indicates that engagement contributes to conversion likelihood,
but no single variable alone is sufficient for prediction.

**Business Action:**  
ConvertIQ should invest in multi-step engagement strategies such as
interactive content, guided funnels, and personalised landing pages
to increase dwell time and page exploration.

---

### Hypothesis 2 — Campaign Channel Affects Conversion Rate

**Statement:** The campaign channel (Email, SEO, PPC, Social Media,
Referral) has a statistically significant effect on conversion rate.

**Validation method:** Chi-square test of independence between
`CampaignChannel` and `Conversion` (α = 0.05).

**Result:** ❌ **REJECTED**

- Chi-square = 2.785  
- p-value = 0.594  
- Degrees of freedom = 4  

No statistically significant association was found (p > 0.05).

**Interpretation:**  
Although minor differences in conversion rates exist across channels,
these differences are not statistically significant. This suggests that
channel selection alone does not meaningfully impact conversion outcomes.

**Learning:**  
Conversion performance is driven more by **user behaviour and engagement**
than by the acquisition channel itself.

---

### Hypothesis 3 — Ad Spend Alone is a Weak Predictor

**Statement:** Advertising spend (`AdSpend`) alone is not a reliable
predictor of conversion and has lower predictive power than behavioural features.

**Validation method:** Point-biserial correlation comparison between
`AdSpend` and engagement features against `Conversion`.

**Result:** ✅ **CONFIRMED**

- AdSpend correlation: 0.1247  
- EmailClicks correlation: 0.1295  
- TimeOnSite correlation: 0.1296  

AdSpend ranks below key engagement features in predictive strength.

**Interpretation:**  
Marketing spend contributes to conversion but does not strongly predict
outcomes on its own. Behavioural signals provide more meaningful insight
into customer intent.

**Business Action:**  
ConvertIQ should prioritise **engagement-based targeting strategies**
rather than increasing raw advertising spend.

---

## Rationale to Map Business Requirements

### BR1 — Customer Behaviour Analysis

**User Story:**  
As a marketing analyst, I want to identify which variables correlate
most strongly with conversion so that I can target high-quality leads.

**Actions:**
- Conduct Pearson and Spearman correlation analysis
- Visualise relationships using an interactive heatmap
- Compare distributions using box and violin plots
- Explore feature interactions using scatter plots
- Analyse conversion rates across campaign categories
- Provide clear business-oriented interpretations

---

### BR2 — Conversion Prediction (Machine Learning)

**User Story:**  
As a sales manager, I want to input a lead profile and receive a
real-time prediction so that my team can prioritise follow-up efforts.

**Actions:**
- Train a binary classification model to predict `Conversion`
- Optimise using RandomizedSearchCV (7 hyperparameters)
- Evaluate with confusion matrix, F1-score, recall, and ROC-AUC
- Save the trained pipeline using `joblib`
- Deploy a Streamlit interface for real-time predictions
- Display prediction probability and business interpretation

---

### BR3 — Campaign ROI Analysis

**User Story:**  
As a marketing director, I want to understand how campaign spend
translates into revenue so I can optimise budget allocation.

**Actions:**
- Clean and prepare the marketing KPIs dataset
- Calculate ROI per campaign category
- Visualise spend vs revenue relationships
- Compare average ROI across categories
- Analyse monthly revenue trends
- Provide actionable business recommendations

---

## ML Business Case

### Conversion Classifier

**Objective:**  
Develop a binary classification model to predict whether a lead will
convert (`Conversion = 1`) based on demographic and behavioural features.  
This directly addresses **Business Requirement 2 (Conversion Prediction)**.

---

### Learning Method

Supervised machine learning — binary classification using a  
**Random Forest Classifier**, optimised through **RandomizedSearchCV**.

---

### Ideal Outcome

A model that:

- Maximises **recall for the positive class (Converted)** to capture
  as many potential customers as possible  
- Maintains a strong balance between precision and recall  
- Enables the sales team to prioritise high-probability leads efficiently  

---

### Success Metrics

- **Minimum requirement:** Recall ≥ 0.75 (Converted class)  
- **Target performance:** F1-score ≥ 0.80 (Converted class)  

These thresholds ensure the model prioritises **business impact over pure accuracy**.

---

### Model Output

- Binary prediction: **Converted / Not Converted**  
- Probability score (0–1)

The probability score allows the business to:

- Rank leads by likelihood of conversion  
- Prioritise outreach efforts  
- Optimise sales resource allocation  

---

### Baseline (Heuristic Comparison)

A naive approach would treat all leads as converters.

- Conversion rate: **87.65%**
- Baseline accuracy: ~88%

However, this approach:

- Provides **no prioritisation capability**
- Fails to identify the **12.35% of non-converting leads**

The ML model adds value by introducing **predictive prioritisation**.

---

### Training Data

- Dataset: Digital Marketing Campaign Dataset  
- Size: 8,000 rows  
- Target: `Conversion` (binary)  
- Features: 15 input variables after cleaning  

To address class imbalance, **SMOTE (Synthetic Minority Oversampling Technique)**  
was applied during training.

---

### Results Achieved

- **Test Recall (Converted):** 0.8395 ✅ (≥ 0.75 requirement)  
- **Test F1-score (Converted):** 0.8767 ✅ (≥ 0.80 target)  
- **ROC-AUC:** 0.7339  
- **Train Accuracy:** 0.8405  
- **Test Accuracy:** 0.7931  

The model successfully meets the defined business performance thresholds.

---

### Model Evaluation

- Strong performance in identifying converting leads (high recall)  
- Balanced precision–recall trade-off (high F1-score)  
- Acceptable generalisation (moderate drop from train to test accuracy)

---

### Limitations

- **Moderate overfitting:**  
  Training accuracy exceeds test accuracy, indicating some loss of generalisation  

- **Minority class performance:**  
  Recall for non-converted leads = 0.4646  
  This reflects the challenge of class imbalance  

- **Model sensitivity:**  
  Performance may vary depending on threshold selection  

---

### Future Improvements

- Adjust classification threshold to optimise business trade-offs  
- Explore additional feature engineering (e.g., interaction terms)  
- Test alternative models (e.g., Gradient Boosting, XGBoost)  
- Improve minority class detection through advanced resampling techniques  

---

### Key ML Terminology

- **Pipeline:** A structured sequence of preprocessing steps and model training  
- **SMOTE:** Generates synthetic samples of the minority class to balance data  
- **Recall:** Ability to correctly identify actual positives  
- **F1-score:** Balance between precision and recall  
- **ROC-AUC:** Measures the model’s ability to distinguish between classes  
- **Feature Importance:** Indicates which variables most influence predictions  
- **Hyperparameter Optimisation:** Process of tuning model parameters to maximise performance

---

## Dashboard Design

The Streamlit dashboard is structured into six interactive pages,
each designed to address a specific business requirement and guide
the user from data exploration to actionable insights.

---

### Page 1 — Project Summary

- Provides an overview of the project and business context (ConvertIQ agency)
- Clearly outlines the three business requirements (BR1, BR2, BR3)
- Displays dataset description with an expandable variable table
- Defines key terminology (Conversion, CTR, ROI, Recall, F1-score)
- Includes links to original data sources

**Business Value:**  
Establishes context and ensures users understand the problem,
data, and objectives before interacting with the analysis.

---

### Page 2 — Customer Behaviour Analysis

- Addresses **BR1 — Customer Behaviour Analysis**
- Interactive Spearman correlation heatmap (Plotly)
- Dynamic box/violin plots showing feature distributions by conversion outcome
- Scatter plot: `TimeOnSite` vs `PagesPerVisit` coloured by conversion
- Bar charts: conversion rate by `CampaignChannel` and `CampaignType`
- Interactive feature selection for exploratory analysis
- Written interpretation provided for each visualisation

**Business Value:**  
Enables identification of key behavioural drivers of conversion,
supporting data-driven targeting and campaign optimisation.

---

### Page 3 — Conversion Predictor

- Addresses **BR2 — Conversion Prediction**
- Interactive input widgets for all 15 model features
- "Predict Conversion" button triggers the trained ML pipeline
- Outputs:
  - Binary prediction (Converted / Not Converted) with visual indicator
  - Probability score displayed via progress bar
- Dynamic interpretation message based on prediction confidence
- Explanation of how the model supports decision-making

**Business Value:**  
Allows the sales team to prioritise leads in real time,
improving efficiency and increasing conversion rates.

---

### Page 4 — Model Performance

- Supports **BR2 — Model Validation**
- Displays key performance metrics:
  - Recall (Converted)
  - F1-score
  - ROC-AUC
- Business performance assessment (meets success criteria or not)
- Confusion matrices (train vs test)
- ROC Curve with AUC score
- Feature importance visualisation
- Hyperparameter tuning summary
- Clear explanation of model limitations

**Business Value:**  
Builds trust in the model by providing transparency,
validation, and clear communication of strengths and limitations.

---

### Page 5 — Campaign ROI Analysis

- Addresses **BR3 — Campaign ROI Intelligence**
- Summary KPIs:
  - Total spend
  - Total revenue
  - ROI
  - Number of orders
- Scatter plot: marketing spend vs revenue by campaign category
- Bar chart: average ROI per category
- Line chart: monthly revenue trends with interactive filters
- Written insights and budget optimisation recommendations

**Business Value:**  
Supports strategic decision-making by identifying
which campaigns deliver the highest return on investment.

---

### Page 6 — Project Hypotheses

- Supports **BR1 and BR2**
- Clearly defined hypotheses with corresponding validation methods
- Statistical results displayed (correlations, p-values, chi-square)
- Each hypothesis confirmed or rejected with evidence
- Visual support:
  - Violin plots (H1)
  - Bar charts (H2)
  - Correlation comparison (H3)
- Summary table consolidating findings
- Business recommendations derived from each result

**Business Value:**  
Demonstrates analytical rigour and ensures that conclusions
are supported by statistical evidence rather than assumptions.

---

## Unfixed Bugs

- **Matplotlib / Seaborn compatibility (Python 3.14):**  
  The ONA development environment initially used Python 3.14, which caused a  
  known `RecursionError` when rendering Seaborn heatmaps.  

  **Resolution:**  
  All visualisations were migrated to **Plotly**, which is fully compatible  
  with Streamlit and provides improved interactivity.  

  This issue is environment-specific and does not affect the deployed application.

---

- **FutureWarning — Pandas Chained Assignment:**  
  The `OrdinalEncoder` from *feature-engine* triggers a `FutureWarning`  
  related to chained assignment in pandas 2.x.

  **Resolution:**  
  This is a known upstream issue and does not impact functionality.  
  Warnings were suppressed in the modelling notebook using:

  ```python
  warnings.filterwarnings("ignore")

---

## Deployment

The app is deployed to **Heroku** using the following configuration:

### Files required for deployment
- `Procfile` — defines the web process:
  `web: sh setup.sh && streamlit run app.py`
- `setup.sh` — configures Streamlit server settings for Heroku
- `.python-version` — specifies Python version: `3.12`
- `requirements.txt` — lists all production dependencies

### Deployment steps
1. Create a new app on [Heroku](https://heroku.com)
2. Connect the GitHub repository to the Heroku app
3. Enable automatic deploys from the `main` branch
4. Set Heroku stack to `heroku-22`
5. Trigger a manual deploy

### Live app
*[Add your Heroku URL here after deployment]*

---

## Main Libraries

| Library | Version | Purpose |
|---|---|---|
| numpy | 2.2.4 | Numerical computations |
| pandas | 2.2.3 | Data manipulation and analysis |
| matplotlib | 3.10.1 | Static plots (used in notebooks) |
| seaborn | 0.13.2 | Statistical visualisations (notebooks) |
| plotly | 5.24.1 | Interactive visualisations in dashboard |
| scikit-learn | 1.6.1 | ML pipeline, preprocessing, RandomizedSearchCV |
| feature-engine | 1.8.3 | OrdinalEncoder for categorical features |
| imbalanced-learn | 0.13.0 | SMOTE for class imbalance |
| joblib | 1.4.2 | Model serialisation (save/load pipeline) |
| streamlit | 1.40.2 | Dashboard web application |
| scipy | 1.15.2 | Statistical tests (chi-square, correlations) |
| kaleido | 0.2.1 | Plotly image export in notebooks |

---

## Credits

### Data Sources
- Dataset 1: [Rabie El Kharoua on Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset)
- Dataset 2: [sinderpreet on Kaggle](https://www.kaggle.com/datasets/sinderpreet/analyze-the-marketing-spending)

### Code References
- Project structure inspired by Code Institute Predictive Analytics
  walkthroughs: Malaria Detector and Churnometer
- Streamlit MultiPage class pattern adapted from Code Institute
  Churnometer walkthrough project

### Tools
- [Streamlit](https://streamlit.io) — dashboard framework
- [Scikit-learn](https://scikit-learn.org) — machine learning
- [Plotly](https://plotly.com) — interactive visualisations
- [Heroku](https://heroku.com) — cloud deployment