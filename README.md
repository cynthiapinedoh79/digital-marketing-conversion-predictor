# 📊 Digital Marketing Conversion Predictor

**Live Application:** https://digital-marketing-conversion-p-baa19eafc972.herokuapp.com/

This project is a **Data Analytics and Machine Learning application**
developed to support data-driven decision-making in digital marketing.

The application analyses customer behaviour and campaign data to identify
conversion patterns and predict which leads are most likely to convert.

It combines exploratory data analysis, statistical validation, and a
machine learning model deployed through an interactive Streamlit dashboard.

---

## 🚀 Key Features

- 📈 **Customer Behaviour Analysis:**  
  Identify patterns and correlations between user engagement and conversion  

- 🤖 **Conversion Prediction Model:**  
  Predict conversion likelihood using a trained ML pipeline  

- 💰 **Campaign ROI Analysis:**  
  Evaluate marketing performance and optimise budget allocation  

- 📊 **Interactive Dashboard:**  
  Real-time insights and predictions via Streamlit  

---

## 🧩 Technologies Used

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-blue)
![Heroku](https://img.shields.io/badge/Heroku-Deploy-purple)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Content](#dataset-content)
3. [Business Requirements](#business-requirements)
4. [Agile Planning](#agile-planning)
5. [Data Analysis Overview](#data-analysis-overview)
6. [Feature Engineering](#feature-engineering)
7. [Hypothesis and Validation](#hypothesis-and-validation)
8. [Rationale to Map Business Requirements](#rationale-to-map-business-requirements)
9. [ML Business Case](#ml-business-case)
10. [Ethical Considerations](#ethical-considerations)
11. [Limitations](#limitations)
12. [Dashboard Design](#dashboard-design)
13. [Testing](#testing)
14. [Additional Documentation](#additional-documentation)
15. [Unfixed Bugs](#unfixed-bugs)
16. [Deployment](#deployment)
17. [Main Libraries](#main-libraries)
18. [Credits](#credits)

---

## Project Overview

This project demonstrates an end-to-end data science workflow, from exploratory data analysis to the deployment of a machine learning solution.

It showcases how data science and machine learning can be leveraged to solve real-world business problems and enable data-driven decision-making.

ConvertIQ is a digital marketing agency that aims to improve lead conversion rates and optimise campaign performance through data-driven decision making.

This project analyses customer behaviour and marketing campaign data to identify the key factors that influence whether a lead converts. Additionally, a machine learning model is developed to predict conversion likelihood, enabling the business to prioritise high-value leads.

The project combines exploratory data analysis, statistical validation, and predictive modelling to support three main business goals:

- Identify behavioural patterns associated with conversion  
- Predict which leads are most likely to convert  
- Evaluate campaign performance and return on investment (ROI)  

The final solution is delivered as an interactive Streamlit dashboard, allowing users to explore insights and generate real-time predictions.

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

These variables provide a mix of demographic, behavioural, and campaign-related features, enabling a comprehensive analysis of conversion drivers.

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

## Agile Planning

This project was developed using Agile methodology, focusing on iterative development, continuous validation, and alignment with business requirements.

---

### User Stories

- As a marketing analyst, I want to identify which leads are likely to convert so that I can prioritise high-value prospects.
- As a business stakeholder, I want to understand which campaign attributes influence conversion so that I can optimise marketing strategies.
- As a decision-maker, I want to evaluate campaign performance metrics so that I can allocate budget efficiently.

---

### Acceptance Criteria

- The application must allow users to input lead and campaign data  
- The system must return a prediction of conversion likelihood  
- The dashboard must provide visual insights into customer behaviour and campaign performance  
- The model performance must be evaluated using appropriate metrics (e.g., accuracy, recall, precision)

---

### Development Tracking

Development progress was managed using GitHub Issues and commits, following an iterative workflow aligned with Agile principles.

Full development tracking:  
https://github.com/cynthiapinedoh79/digital-marketing-conversion-predictor

---

## Data Analysis Overview

This section summarises the exploratory data analysis performed to understand customer behaviour and campaign performance.

Key steps included:
- Identifying correlations between behavioural features and conversion
- Comparing distributions of key variables across converted and non-converted leads
- Analysing campaign performance across channels and types
- Validating hypotheses using statistical testing

Detailed analysis and results are presented in the following sections.

### Key Insights

- Engagement features such as **TimeOnSite** and **PagesPerVisit** show the strongest relationship with conversion
- Campaign channel has no statistically significant impact on conversion outcomes
- Behavioural metrics outperform financial variables such as AdSpend in predictive power
- High-engagement users are significantly more likely to convert

These insights directly informed feature selection and model design.

---

## Feature Engineering

The following preprocessing steps were applied:

- Removal of non-informative and confidential variables  
- Encoding of categorical features using OrdinalEncoder  
- Handling of missing values  
- Application of SMOTE to address class imbalance  

These steps improved model performance and ensured better generalisation.

Feature engineering decisions were guided by insights obtained during exploratory data analysis.

Feature engineering was implemented as part of a reusable preprocessing pipeline integrated into the machine learning workflow.

### Summary of Transformations

| Step | Technique | Purpose |
|------|----------|--------|
| Cleaning | Remove confidential variables | Avoid data leakage |
| Encoding | OrdinalEncoder | Convert categorical data |
| Missing Values | Imputation | Ensure model stability |
| Class Imbalance | SMOTE | Improve minority class recall |

---

## Hypothesis and Validation

---

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

---

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
Random Forest Classifier, optimised through RandomizedSearchCV.

Random Forest was selected due to its ability to handle nonlinear relationships, robustness to overfitting, and strong performance with mixed data types (categorical and numerical features).

---

### Model Selection Rationale

Random Forest was chosen due to:

- Its ability to handle complex, non-linear relationships  
- Robustness to noise and overfitting  
- Strong performance with tabular data  
- Interpretability through feature importance  

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

### Business Impact

The model enables ConvertIQ to prioritise high-value leads,
significantly improving sales efficiency and reducing wasted outreach.

By focusing on high-probability conversions, the company can:

- Increase marketing ROI  
- Optimise budget allocation  
- Improve conversion rates  
- Reduce acquisition costs  

This transforms raw data into actionable business intelligence.

---

### Model Performance

The model demonstrates strong predictive capability in identifying converting leads, with a high recall and balanced F1-score. Performance was evaluated on both training and test sets to ensure generalisation.

---

### Model Evaluation

- Strong performance in identifying converting leads (high recall)  
- Balanced precision–recall trade-off (high F1-score)  
- Acceptable generalisation (moderate drop from train to test accuracy)

---

### Results Achieved

- **Test Recall (Converted):** 0.8395 ✅ (≥ 0.75 requirement)  
- **Test F1-score (Converted):** 0.8767 ✅ (≥ 0.80 target)  
- **ROC-AUC:** 0.7339  
- **Train Accuracy:** 0.8405  
- **Test Accuracy:** 0.7931  

The model successfully meets the defined business performance thresholds. 

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

## Ethical Considerations

- The dataset does not include personally identifiable information (PII)  
- Predictions should support, not replace, human decision-making  
- Behavioural data may introduce bias into the model  
- Continuous monitoring is recommended to ensure fairness and reliability  

---

## Limitations

**Model-related limitations:**
- Moderate overfitting: training accuracy exceeds test accuracy  
- Minority class performance: recall for non-converted leads = 0.4646  
- Model sensitivity: performance depends on threshold selection  

**Data-related limitations:**
- Model performance depends on dataset quality and feature representativeness  
- SMOTE may introduce synthetic bias and affect generalisation  

**Deployment limitations:**
- Real-world performance may vary due to unseen data patterns and data drift    

---

## Dashboard Design

The Streamlit dashboard is structured into six interactive pages,
each designed to address a specific business requirement and guide
the user from data exploration to actionable insights.

---

### Dashboard Preview

![Project Summary](docs/screenshots/dashboard/page_summary.png)
![Behaviour Analysis](docs/screenshots/dashboard/page_analysis.png)
![Prediction Page](docs/screenshots/dashboard/page_predictor.png)

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

## Testing

Testing was conducted throughout the development of this project to ensure reliability, correctness, and alignment with business requirements.

---

### Automated Testing

Automated unit tests were not required for this project, as reliability is ensured through model validation and evaluation techniques.

---

### Manual Testing

The application was manually tested across all dashboard pages to validate functionality and user interaction.

| Feature | Test Case | Expected Outcome | Result |
|--------|----------|----------------|--------|
| Navigation | Select each page from sidebar | Correct page loads | ✅ Pass |
| Project Summary | Click README link | Opens GitHub repository | ✅ Pass |
| Behaviour Analysis | Select different features | Plots update correctly | ✅ Pass |
| Conversion Predictor | Input valid data | Prediction returned | ✅ Pass |
| Conversion Predictor | Input extreme values | Model still returns prediction | ✅ Pass |
| Model Performance | Load page | Metrics and plots display correctly | ✅ Pass |
| ROI Analysis | Filter data | Charts update dynamically | ✅ Pass |

---

### Model Testing

The machine learning model was evaluated using a hold-out test set.

| Metric | Result |
|-------|--------|
| Recall (Converted) | 0.8395 |
| F1-score (Converted) | 0.8767 |
| ROC-AUC | 0.7339 |

**Interpretation:**
- The model meets the defined success criteria (Recall ≥ 0.75, F1 ≥ 0.80)
- Strong performance in identifying converting leads
- Lower recall for non-converted class due to class imbalance

---

### Data Validation

- Checked for missing values and handled appropriately  
- Verified correct data types for all features  
- Removed or excluded irrelevant/confidential columns  
- Validated feature distributions before modelling  

---

### Browser Testing

The application was tested in:

- Google Chrome  
- Microsoft Edge  

All core functionalities worked as expected.

---

### Responsiveness

The dashboard layout was tested on different screen sizes.  
Streamlit’s responsive design ensured usability across devices.

---

### Edge Case Testing

| Scenario | Expected Behaviour | Result |
|----------|------------------|--------|
| Missing inputs | Validation message shown | ✅ Pass |
| Extremely high values | Model still predicts | ✅ Pass |
| Invalid categorical values | Handled safely | ✅ Pass |

---

### Bugs and Fixes

All identified bugs were addressed where possible.  
Remaining issues are documented in the **Unfixed Bugs** section.

---


## Additional Documentation

---

### Project Structure

The project follows a clear and well-organised folder structure to separate application logic, data analysis, outputs, and documentation.

The structure is designed to support both data science workflows and production-ready application deployment. It follows best practices by separating data processing, model development, and application logic, ensuring modularity, scalability, and maintainability.

#### Key Components

- `app.py`: main Streamlit entry point  
- `app_pages/`: modular UI pages  
- `jupyter_notebooks/`: exploratory data analysis and model development  
- `inputs/`: raw datasets  
- `outputs/`: trained models and processed data  
- `src/`: reusable data processing and ML logic  
- `docs/`: project documentation and validation evidence  

#### Folder Structure

```bash
project-root/
│
├── app.py
├── app_pages/
├── jupyter_notebooks/
├── inputs/
├── outputs/
├── src/
│
├── docs/
│   └── screenshots/
│       ├── dashboard/
│       ├── python-linter/
│       └── testing/
│
├── .devcontainer/
├── .ona/
├── .venv/              (not committed)
├── .env                (not committed - contains sensitive credentials)
│
├── .gitignore
├── .python-version
├── requirements.txt
├── requirements-dev.txt
├── kaggle.json         (not committed)
├── README.md
```

This structure ensures clear separation between data processing, model development, application logic, and documentation, improving maintainability and scalability.

---

### Code Validation

All Python files in this project were rigorously validated using the  
Code Institute Python Linter:

https://pep8ci.herokuapp.com/

Each file was checked to ensure full compliance with PEP8 standards, improving code readability, consistency, and maintainability.

Screenshots of the validation results are available in the `docs/screenshots/python-linter/` folder.

![Python Linter Validation](docs/screenshots/python-linter/page_summary.png)

**Results:**
- No major errors detected  
- Minor warnings (e.g., line length) were resolved where possible  
- Code follows best practices for readability and structure  

This validation process ensures a clean, professional, and maintainable codebase aligned with industry standards.

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
  ```

---

## Deployment

The app is deployed to **Heroku** using the following configuration:

---

### Environment Variables

No sensitive data is stored in the repository.

Environment variables are used for:
- API keys (if applicable)
- Configuration settings

These are managed securely via Heroku Config Vars.

---

### Files required for deployment
- `Procfile` — defines the web process:
  `web: sh setup.sh && streamlit run app.py`
- `setup.sh` — configures Streamlit server settings for Heroku
- `.python-version` — specifies Python version: `3.12`
- `requirements.txt` — lists all production dependencies

---

### Deployment steps
1. Create a new app on [Heroku](https://heroku.com)
2. Connect the GitHub repository to the Heroku app
3. Enable automatic deploys from the `main` branch
4. Set Heroku stack to `heroku-22`
5. Trigger a manual deploy

---

### Live App

[View Live Application](https://digital-marketing-conversion-p-baa19eafc972.herokuapp.com/)

---

### Notes

- The app is deployed using a production-ready environment
- All dependencies are defined in requirements.txt
- The app has been tested after deployment to ensure full functionality

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

---

### Data Sources
- Dataset 1: [Rabie El Kharoua on Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset)
- Dataset 2: [Sinderpreet on Kaggle](https://www.kaggle.com/datasets/sinderpreet/analyze-the-marketing-spending)

---

### Code References
- Project structure inspired by Code Institute Predictive Analytics walkthroughs:
  - Malaria Detector  
  - Churnometer  
- Streamlit MultiPage class pattern adapted from the Churnometer project

---

### Tools & Technologies
- [Streamlit](https://streamlit.io) — dashboard framework  
- [Scikit-learn](https://scikit-learn.org) — machine learning  
- [Plotly](https://plotly.com) — interactive visualisations  
- [Heroku](https://heroku.com) — cloud deployment

---
