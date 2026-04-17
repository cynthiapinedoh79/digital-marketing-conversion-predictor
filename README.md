# 📊 Digital Marketing Conversion Predictor

**Live Application:** https://digital-marketing-conversion-p-baa19eafc972.herokuapp.com/

This project is a **Data Analytics and Machine Learning application**
designed to support data-driven decision-making in digital marketing,
helping businesses identify high-value leads and optimise campaign ROI.

The application analyses customer behaviour and campaign data to identify
conversion patterns and predict which leads are most likely to convert.

It combines exploratory data analysis, statistical validation, and a
machine learning model deployed through an interactive Streamlit dashboard.

---

## 🚀 Key Features

- 📈 **Customer Behaviour Analysis:**  
  Identify behavioural patterns and correlations associated with conversion

- 🤖 **Conversion Prediction Model:**  
  Predict conversion likelihood using a trained Random Forest pipeline to prioritise high-value leads

- 💰 **Campaign ROI Analysis:**  
  Evaluate marketing efficiency and support budget allocation decisions

- 📱 **Social Media Platform Insight:**  
  Compare platform-level performance within social media campaigns using KPI data

- 📊 **Interactive Dashboard:**  
  Explore real-time insights and predictions through a Streamlit interface

---

## 🚀 How to Use

1. Open the live application or run the app locally
2. Navigate through the dashboard pages using the sidebar
3. Review customer behaviour and campaign ROI insights
4. Use the Conversion Predictor to input a lead profile
5. Generate a prediction and review the recommended business actions
6. Use the model performance and hypothesis pages to understand the reliability of the analysis

The dashboard is designed to be intuitive and accessible for both technical and non-technical users.

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
2. [Target Users](#target-users)
3. [Dataset Content](#dataset-content)
4. [Business Requirements](#business-requirements)
5. [Agile Planning](#agile-planning)
6. [Data Analysis Overview](#data-analysis-overview)
7. [Feature Engineering](#feature-engineering)
8. [Hypothesis and Validation](#hypothesis-and-validation)
9. [Rationale to Map Business Requirements](#rationale-to-map-business-requirements)
10. [ML Business Case](#ml-business-case)
11. [Ethical Considerations](#ethical-considerations)
12. [Limitations](#limitations)
13. [Dashboard Design](#dashboard-design)
14. [Testing](#testing)
15. [Additional Documentation](#additional-documentation)
16. [Unfixed Bugs](#unfixed-bugs)
17. [Deployment](#deployment)
18. [Main Libraries](#main-libraries)
19. [Credits](#credits)
20. [Acknowledgements](#acknowledgements)

---

## Project Overview

**ConvertIQ** is a fictional digital marketing agency that runs campaigns across five channels — Email, SEO, PPC, Social Media, and Referral — for a diverse client base.

### The Problem

Without a data-driven approach, the sales team spends equal time on every lead regardless of conversion likelihood. This creates three measurable business problems:

- **Wasted sales effort:** Representatives contact leads that statistically will not convert
- **Missed revenue:** High-probability leads do not receive timely follow-up
- **Poor ROI visibility:** Budget is allocated across campaigns without understanding which channels and types actually deliver returns

### The Solution

This application provides three capabilities that directly address these problems:

**1. Conversion Prediction (BR2)**
A trained Random Forest model predicts whether a lead will convert based on 15 features. The sales team inputs a lead profile and receives an instant probability score:
- **High-value lead (>75%)** — prioritise immediate direct outreach
- **Potential lead (40–75%)** — send targeted follow-up campaigns to nurture interest
- **Cold lead (<40%)** — deprioritise and move to long-term nurture sequences

**2. Customer Behaviour Analysis (BR1)**
Statistical analysis of 8,000 leads reveals the following actionable guidance for ConvertIQ:

- **Prioritise engagement over spend** — TimeOnSite (r=0.13) and EmailClicks (r=0.13) are stronger predictors than AdSpend (r=0.12). A lead that spends more time on site and clicks emails is more likely to convert than one simply exposed to high ad spend.
- **Focus on content depth** — Converted leads spend 47% more time on site on average. Campaigns should drive leads to explore multiple pages rather than land and leave.
- **Channel does not determine outcome** — Chi-square testing (p=0.594) confirms that Email, SEO, PPC, Social Media and Referral channels produce statistically equivalent conversion rates. ConvertIQ should not reallocate budget based on channel alone.
- **Engagement-based lead scoring** — The combination of EmailOpens, EmailClicks, TimeOnSite and PagesPerVisit provides a reliable behavioural signal. Leads showing all four signals should be fast-tracked to the sales team.

**3. Campaign ROI Intelligence (BR3)**
Analysis of the marketing KPIs dataset provides budget allocation guidance:

- **Instagram outperforms Facebook** with a 43.6% ROI compared to Facebook's -34.1% within social media spend. Budget reallocation from Facebook to Instagram social campaigns is recommended.
- **Search campaigns** generate consistent revenue with controlled spend.
- Monthly revenue trends reveal seasonal patterns that can inform campaign timing decisions.

### Measurable Business Value

| Without the App | With the App |
|---|---|
| Equal time spent on all 8,000 leads | Sales effort focused on predicted converters |
| No visibility into what drives conversion | Ranked behavioural predictors with statistical evidence |
| Budget allocated by assumption | ROI-driven decisions backed by real KPI data |
| Reactive campaign adjustments | Proactive targeting based on engagement signals |
| 87.65% baseline conversion treated equally | High/medium/low lead segmentation with probability scores |

The final solution is delivered as an interactive Streamlit dashboard, enabling marketing analysts, sales managers, and business directors to explore insights and generate real-time predictions without technical expertise.

---

## Target Users

This application is designed for:

- Digital marketing analysts seeking data-driven insights
- Marketing managers optimising campaign budgets
- Sales teams prioritising high-value leads
- Business stakeholders interested in ROI and conversion performance

The tool supports both technical and non-technical users by combining
historical analysis, machine learning predictions, and actionable business insights.

---

## Dataset Content

These datasets allow the project to combine predictive analytics with business performance evaluation.

This project combines two complementary datasets: one for conversion modelling
and customer behaviour analysis, and a second one for campaign ROI and platform-level KPI evaluation.
Together, they support both predictive modelling and business decision-making.

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

This project was developed using Agile methodology, organised into Epics and User Stories tracked via GitHub Issues. All 8 User Stories were completed and closed.

Full development tracking: https://github.com/cynthiapinedoh79/digital-marketing-conversion-predictor/issues

---

### Epic 1 — Information Gathering and Data Collection

**Issue #7 — USER STORY: Data Collection and Initial Inspection**
As a data analyst, I can load and inspect both datasets so that I can confirm they are valid and ready for further processing.

---

### Epic 2 — Data Visualisation, Cleaning and Preparation

**Issue #2 — USER STORY: Customer Behaviour Analysis**
As a marketing analyst, I can explore interactive visualisations of customer behaviour so that I can identify the key drivers of conversion (BR1).

**Issue #6 — USER STORY: Project Hypotheses Validation**
As an evaluator, I can view the project hypotheses and their statistical validation so that I can confirm conclusions are evidence-based.

---

### Epic 3 — Model Training, Optimisation and Validation

**Issue #4 — USER STORY: Model Performance**
As a technical user, I can view model performance metrics, confusion matrices and feature importance so that I can validate the model meets the business success criteria (BR2).

---

### Epic 4 — Dashboard Planning, Design and Development

**Issue #1 — USER STORY: Project Summary Page**
As a non-technical user, I can view a project summary that describes the business context, datasets and requirements so that I can understand the project at a glance.

**Issue #3 — USER STORY: Conversion Predictor**
As a sales manager, I can input a lead profile and receive an instant conversion prediction so that my team can prioritise outreach efficiently (BR2).

**Issue #5 — USER STORY: Campaign ROI Analysis**
As a marketing director, I can analyse campaign ROI across categories so that I can make informed budget allocation decisions (BR3).

**Issue #8 — USER STORY: Social Media Platform Insight**
As a marketing analyst, I can see platform-level social media performance data when Social Media is selected so that I can understand which platform delivers the best ROI.

---

### Epic 5 — Deployment and Release

**Issue #9 — USER STORY: Live Application Access**
As a user, I can access the project dashboard on a live deployed Heroku application so that I can interact with it from any browser.

**Issue #10 — USER STORY: Repository Fork and Clone**
As a technical user, I can follow the README instructions to fork and clone the repository so that I can deploy the project independently.

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

### Business Requirements Mapping

| Business Requirement | Feature | Outcome |
|---------------------|--------|--------|
| BR1 | Customer Behaviour Analysis | Identify key conversion drivers |
| BR2 | Conversion Predictor | Prioritise high-value leads |
| BR3 | ROI Analysis Dashboard | Optimise budget allocation |

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

### Page 1 — Project Summary

- Provides an overview of the project and business context (ConvertIQ agency)
- Clearly outlines the three business requirements (BR1, BR2, BR3)
- Displays dataset description with an expandable variable table
- Defines key terminology (Conversion, CTR, ROI, Recall, F1-score)
- Includes links to original data sources

![Project Summary](docs/screenshots/dashboard/ps-web1.png)
![Project Summary](docs/screenshots/dashboard/ps-web2.png)
![Project Summary](docs/screenshots/dashboard/ps-web2a.png)
![Project Summary](docs/screenshots/dashboard/ps-web2b.png)
![Project Summary](docs/screenshots/dashboard/ps-web3.png)

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

![Customer Behaviour Analysis](docs/screenshots/dashboard/cba-web1.png)
![Customer Behaviour Analysis](docs/screenshots/dashboard/cba-web2.png)
![Customer Behaviour Analysis](docs/screenshots/dashboard/cba-web3.png)
![Customer Behaviour Analysis](docs/screenshots/dashboard/cba-web4.png)

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
- Social Media platform insight displayed when Social Media channel is selected
- Explanation of how the model supports decision-making

**Lead Profile Input Form**
![Conversion Predictor - Input Form](docs/screenshots/dashboard/cp-web0.png)

**Prediction Example 1 — Not Converted (7% probability)**  
Low engagement profile: minimal email interaction, low time on site.
Model correctly identifies this as a **Cold lead** and recommends a nurture strategy.

![Conversion Predictor - Input Form](docs/screenshots/dashboard/cp-web1.png)
![Conversion Predictor - Input Form](docs/screenshots/dashboard/cp-web2.png)

**Prediction Example 2 — Moderate Conversion (72% probability)**  
Mixed engagement profile: some email clicks, moderate time on site.
Model identifies this as a **Potential lead** and recommends targeted follow-up.
Social Media platform insight is displayed showing Instagram vs Facebook ROI.

![Conversion Predictor - Not Converted](docs/screenshots/dashboard/cp-web3.png)
![Conversion Predictor - Not Converted with Social Media Insight](docs/screenshots/dashboard/cp-web4.png)

**Prediction Example 3 — High Conversion (83% probability)**  
Strong engagement profile: high time on site, multiple email clicks, previous purchases.
Model identifies this as a **High-value lead** and recommends immediate direct outreach.

![Conversion Predictor - Moderate](docs/screenshots/dashboard/cp-web5.png)
![Conversion Predictor - Moderate with Social Media](docs/screenshots/dashboard/cp-web6.png)
![Conversion Predictor - Moderate Recommendations](docs/screenshots/dashboard/cp-web7.png)

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

![Model Performance](docs/screenshots/dashboard/mp-web1.png)
![Model Performance](docs/screenshots/dashboard/mp-web2.png)
![Model Performance](docs/screenshots/dashboard/mp-web3.png)
![Model Performance](docs/screenshots/dashboard/mp-web4.png)

**Business Value:**  
Builds trust in the model by providing transparency,
validation, and clear communication of strengths and limitations.

---

### Page 5 — Campaign ROI Analysis

- Addresses **BR3 — Campaign ROI Intelligence**
- Summary KPIs: Total spend, Total revenue, ROI, Number of orders
- Scatter plot: marketing spend vs revenue by campaign category
- Bar chart: average ROI per category
- Daily revenue trend with interactive filters
- Written insights and budget optimisation recommendations

![Campaign ROI Analysis](docs/screenshots/dashboard/cROIa-web1.png)
![Campaign ROI Analysis](docs/screenshots/dashboard/cROIa-web2.png)
![Campaign ROI Analysis](docs/screenshots/dashboard/cROIa-web3.png)

**Business Value:**  
Supports strategic decision-making by identifying
which campaigns deliver the highest return on investment.

---

### Page 6 — Project Hypotheses

- Supports **BR1 and BR2**
- Clearly defined hypotheses with corresponding validation methods
- Statistical results displayed (correlations, p-values, chi-square)
- Each hypothesis confirmed or rejected with evidence
- Summary table consolidating findings
- Business recommendations derived from each result

![Project Hypotheses](docs/screenshots/dashboard/ph-web1.png)
![Project Hypotheses](docs/screenshots/dashboard/ph-web2.png)
![Project Hypotheses](docs/screenshots/dashboard/ph-web3.png)


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

All features were tested against the defined business requirements (BR1, BR2, BR3)
to ensure full alignment between functionality and business objectives.

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

### Lighthouse Testing

The application was tested using Google Lighthouse on Desktop mode.

![Lighthouse Report](docs/screenshots/lighthouse/lighthouse_project_summary.png)

| Category | Score |
|---|---|
| Performance | 83 |
| Accessibility | 88 |
| Best Practices | 100 |
| SEO | 82 |

Performance score is expected to be lower for Streamlit applications due to the framework's rendering approach. A score of 100 for Best Practices confirms the application meets modern web standards.

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

There were no unresolved bugs after manual testing. The following bugs were identified and fixed during development:

**1. Seaborn RecursionError — Python 3.14 Incompatibility**
- Bug: The ONA environment used Python 3.14, which caused a `RecursionError` when rendering Seaborn heatmaps via `ax.set()` with tick parameters.
- Fix: Migrated all visualisations to Plotly, which is fully compatible with Python 3.14 and provides improved interactivity in Streamlit.

**2. Kaleido Version Conflict — Plotly Image Export Failing**
- Bug: `kaleido 1.x` was incompatible with Plotly 5.24.1, causing `.png` exports from notebooks to fail silently.
- Fix: Downgraded kaleido to version `0.2.1`, which is the stable version compatible with Plotly 5.x.

**3. FutureWarning — OrdinalEncoder Chained Assignment**
- Bug: `feature-engine`'s OrdinalEncoder triggered a `FutureWarning` related to chained assignment in pandas 2.x during feature engineering.
- Fix: Suppressed with `warnings.filterwarnings("ignore")` in the modelling notebook. This is a known upstream issue that does not affect functionality.

**4. Virtual Environment Not Found After ONA Container Reset**
- Bug: After ONA restarted the container, the `.venv` kernel was no longer available in VS Code and Python was not recognised.
- Fix: Installed Python 3.12.3 via `apt-get`, recreated the `.venv` virtual environment, reinstalled all dependencies from `requirements.txt`, and reregistered the kernel using `ipykernel install --user`.

**5. Heroku Deployment — Missing `setup.sh` Configuration**
- Bug: Initial Heroku deployment failed because Streamlit required server configuration not present by default.
- Fix: Added `setup.sh` with the required Streamlit server settings and updated `Procfile` to run `sh setup.sh && streamlit run app.py`.

**6. Bar Charts Too Wide in Streamlit — Social Media Insight**
- Bug: Using `st.bar_chart()` produced oversized bars that were not visually proportional when displaying platform comparison data.
- Fix: Replaced `st.bar_chart()` with Plotly `px.bar()` and adjusted `bargap=0.25` for better proportions and readability.

**7. Model File Not Found — Deployment Path Issue**
- Bug: The application failed to load the trained model (`.pkl`) in production due to incorrect relative file paths when deployed on Heroku.
- Fix: Updated the file loading logic to use robust relative paths based on the project root, ensuring compatibility across local and deployment environments.

**8. Probability Threshold Logic — Incorrect Output Classification**
- Bug: All predictions were being classified as low probability due to incorrect threshold logic in the conditional statements.
- Fix: Corrected the conditional structure to properly evaluate probability bands (<0.40, 0.40–0.75, >0.75), ensuring accurate classification and corresponding recommendations.

**9. Streamlit Rendering Order — Layout Inconsistency**
- Bug: Certain UI components (insights and action sections) were rendered in an inconsistent order due to misplaced code blocks within the try/except structure.
- Fix: Reorganised the code to ensure a logical flow: prediction → insight → channel analysis → recommendations.

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
│   ├── multipage.py
│   ├── page_summary.py
│   ├── page_data_analysis.py
│   ├── page_predictor.py
│   ├── page_model_performance.py
│   ├── page_roi_analysis.py
│   └── page_hypothesis.py
│
├── jupyter_notebooks/
│   ├── 01_DataCollection.ipynb
│   ├── 02_DataAnalysis.ipynb
│   ├── 03_DataCleaning.ipynb
│   ├── 04_FeatureEngineering.ipynb
│   └── 05_Modelling.ipynb
│
├── inputs/
│   └── datasets/
│       ├── raw/
│       ├── cleaned/
│       └── featured/
│
├── outputs/
│   └── ml_pipeline/
│       └── v1/
│
├── src/
│
├── docs/
│   └── screenshots/
│       ├── dashboard/
│       ├── python-linter/
│       │   └── app_pages/
│       ├── jsHint/
│       ├── lighthouse/
│       └── testing/
│
├── .devcontainer/
├── .venv/              (not committed)
├── .env                (not committed)
│
├── .gitignore
├── .python-version
├── requirements.txt
├── requirements-dev.txt
├── kaggle.json         (not committed)
├── Procfile
├── setup.sh
└── README.md
```

This structure ensures clear separation between data processing, model development, application logic, and documentation, improving maintainability and scalability.

---

### Code Validation

#### Python — PEP8

All Python files were validated using the Code Institute Python Linter:
https://pep8ci.herokuapp.com/

**`app.py`** — Main Streamlit entry point. Initialises the multipage app and registers all pages.
![app.py](docs/screenshots/python-linter/pl-app.py.png)

**`multipage.py`** — Multipage class that handles page registration and navigation.
![multipage.py](docs/screenshots/python-linter/app_pages/pl-multipage.py.png)

**`page_summary.py`** — Project Summary page. Displays business context, dataset descriptions, key terms and business conclusions.
![page_summary.py](docs/screenshots/python-linter/app_pages/pl-page_summary.py.png)

**`page_data_analysis.py`** — Customer Behaviour Analysis page. Displays correlation heatmap, box plots, scatter plot and conversion rate charts (BR1).
![page_data_analysis.py](docs/screenshots/python-linter/app_pages/pl-page_data_analysis.py.png)

**`page_predictor.py`** — Conversion Predictor page. Loads the trained ML pipeline and returns real-time conversion predictions with probability scores (BR2).
![page_predictor.py](docs/screenshots/python-linter/app_pages/pl-page_predictor.py.png)

**`page_model_performance.py`** — Model Performance page. Displays confusion matrices, ROC curve, feature importance and hyperparameter tuning summary (BR2).
![page_model_performance.py](docs/screenshots/python-linter/app_pages/pl-page_model_performance.py.png)

**`page_roi_analysis.py`** — Campaign ROI Analysis page. Displays spend vs revenue scatter, ROI bar chart and daily revenue trend by category (BR3).
![page_roi_analysis.py](docs/screenshots/python-linter/app_pages/pl-page_roi_analysis.py.png)

**`page_hypothesis.py`** — Project Hypotheses page. Validates three statistical hypotheses using correlation tests and chi-square analysis.
![page_hypothesis.py](docs/screenshots/python-linter/app_pages/pl-page_hypothesis.py.png)

**Results:** No errors detected. Code follows PEP8 standards for readability and maintainability.

---

#### JSON — JSHint

Configuration JSON files were validated using JSHint:
https://jshint.com/

**`devcontainer.json`** — Development container configuration for the ONA/VS Code environment.
![devcontainer.json](docs/screenshots/jsHint/jsHint-devcontainer.json.png)

**`settings.json`** — VS Code workspace settings configuration.
![settings.json](docs/screenshots/jsHint/jsHint-settings.json.png)

**Results:** No errors detected in configuration files.

---

#### Jupyter Notebooks

All data analysis and model development was conducted in Jupyter Notebooks, available in the `jupyter_notebooks/` folder of this repository.

| Notebook | Purpose |
|---|---|
| [01_DataCollection.ipynb](jupyter_notebooks/01_DataCollection.ipynb) | Load and inspect both datasets, confirm validity for further processing |
| [02_DataAnalysis.ipynb](jupyter_notebooks/02_DataAnalysis.ipynb) | Exploratory data analysis, correlation study, hypothesis validation and visualisations (BR1) |
| [03_DataCleaning.ipynb](jupyter_notebooks/03_DataCleaning.ipynb) | Remove irrelevant columns, handle missing values, standardise categorical variables |
| [04_FeatureEngineering.ipynb](jupyter_notebooks/04_FeatureEngineering.ipynb) | Encode categorical features, apply transformations and SMOTE for class imbalance |
| [05_Modelling.ipynb](jupyter_notebooks/05_Modelling.ipynb) | Train Random Forest pipeline, RandomizedSearchCV optimisation, model evaluation and saving (BR2) |

Each notebook was executed in order and outputs are saved to `inputs/datasets/` and `outputs/ml_pipeline/v1/`.

---

## Unfixed Bugs

There are no known unresolved bugs affecting the deployed application at the time of submission.

During development, the following environment-specific considerations were identified:

- **Matplotlib / Seaborn compatibility (Python 3.14):**  
  The ONA development environment caused a `RecursionError` when rendering Seaborn heatmaps.

  **Resolution:**  
  All visualisations were migrated to Plotly, ensuring full compatibility and improved interactivity.

- **FutureWarning — Pandas Chained Assignment:**  
  The `OrdinalEncoder` from *feature-engine* triggers a `FutureWarning`  
  related to chained assignment in pandas 2.x.
  ```python
  warnings.filterwarnings("ignore")
  ```

  **Note:**  
  This is a known upstream library issue and does not affect application functionality or predictions.

These issues are environment-specific and do not impact the deployed application.
No functional or user-impacting issues remain in the deployed application.

---

## Deployment

The app is deployed to **Heroku**.

The deployment uses a Streamlit-compatible Heroku setup with a `Procfile`,
a `setup.sh` configuration script, and a defined Python version to ensure
consistent production behaviour.

**Live App:** [Digital Marketing Conversion Predictor](https://digital-marketing-conversion-p-baa19eafc972.herokuapp.com/)

---

### Files Required for Deployment

**`setup.sh`** — configures Streamlit server settings for Heroku:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

**`Procfile`** — defines the web process:

    web: sh setup.sh && streamlit run app.py

**`.python-version`** — specifies Python version:

    3.12

**`requirements.txt`** — lists all production dependencies.

---

### Deployment Steps

1. Add `setup.sh`, `Procfile`, `.python-version` and `requirements.txt` to the working directory
2. Log in to [Heroku](https://heroku.com) and create a new app
3. At the **Deploy** tab, select **GitHub** as the deployment method
4. Search for your repository name and click **Connect**
5. Select the `main` branch and click **Deploy Branch**
6. Once the build completes, click **Open App** to access the live application
7. If the slug size is too large, add large files not required for the app to a `.slugignore` file
8. Troubleshoot any build errors by reviewing the build log

---

### Forking

To fork this repository:

1. On the main repository page, click **Fork** in the top-right corner
2. Choose the desired owner from the dropdown
3. Optionally rename the repository and add a description
4. Ensure **"Copy the main branch only"** is checked
5. Click **Create fork**

---

### Cloning

To clone this repository locally:

1. On the main repository page, click **Code**
2. Copy the HTTPS URL
3. Open your terminal and navigate to your desired directory
4. Run: `git clone <paste-the-copied-URL>`
5. Press **Enter** to clone the repository

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

## Acknowledgements

- I would like to thank my mentor, **[Mo Shami]**, at Code Institute for their guidance and feedback throughout this project. Their advice on machine learning evaluation and dashboard design provided valuable direction at key stages of development.
- I would like to thank the Code Institute tutor support team for their assistance with environment configuration and dependency issues during development.
- I would like to thank the Code Institute Slack community for their support and shared knowledge throughout the Predictive Analytics module.
- Project structure and methodology were inspired by the Code Institute Predictive Analytics walkthroughs: **Malaria Detector** and **Churnometer**.