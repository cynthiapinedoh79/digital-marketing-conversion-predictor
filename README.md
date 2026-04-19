# 📊 Digital Marketing Conversion Predictor

**Live Application:** https://digital-marketing-conversion-p-baa19eafc972.herokuapp.com/

**Digital Marketing Conversion Predictor** is a data-driven application designed to help marketing teams prioritise leads based on their likelihood to convert. Using machine learning, the system analyses customer demographics, campaign attributes, and engagement behaviour to estimate conversion probability in real time.

👉 **Key Value:** Focus resources on leads most likely to convert, increasing ROI and campaign performance.

<br>

---

## Table of Contents

1. [🏢 Project Overview](#project-overview)
   - [The Problem](#the-problem)
   - [The Solution](#the-solution)
   - [Measurable Business Value](#measurable-business-value)
2. [🚀 Key Features](#key-features)
3. [📖 How to Use](#how-to-use)
4. [🧩 Technologies Used](#technologies-used)
5. [🎯 Target Users](#target-users)
6. [📊 Dataset Content](#dataset-content)
7. [📋 Business Requirements](#business-requirements)
8. [🧩 Agile Planning](#agile-planning)
   - [Epic 1 — Information Gathering and Data Collection](#epic-1--information-gathering-and-data-collection)
   - [Epic 2 — Data Visualisation, Cleaning and Preparation](#epic-2--data-visualisation-cleaning-and-preparation)
   - [Epic 3 — Model Training, Optimisation and Validation](#epic-3--model-training-optimisation-and-validation)
   - [Epic 4 — Dashboard Planning, Design and Development](#epic-4--dashboard-planning-design-and-development)
   - [Epic 5 — Deployment and Release](#epic-5--deployment-and-release)
9. [🔍 Data Analysis Overview](#data-analysis-overview)
10. [⚙️ Feature Engineering](#feature-engineering)
11. [💡 Hypothesis and Validation](#hypothesis-and-validation)
   - [Hypothesis 1 — Engagement Depth Predicts Conversion](#hypothesis-1--engagement-depth-predicts-conversion)
   - [Hypothesis 2 — Campaign Channel Affects Conversion Rate](#hypothesis-2--campaign-channel-affects-conversion-rate)
   - [Hypothesis 3 — Ad Spend Alone is a Weak Predictor](#hypothesis-3--ad-spend-alone-is-a-weak-predictor)
12. [🗺️ Rationale to Map Business Requirements](#rationale-to-map-business-requirements)
13. [🤖 ML Business Case](#ml-business-case)
    - [Learning Method](#learning-method)
    - [Model Selection Rationale](#model-selection-rationale)
    - [Success Metrics](#success-metrics)
    - [Model Output](#model-output)
    - [Baseline Heuristic Comparison](#baseline-heuristic-comparison)
    - [Training Data](#training-data)
    - [Hyperparameter Optimisation](#hyperparameter-optimisation)
    - [Results Achieved](#results-achieved)
    - [Key ML Terminology](#key-ml-terminology)
14. [⚖️ Ethical Considerations](#ethical-considerations)
15. [⚠️ Limitations](#limitations)
16. [🖥️ Dashboard Design](#dashboard-design)
    - [Page 1 — Project Summary](#page-1--project-summary)
    - [Page 2 — Customer Behaviour Analysis](#page-2--customer-behaviour-analysis)
    - [Page 3 — Conversion Predictor](#page-3--conversion-predictor)
    - [Page 4 — Model Performance](#page-4--model-performance)
    - [Page 5 — Campaign ROI Analysis](#page-5--campaign-roi-analysis)
    - [Page 6 — Project Hypotheses](#page-6--project-hypotheses)
17. [🧪 Testing](#testing)
    - [Manual Testing](#manual-testing)
    - [Model Testing](#model-testing)
    - [Data Validation](#data-validation)
    - [Browser Testing](#browser-testing)
    - [Lighthouse Testing](#lighthouse-testing)
    - [Edge Case Testing](#edge-case-testing)
    - [Bugs and Fixes](#bugs-and-fixes)
18. [📁 Additional Documentation](#additional-documentation)
    - [Project Structure](#project-structure)
    - [Code Validation](#code-validation)
    - [Jupyter Notebooks](#jupyter-notebooks)
    - [Data Cleaning and Feature Engineering Spreadsheet](#data-cleaning-and-feature-engineering-spreadsheet)
19. [🐛 Unfixed Bugs](#unfixed-bugs)
20. [🚀 Deployment](#deployment)
    - [Files Required for Deployment](#files-required-for-deployment)
    - [Deployment Steps](#deployment-steps)
    - [Forking](#forking)
    - [Cloning](#cloning)
21. [📚 Main Libraries](#main-libraries)
22. [📌 Conclusion](#conclusion)
23. [🙏 Credits](#credits)
24. [👏 Acknowledgements](#acknowledgements)

<br>

---

## Project Overview

**ConvertIQ** is a fictional digital marketing agency that runs campaigns across five channels — Email, SEO, PPC, Social Media, and Referral — for a diverse client base.

---

### The Problem

Without a data-driven approach, the sales team spends equal time on every lead regardless of conversion likelihood. This creates three measurable business problems:

- **Wasted sales effort:** Representatives contact leads that statistically will not convert
- **Missed revenue:** High-probability leads do not receive timely follow-up
- **Poor ROI visibility:** Budget is allocated across campaigns without understanding which channels and types actually deliver returns

---

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

---

### Measurable Business Value

| Without the App | With the App |
|---|---|
| Equal time spent on all 8,000 leads | Sales effort focused on predicted converters |
| No visibility into what drives conversion | Ranked behavioural predictors with statistical evidence |
| Budget allocated by assumption | ROI-driven decisions backed by real KPI data |
| Reactive campaign adjustments | Proactive targeting based on engagement signals |
| 87.65% baseline conversion treated equally | High/medium/low lead segmentation with probability scores |

The final solution is delivered as an interactive Streamlit dashboard, enabling marketing analysts, sales managers, and business directors to explore insights and generate real-time predictions without technical expertise.

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Key Features

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

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## How to Use

1. Open the live application or run the app locally
2. Navigate through the dashboard pages using the sidebar
3. Review customer behaviour and campaign ROI insights
4. Use the Conversion Predictor to input a lead profile
5. Generate a prediction and review the recommended business actions
6. Use the model performance and hypothesis pages to understand the reliability of the analysis

The dashboard is designed to be intuitive and accessible for both technical and non-technical users.

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-blue)
![Heroku](https://img.shields.io/badge/Heroku-Deploy-purple)

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Target Users

This application is designed for:

- Digital marketing analysts seeking data-driven insights
- Marketing managers optimising campaign budgets
- Sales teams prioritising high-value leads
- Business stakeholders interested in ROI and conversion performance

The tool supports both technical and non-technical users by combining
historical analysis, machine learning predictions, and actionable business insights.

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Dataset Content

These datasets allow the project to combine predictive analytics with business performance evaluation.

This project combines two complementary datasets: one for conversion modelling
and customer behaviour analysis, and a second one for campaign ROI and platform-level KPI evaluation.
Together, they support both predictive modelling and business decision-making.

The project uses two datasets sourced from Kaggle:

<br>

---

**Dataset 1 — Predict Conversion in Digital Marketing Dataset**
Source: [Kaggle — Rabie El Kharoua](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset)  
License: CC BY 4.0 — Synthetic dataset created for educational purposes  
- 8,000 rows and 20 columns (16 after cleaning)
- Each row represents a unique lead exposed to a digital marketing campaign
- Includes demographic data, marketing metrics, customer engagement indicators and historical purchase data
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

**Important note on engagement variables:**  
Metrics such as `EmailOpens`, `EmailClicks`, and `WebsiteVisits` are recorded as cumulative values per lead in the source dataset. The dataset does not define a fixed time window (e.g. daily, weekly, or monthly). For business interpretation, these variables should therefore be understood as lead-level engagement summaries across the campaign journey. In practice, they are best used within a consistent operational timeframe, such as the first 2–4 weeks after lead acquisition.

<br>

---

**Dataset 2 — Digital Marketing Metrics & KPIs (SQL)**
Source: [Kaggle — Sinderpreet](https://www.kaggle.com/datasets/sinderpreet/analyze-the-marketing-spending)  
- 308 rows and 11 columns (13 after cleaning)
- Daily campaign performance metrics per campaign across 4 categories: social, search, influencer and media
- Campaigns include: Facebook Tier 1/2, Instagram Tier 1/2, Google Hot/Wide, YouTube Blogger, Instagram Blogger, Banner Partner
- Used for supplementary ROI and campaign performance analysis (BR3)

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

[🔝 Back to Table of Contents](#table-of-contents)

<br>

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

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Agile Planning

This project was developed using Agile methodology, organised into Epics and User Stories tracked via GitHub Issues. All 10 User Stories were completed and closed.

Full development tracking: https://github.com/cynthiapinedoh79/digital-marketing-conversion-predictor/issues

---

### Epic 1 — Information Gathering and Data Collection

**Issue #7 — USER STORY: Data Collection and Initial Inspection**
As a data practitioner, I want to load and inspect both datasets from the raw inputs folder so that I can confirm they are valid and understand their structure before analysis.

**Acceptance Criteria:**
- AC1: Both datasets load without errors
- AC2: Shape, dtypes and missing values are documented
- AC3: Target variable distribution is confirmed
- AC4: Confidential columns are identified for removal
- AC5: Conclusions documented in notebook

<br>

---

### Epic 2 — Data Visualisation, Cleaning and Preparation

**Issue #2 — USER STORY: Customer Behaviour Analysis**
As a marketing analyst, I want to see which customer and campaign variables correlate most with conversion so that I can identify the characteristics of a high-quality lead.

**Acceptance Criteria:**
- AC1: Correlation heatmap shows relationships between all numeric features and Conversion
- AC2: Box plots compare key features between converted and non-converted leads
- AC3: Scatter plot shows TimeOnSite vs PagesPerVisit coloured by conversion outcome
- AC4: Bar charts show conversion rate per CampaignChannel and CampaignType
- AC5: Written conclusions explain each visualisation

---

**Issue #6 — USER STORY: Project Hypotheses Validation**
As an evaluator reviewing this project, I want to see the project hypotheses and their statistical validation so that I can confirm the analytical conclusions are evidence-based.

**Acceptance Criteria:**
- AC1: At least 3 hypotheses are stated clearly
- AC2: Each hypothesis has a validation method described
- AC3: Statistical results (p-value, correlation) are shown
- AC4: Each hypothesis is confirmed or rejected with evidence
- AC5: Recommended actions follow from each conclusion

<br>

---

### Epic 3 — Model Training, Optimisation and Validation

**Issue #4 — USER STORY: Model Performance**
As a data practitioner, I want to see how well the ML model performs so that I can confirm it meets the business success criteria.

**Acceptance Criteria:**
- AC1: Confusion matrix shown for both train and test sets
- AC2: Classification report table displayed
- AC3: ROC-AUC curve shown for test set
- AC4: Feature importance chart displayed
- AC5: Clear statement confirms if model meets success metrics

<br>

---

### Epic 4 — Dashboard Planning, Design and Development

**Issue #1 — USER STORY: Project Summary Page**
As a data practitioner reviewing this project, I want to see a clear project summary page in the dashboard so that I can quickly understand the business context, dataset, and goals.

**Acceptance Criteria:**
- AC1: Dashboard displays project title and business context
- AC2: Dataset description with variable table is shown
- AC3: Business requirements (BR1, BR2, BR3) are listed clearly
- AC4: Key terms and jargon are defined
- AC5: Links to data sources are included

---

**Issue #3 — USER STORY: Conversion Predictor**
As a sales manager, I want to input a lead's profile and receive an instant prediction so that my team can prioritise follow-up with the most likely converters.

**Acceptance Criteria:**
- AC1: Dashboard has input widgets for all 15 lead features
- AC2: A Predict button triggers the ML pipeline
- AC3: Output shows Converted / Not Converted label
- AC4: Output shows probability score as a progress bar
- AC5: Interpretation text explains what the prediction means

---

**Issue #5 — USER STORY: Campaign ROI Analysis**
As a marketing director, I want to see how marketing spend translates into revenue by campaign category so that I can make informed budget allocation decisions.

**Acceptance Criteria:**
- AC1: Summary KPIs show total spend, revenue and overall ROI
- AC2: Scatter plot shows spend vs revenue by campaign category
- AC3: Bar chart shows average ROI per category
- AC4: Line chart shows daily revenue trend per category
- AC5: Conclusions recommend best-performing channel

---

**Issue #8 — USER STORY: Social Media Platform Insight**
As a marketing analyst, I want to see platform-level performance data when a Social Media campaign is selected so that I can understand which platform delivers the best ROI and prioritise budget accordingly.

**Acceptance Criteria:**
- AC1: Social Media insight section only appears when Campaign Channel is set to "Social Media"
- AC2: Orders by Platform bar chart displays real data from the KPI dataset
- AC3: ROI (%) by Platform bar chart displays calculated ROI per platform
- AC4: A summary table shows Orders, Revenue, Spent and ROI per platform
- AC5: Interpretive text dynamically reflects which platform outperforms the other

<br>

---

### Epic 5 — Deployment and Release

**Issue #9 — USER STORY: Live Application Access**
As a user, I can access the project dashboard on a live deployed Heroku application so that I can interact with it from any browser.

**Acceptance Criteria:**
- AC1: Application is accessible via a public Heroku URL
- AC2: All six dashboard pages load correctly
- AC3: App is tested after deployment to confirm full functionality

---

**Issue #10 — USER STORY: Repository Fork and Clone**
As a technical user, I can follow the README instructions to fork and clone the repository so that I can deploy the project independently.

**Acceptance Criteria:**
- AC1: README includes clear forking instructions
- AC2: README includes clear cloning instructions
- AC3: Repository is public and accessible

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Data Analysis Overview

This section summarises the exploratory data analysis performed to understand customer behaviour and campaign performance.

Key steps included:
- Identifying correlations between behavioural features and conversion
- Comparing distributions of key variables across converted and non-converted leads
- Analysing campaign performance across channels and types
- Validating hypotheses using statistical testing

Detailed analysis and results are presented in the following sections.

---

### Key Insights

- Engagement features such as **TimeOnSite** and **PagesPerVisit** show the strongest relationship with conversion
- Campaign channel has no statistically significant impact on conversion outcomes
- Behavioural metrics outperform financial variables such as AdSpend in predictive power
- High-engagement users are significantly more likely to convert

These insights directly informed feature selection and model design.

[🔝 Back to Table of Contents](#table-of-contents)

<br>

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

---

### Summary of Transformations

| Step | Technique | Purpose |
|------|----------|--------|
| Cleaning | Remove confidential variables | Avoid data leakage |
| Encoding | OrdinalEncoder | Convert categorical data |
| Missing Values | Imputation | Ensure model stability |
| Class Imbalance | SMOTE | Improve minority class recall |

[🔝 Back to Table of Contents](#table-of-contents)

<br>

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

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Rationale to Map Business Requirements

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

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## ML Business Case

### Conversion Classifier

**Objective:**
Develop a binary classification model to predict whether a lead will
convert (`Conversion = 1`) based on demographic and behavioural features.
This directly addresses **Business Requirement 2 (Conversion Prediction)**.

---

### Learning Method

The **learning method** is supervised machine learning — specifically a
**binary classification task** using a **Random Forest Classifier**,
optimised through **RandomizedSearchCV**.

Random Forest was selected due to its ability to handle nonlinear
relationships, robustness to overfitting, and strong performance with
mixed data types (categorical and numerical **features**).

---

### Model Selection Rationale

Random Forest was chosen as the **machine learning task** for this
classification problem due to:

- Its ability to handle complex, non-linear relationships between **features** and the **target**
- Robustness to noise and overfitting via ensemble learning
- Strong performance with tabular data containing both numerical and categorical **attributes**
- Interpretability through **feature importance** scores

---

### Ideal Outcome

A model that:

- Maximises **recall** for the positive **label** (Converted) to capture
  as many potential customers as possible
- Maintains a strong balance between precision and recall
- Enables the sales team to prioritise high-probability leads efficiently

---

### Success Metrics

The following **model metrics** define success:

- **Minimum requirement:** Recall ≥ 0.75 for the positive **label** (Converted class)
- **Target performance:** F1-score ≥ 0.80 for the positive **label** (Converted class)

These thresholds ensure the model prioritises **business impact over pure accuracy**.

---

### Model Output

After training and fitting the model on the training data, the **model output** is:

- A binary **prediction**: **Converted / Not Converted** (the **label**)
- A probability score (0–1) representing the likelihood of conversion

The probability score allows the business to:

- Rank leads by predicted conversion likelihood
- Prioritise outreach efforts based on **model predictions**
- Optimise sales resource allocation

---

### Baseline (Heuristic Comparison)

A naive heuristic approach would treat all leads as converters.

- Conversion rate: **87.65%**
- Baseline accuracy: ~88%

However, this approach:

- Provides **no prioritisation capability**
- Fails to identify the **12.35% of non-converting leads**

The ML model adds value by introducing **predictive prioritisation** beyond the baseline heuristic.

---

### Training Data

The model was **trained** and **fitted** on the following data:

- Dataset: Digital Marketing Campaign Dataset
- Size: 8,000 rows (6,400 train / 1,600 test — 80/20 stratified split)
- **Target variable**: `Conversion` (binary — 1 = Converted, 0 = Not Converted)
- **Features**: 15 input **variables** after cleaning
- **Labels**: 2 classes — Converted (1) and Not Converted (0)

To address class imbalance in the **training** data, **SMOTE** was applied
to oversample the minority **label** during model fitting.

---

### Hyperparameter Optimisation

**RandomizedSearchCV** was used to tune 7 **hyperparameters** with 3+
values each across 30 iterations and 5-fold cross-validation, optimising
for F1-score on the positive **label**:

| Hyperparameter | Values | Rationale |
|---|---|---|
| n_estimators | 100, 200, 300 | More trees improve stability; diminishing returns above 300 |
| max_depth | 4, 6, 8, 10 | Limits tree depth to prevent memorisation of training data |
| min_samples_split | 10, 20, 30 | Higher values prevent splitting on noise |
| min_samples_leaf | 5, 10, 15 | Ensures meaningful leaf nodes, reduces overfitting |
| max_features | sqrt, log2, 0.5 | Controls feature randomness per split |
| class_weight | balanced, balanced_subsample, None | Addresses class imbalance in the **target** |
| criterion | gini, entropy, log_loss | Tests impurity measures for split quality |

---

### Results Achieved

After **fitting** the pipeline on the training set and evaluating **predictions**
on the held-out test set, the following **model metrics** were recorded:

- **Test Recall (Converted label):** 0.8381 ✅ (≥ 0.75 requirement)
- **Test F1-score (Converted label):** 0.8762 ✅ (≥ 0.80 target)
- **ROC-AUC:** 0.7331
- **Train Accuracy:** 0.8528
- **Test Accuracy:** 0.7925

The **model** successfully meets the defined business performance thresholds.

---

### Business Impact

The trained **model** enables ConvertIQ to prioritise high-value leads,
significantly improving sales efficiency and reducing wasted outreach.

By focusing on high-probability **predictions**, the company can:

- Increase marketing ROI
- Optimise budget allocation
- Improve conversion rates
- Reduce acquisition costs

---

### Future Improvements

- Adjust classification threshold to optimise the precision/recall trade-off
- Explore additional **feature** engineering (e.g., interaction terms)
- **Train** alternative models (e.g., Gradient Boosting, XGBoost)
- Improve minority **label** detection through advanced resampling techniques

---

### Key ML Terminology

- **Pipeline:** A structured sequence of preprocessing steps and model training
- **Features / Variables / Attributes:** The 15 input columns used to train the model
- **Target:** The `Conversion` column — what the model is trained to predict
- **Labels:** The two output classes — Converted (1) and Not Converted (0)
- **Train / Fit:** The process of exposing the model to training data to learn patterns
- **Prediction:** The model output for a given set of input features
- **Model Metrics:** Quantitative measures of model performance (Recall, F1, ROC-AUC)
- **SMOTE:** Generates synthetic samples of the minority label to balance training data
- **Recall:** Proportion of actual positive labels correctly identified
- **F1-score:** Harmonic mean of precision and recall
- **ROC-AUC:** Measures the model's ability to distinguish between labels
- **Feature Importance:** Indicates which variables most influence predictions
- **Hyperparameter Optimisation:** Tuning model parameters to maximise performance metrics

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Ethical Considerations

- The dataset does not include personally identifiable information (PII)  
- Predictions should support, not replace, human decision-making  
- Behavioural data may introduce bias into the model  
- Continuous monitoring is recommended to ensure fairness and reliability  

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Limitations

### Model Limitations
- **Moderate overfitting:** train accuracy (0.8528) exceeds test accuracy (0.7925), indicating some loss of generalisation on unseen data
- **Minority class performance:** recall for Not Converted = 0.4697, reflecting the challenge of class imbalance despite SMOTE
- **Threshold sensitivity:** model performance may vary depending on the classification threshold selected

---

### Data Limitations
- Model performance depends on dataset quality and feature representativeness
- SMOTE generates synthetic samples which may introduce bias and affect real-world generalisation
- Engagement metrics (EmailOpens, EmailClicks, WebsiteVisits) lack a defined time window, limiting direct business comparability

---

### Deployment Limitations
- Real-world performance may vary due to unseen data patterns and distribution shifts over time
- The model was trained on a static dataset and may require periodic retraining as campaign behaviour evolves
- Predictions should support, not replace, human decision-making 

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Dashboard Design

The Streamlit dashboard is structured into six interactive pages,
each designed to address a specific business requirement and guide
the user from data exploration to actionable insights.

<br>

---

### Page 1 — Project Summary

- Provides an overview of the project and business context (ConvertIQ agency)
- Clearly outlines the three business requirements (BR1, BR2, BR3)
- Displays dataset description with an expandable variable table
- Defines key terminology (Conversion, CTR, ROI, Recall, F1-score)
- Includes links to original data sources

---

![Project Summary](docs/screenshots/dashboard/ps-web1.png)

---

![Project Summary](docs/screenshots/dashboard/ps-web2.png)

![Project Summary](docs/screenshots/dashboard/ps-web2a.png)

![Project Summary](docs/screenshots/dashboard/ps-web2b.png)

---

![Project Summary](docs/screenshots/dashboard/ps-web3.png)

**Business Value:**  
This page satisfies the project's overall context requirement by establishing
the business problem, datasets and requirements before the user interacts
with any analysis. It ensures evaluators and non-technical users can
understand the purpose of the dashboard at a glance.

<br>

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

---

![Customer Behaviour Analysis](docs/screenshots/dashboard/cba-web2a.png)

![Customer Behaviour Analysis](docs/screenshots/dashboard/cba-web2b.png)

---

![Customer Behaviour Analysis](docs/screenshots/dashboard/cba-web3.png)

![Customer Behaviour Analysis](docs/screenshots/dashboard/cba-web4.png)

**Business Value:**  
This page directly satisfies **BR1** because it provides data visualisations
of the variables most correlated with conversion, enabling marketing analysts
to identify high-quality leads based on statistical evidence rather than assumptions.

<br>

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

---

**Lead Profile Input Form**

![Conversion Predictor - Input Form](docs/screenshots/dashboard/cp-web0.png)

---

**Prediction Example 1 — Not Converted (7% probability)**  
Low engagement profile: minimal email interaction, low time on site.
Model correctly identifies this as a **Cold lead** and recommends a nurture strategy.

![Conversion Predictor - Input Form](docs/screenshots/dashboard/cp-web1.png)

![Conversion Predictor - Input Form](docs/screenshots/dashboard/cp-web2.png)

![Conversion Predictor - Input Form](docs/screenshots/dashboard/cp-web3.png)

![Conversion Predictor - Input Form](docs/screenshots/dashboard/cp-web4.png)

---

**Prediction Example 2 — Moderate Conversion (73% probability)**  
Mixed engagement profile: some email clicks, moderate time on site.
Model identifies this as a **Potential lead** and recommends targeted follow-up.

![Conversion Predictor - Input Form](docs/screenshots/dashboard/cp-web5.png)

![Conversion Predictor - Not Converted](docs/screenshots/dashboard/cp-web6.png)

![Conversion Predictor - Not Converted](docs/screenshots/dashboard/cp-web7.png)

---

**Prediction Example 3 — High Conversion (85% probability)**  
Strong engagement profile: high time on site, multiple email clicks, previous purchases.
Model identifies this as a **High-value lead** and recommends immediate direct outreach.
Social Media platform insight is displayed showing Instagram vs Facebook ROI.

![Conversion Predictor - Not Converted](docs/screenshots/dashboard/cp-web8.png)

![Conversion Predictor - Moderate](docs/screenshots/dashboard/cp-web9.png)

![Conversion Predictor - Moderate with Social Media](docs/screenshots/dashboard/cp-web10.png)

![Conversion Predictor - Not Converted with Social Media Insight](docs/screenshots/dashboard/cp-web11.png)

---

**Business Value:**  
This page directly satisfies **BR2** because it deploys the trained ML pipeline
as an interactive tool, enabling the sales team to input a lead profile and
receive an instant conversion prediction with probability score and recommended action.

<br>

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

![Model Performance](docs/screenshots/dashboard/mp-web5.png)

![Model Performance](docs/screenshots/dashboard/mp-web6.png)

**Business Value:**  
This page supports **BR2** by providing full transparency into the ML model's
performance, confirming it meets the defined success criteria (Recall ≥ 0.75,
F1 ≥ 0.80), and communicating its limitations clearly to build user trust.

<br>

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
This page directly satisfies **BR3** because it analyses marketing spend
efficiency across campaign categories, enabling budget owners to make
ROI-driven allocation decisions backed by real KPI data.

<br>

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
This page supports **BR1 and BR2** by validating three project hypotheses
with statistical evidence, ensuring that conclusions about conversion drivers
are grounded in data rather than assumptions.

[🔝 Back to Table of Contents](#table-of-contents)

<br>

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
| Recall (Converted) | 0.8381 |
| F1-score (Converted) | 0.8762 |
| ROC-AUC | 0.7331 |

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

<br>

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

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Additional Documentation

### Project Structure

The project follows a clear and well-organised folder structure to separate application logic, data analysis, outputs, and documentation.

The structure is designed to support both data science workflows and production-ready application deployment. It follows best practices by separating data processing, model development, and application logic, ensuring modularity, scalability, and maintainability.

---

#### Key Components

- `app.py`: main Streamlit entry point  
- `app_pages/`: modular UI pages  
- `jupyter_notebooks/`: exploratory data analysis and model development  
- `inputs/`: raw datasets  
- `outputs/`: trained models and processed data  
- `src/`: reusable data processing and ML logic  
- `docs/`: project documentation and validation evidence  

---

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

<br>

---

### Code Validation

#### Python — PEP8

All Python files were validated using the Code Institute Python Linter:
https://pep8ci.herokuapp.com/

**`app.py`** — Main Streamlit entry point. Initialises the multipage app and registers all pages.

![app.py](docs/screenshots/python-linter/pl-app.py.png)

---

**`multipage.py`** — Multipage class that handles page registration and navigation.

![multipage.py](docs/screenshots/python-linter/app_pages/pl-multipage.py.png)

---

**`page_summary.py`** — Project Summary page. Displays business context, dataset descriptions, key terms and business conclusions.

![page_summary.py](docs/screenshots/python-linter/app_pages/pl-page_summary.py.png)

---

**`page_data_analysis.py`** — Customer Behaviour Analysis page. Displays correlation heatmap, box plots, scatter plot and conversion rate charts (BR1).

![page_data_analysis.py](docs/screenshots/python-linter/app_pages/pl-page_data_analysis.py.png)

---

**`page_predictor.py`** — Conversion Predictor page. Loads the trained ML pipeline and returns real-time conversion predictions with probability scores (BR2).

![page_predictor.py](docs/screenshots/python-linter/app_pages/pl-page_predictor.py.png)

---

**`page_model_performance.py`** — Model Performance page. Displays confusion matrices, ROC curve, feature importance and hyperparameter tuning summary (BR2).

![page_model_performance.py](docs/screenshots/python-linter/app_pages/pl-page_model_performance.py.png)

---

**`page_roi_analysis.py`** — Campaign ROI Analysis page. Displays spend vs revenue scatter, ROI bar chart and daily revenue trend by category (BR3).

![page_roi_analysis.py](docs/screenshots/python-linter/app_pages/pl-page_roi_analysis.py.png)

---

**`page_hypothesis.py`** — Project Hypotheses page. Validates three statistical hypotheses using correlation tests and chi-square analysis.

![page_hypothesis.py](docs/screenshots/python-linter/app_pages/pl-page_hypothesis.py.png)

**Results:** No errors detected. Code follows PEP8 standards for readability and maintainability.

<br>

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

#### Data Cleaning and Feature Engineering Spreadsheet

As part of the data preparation workflow, a spreadsheet was created to document all variables, their data types, missing value analysis, and the feature engineering transformations considered for each variable.

**Data Cleaning**

![Data Cleaning](docs/screenshots/data_cleaning_spreadsheet.png)

---

**Feature Engineering**

![Feature Engineering](docs/screenshots/feature_engineering_spreadsheet.png)

[📊 Download Spreadsheet](docs/data_cleaning_and_feature_engineering.xlsx)

[🔝 Back to Table of Contents](#table-of-contents)

<br>

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

[🔝 Back to Table of Contents](#table-of-contents)

<br>

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

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Main Libraries

| Library | Version | Purpose |
|---|---|---|
| numpy | 2.2.4 | Numerical computations in data analysis and feature engineering notebooks |
| pandas | 2.2.3 | Data loading, cleaning, manipulation and feature engineering across all notebooks |
| matplotlib | 3.10.1 | Static plot rendering backend used in notebooks |
| seaborn | 0.13.2 | Statistical visualisations in data analysis notebook (box plots, violin plots) |
| plotly | 5.24.1 | Interactive visualisations in all dashboard pages and notebook exports |
| scikit-learn | 1.6.1 | ML pipeline, RandomForestClassifier, RandomizedSearchCV, confusion matrix, ROC-AUC |
| feature-engine | 1.8.3 | OrdinalEncoder for categorical feature encoding in the ML pipeline |
| imbalanced-learn | 0.13.0 | SMOTE oversampling to address class imbalance during model training |
| joblib | 1.4.2 | Saving and loading the trained ML pipeline (.pkl files) |
| streamlit | 1.40.2 | Interactive dashboard web application with 6 pages |
| scipy | 1.15.2 | Statistical tests — point-biserial correlation and chi-square hypothesis validation |
| kaleido | 0.2.1 | Plotly static image export (.png) for confusion matrix and ROC curve in notebooks |
| nbformat | ≥4.2.0 | Jupyter notebook format support for rendering plots in notebooks |
| openpyxl | 3.1.5 | Excel file creation for data cleaning and feature engineering spreadsheet |

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Conclusion

This project demonstrates how machine learning can be applied to real-world business scenarios, transforming raw data into actionable insights that drive measurable outcomes. By combining exploratory data analysis, statistical validation, and a deployed ML pipeline, ConvertIQ can now prioritise high-value leads, optimise campaign spend, and make data-driven decisions in real time.

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Credits

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

[🔝 Back to Table of Contents](#table-of-contents)

<br>

---

## Acknowledgements

- I would like to thank my mentor, **Mo Shami**, at Code Institute for their guidance and feedback throughout this project. Their advice on machine learning evaluation and dashboard design provided valuable direction at key stages of development.
- I would like to thank the Code Institute tutor support team for their assistance with environment configuration and dependency issues during development.
- I would like to thank the Code Institute Slack community for their support and shared knowledge throughout the Predictive Analytics module.
- Project structure and methodology were inspired by the Code Institute Predictive Analytics walkthroughs: **Malaria Detector** and **Churnometer**.

[🔝 Back to Table of Contents](#table-of-contents)