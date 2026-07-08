# house-price-prediction
End-to-end house price prediction pipeline in Python. Features data preprocessing (imputation, encoding, outlier removal), statistical EDA to extract market trends, and predictive Machine Learning regression modeling to accurately forecast property values.

# Objective
The goal of this project is to build an end-to-end data science pipeline that analyzes urban housing market dynamics and predicts residential property values. By leveraging Pandas for rigorous data preprocessing and Exploratory Data Analysis (EDA), the project aims to uncover the primary structural and environmental drivers behind real estate pricing. Ultimately, these insights are used to train, evaluate, and optimize predictive Machine Learning regression models capable of forecasting property values with high accuracy.


# Data Cleaning
1. Standardized column names and data types across the dataset.
2. Handled missing data via targeted row deletion and logical feature imputation.
3. Encoded text-based categorical variables into proper boolean types.
4. Isolated column-wise statistical outliers and purged duplicate records.

# Exploratory Data Analysis (EDA)
### Univariate Analysis:
### Bivariate Analysis:
### Multivariate Analysis
# Machine Learning

# Key Insights
1. There is a clear positive correlation between longer wait times and higher no-show rates. Patients forced to wait several weeks are highly prone to missing their appointments.
2. Patients managing chronic conditions (Group 1) have a visibly lower no-show: show ratio than healthier patients (Group 0). This suggests that the ongoing necessity of disease management acts as a protective factor against missed appointments, making these patients inherently more reliable.
3. The 'Retired' cohort demonstrated the lowest proportional no-show rate of any employment category. This indicates that schedule flexibility and healthcare prioritization lead to superior appointment adherence for this group.
4. Looking strictly at absolute counts, the shape of the no-show distribution identically mirrors the overall attendance volume. Missed appointments essentially scale proportionally with the baseline age demographics, requiring deeper normalized percentage calculations to isolate specific age-based risks.
5. Age and Gender differences are not strong predictors for whether a patient would show up or not.
6. Low to middle-class-income people are more likely to show higher no_show rates.
7. Some patients rarely miss an appointment. If the patient has a previous history of no_shows he/she is likely to miss more appointments.
8. According to this specific dataset, previous SMS and reminder calls are not strong predictors of whether a patient will show up or not.

# Images
### Chronic Disease vs No Show
![Chronic Disease vs No Show](images/chronic_disease_vs_no_show.png)
* Patients with Chronic Diseases are more likely to show up in the follow dates.
---
### Demographics vs No Show
![Demographics vs No Show](images/demographics_vs_no_show.png)
* **Class Imbalance:** The density plots (diagonal) clearly illustrate a significant imbalance, with the vast majority of patients attending their appointments (Class 0).
* **Feature Relationships:** The scatter distributions between `age`, `gender`, and `children_count` do not reveal strong linear correlations or distinct clusters that isolate the no-show instances (Class 1), suggesting that behavioral or clinical factors may be stronger predictors than basic demographics alone.
