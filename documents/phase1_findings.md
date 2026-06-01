# Credit Risk Model - Phase 1

## Dataset

Home Credit Default Risk Dataset

Total Records: 307,511

Target Distribution:
- Non Default (0): 91.9%
- Default (1): 8.1%

Problem Type:
Binary Classification

---

## Engineered Features

Created Features:

- INCOME_CREDIT_RATIO
- ANNUITY_INCOME_RATIO
- CREDIT_GOODS_RATIO

Additional Features:

- DAYS_LAST_PHONE_CHANGE
- DAYS_ID_PUBLISH
- REG_CITY_NOT_WORK_CITY
- REG_CITY_NOT_LIVE_CITY
- FLAG_EMP_PHONE
- FLAG_DOCUMENT_3
- OWN_CAR_AGE

---

## Model

Algorithm:

LightGBM Classifier

Parameters:

- n_estimators = 500
- learning_rate = 0.05
- scale_pos_weight = 11
- random_state = 42

Total Features Used: 21

---

## Performance

ROC-AUC: 0.7516

Best Threshold: 0.67

Accuracy: 0.862

Precision: 0.2566

Recall: 0.3742

F1 Score: 0.3045

---

## Key Drivers of Default Risk

Feature Importance Ranking:

1. EXT_SOURCE_3
2. EXT_SOURCE_2
3. DAYS_BIRTH
4. DAYS_LAST_PHONE_CHANGE
5. EXT_SOURCE_1
6. DAYS_EMPLOYED
7. DAYS_ID_PUBLISH
8. AMT_ANNUITY

These variables had the highest impact on model predictions.

SHAP Analysis confirmed that the following features had the strongest influence on default risk predictions:

- EXT_SOURCE_1
- EXT_SOURCE_2
- EXT_SOURCE_3
- AMT_GOODS_PRICE
- AMT_ANNUITY
- CREDIT_GOODS_RATIO
- DAYS_EMPLOYED

---

## Features Used

- EXT_SOURCE_1
- EXT_SOURCE_2
- EXT_SOURCE_3
- AMT_INCOME_TOTAL
- AMT_CREDIT
- AMT_ANNUITY
- AMT_GOODS_PRICE
- DAYS_BIRTH
- DAYS_EMPLOYED
- REGION_RATING_CLIENT
- REGION_RATING_CLIENT_W_CITY
- INCOME_CREDIT_RATIO
- ANNUITY_INCOME_RATIO
- CREDIT_GOODS_RATIO
- DAYS_LAST_PHONE_CHANGE
- DAYS_ID_PUBLISH
- REG_CITY_NOT_WORK_CITY
- REG_CITY_NOT_LIVE_CITY
- FLAG_EMP_PHONE
- FLAG_DOCUMENT_3
- OWN_CAR_AGE

Total Features Used: 21

---

## Generated Artifacts

- model.pkl
- metrics.json
- feature_names.json
- shap_values.npy

---

## Saved Visualizations

- target_distribution.png
- income_distribution.png
- credit_distribution.png
- default_by_gender.png
- default_by_contract_type.png
- roc_curve.png
- feature_importance.png
- shap_summary.png
- confusion_matrix.png

---

## Summary

A LightGBM-based Credit Risk Prediction model was developed using 21 carefully selected financial, demographic, and engineered features. Class imbalance was handled using scale_pos_weight, resulting in improved identification of default cases.

The final model achieved:

- ROC-AUC: 0.7516
- Accuracy: 86.2%
- Precision: 25.7%
- Recall: 37.4%
- F1 Score: 30.5%

The most influential predictors were external credit bureau scores (EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3), age-related variables, employment history, and financial ratio features.

The trained model and supporting artifacts were saved for deployment in the Credit Risk Prediction Platform.