# Fraud Detection using Machine Learning in R

## Project Overview
This project focuses on detecting fraudulent financial transactions using machine learning in R. The goal is to build a model that can identify fraud accurately, even when fraudulent transactions are rare.

The dataset used is PaySim, which simulates real-world mobile money transactions. This makes the project practical and relevant to real financial systems.

---

## Objectives
- Detect fraudulent transactions
- Handle highly imbalanced data
- Create meaningful features from raw data
- Build a reliable machine learning model

---

## Approach

### 1. Data Preparation
- Loaded the dataset using data.table
- Reduced dataset size to improve performance
- Converted categorical variables into numeric form

### 2. Feature Engineering
Feature engineering was applied to improve model performance:
- Converted transaction types into numeric values
- Used balance-related variables to capture suspicious patterns
- Created features that reflect transaction behavior

### 3. Handling Imbalanced Data
Fraud datasets typically contain very few fraud cases.

To address this:
- Used scale_pos_weight in XGBoost to give more importance to fraud cases

### 4. Model Development
- Built a machine learning model using XGBoost
- Applied binary classification (fraud vs non-fraud)
- Optimized the model using AUC

### 5. Model Evaluation
- Split the data into training and testing sets (70/30)
- Generated predictions using probability thresholds
- Evaluated performance using a confusion matrix

---

## Results
The model showed strong performance in identifying fraudulent transactions:

- Strong classification performance on test data
- Ability to detect fraud cases effectively
- Good handling of imbalanced data

Feature importance analysis showed that variables such as transaction amount, balances, and transaction type were important in predicting fraud.

---

## Technologies Used
- R
- data.table
- dplyr
- xgboost
- caret

---

## Project Structure

```
fraud-detection-ml-r/
│
├── data/
├── scripts/
│   ├── 01_data_loading.R
│   ├── 02_feature_engineering.R
│   ├── 03_model_xgboost.R
│   ├── 04_evaluation.R
│
├── outputs/
└── README.md
```

---

## Key Insights
- Feature engineering is critical for fraud detection
- Handling imbalanced data improves model performance
- XGBoost performs well on structured financial data

---

## Skills Demonstrated
- Machine Learning (Classification)
- Feature Engineering
- Imbalanced Data Handling
- Data Preprocessing in R
- Model Evaluation

---

## Future Improvements
- Apply SMOTE for better class balancing
- Tune model parameters
- Add ROC and Precision-Recall curves
- Deploy the model for real-world use

---

## Author
Nkonzenhle Khumalo

---

## Notes
This project demonstrates a practical approach to fraud detection using machine learning, focusing on building a model that can detect suspicious financial activity effectively.
