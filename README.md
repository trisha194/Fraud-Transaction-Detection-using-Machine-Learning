# Fraud Transaction Detection using Machine Learning

This project involves building a machine learning model to predict fraudulent transactions for a financial company. The dataset provided contains over 6 million transactions with 10 features. The task is to develop a fraud detection model and use insights from the model to build an actionable plan to prevent future fraud.

## Table of Contents
- [Business Context](#business-context)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Data Preprocessing](#data-preprocessing)
- [Using Vaex for Large Dataset](#using-vaex-for-large-dataset)
- [Model Selection](#model-selection)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Insights & Future Work](#insights--future-work)

## Business Context
The primary goal of this project is to detect fraudulent transactions and help the company develop proactive fraud prevention strategies. Fraudulent activities cost companies millions of dollars every year, and a good predictive model can help detect these anomalies before they cause significant damage.

## Dataset
- **Source**: Financial transaction dataset (CSV format).
- **Size**: 6,362,620 rows and 10 columns.
- **Features**:
  - `step`: Time step of the transaction.
  - `type`: Type of transaction (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER).
  - `amount`: Transaction amount.
  - `oldbalanceOrg`: Balance of the sender before the transaction.
  - `newbalanceOrig`: Balance of the sender after the transaction.
  - `oldbalanceDest`: Balance of the recipient before the transaction.
  - `newbalanceDest`: Balance of the recipient after the transaction.
  - `isFraud`: Target variable (1 for fraud, 0 for non-fraud).
  - `isFlaggedFraud`: Whether the transaction was flagged as fraud.
  - 
The dataset is too large to be included directly in this repository. You can download it from the following link:

[Download Dataset from Google Drive](https://drive.google.com/file/d/1g6TVJL67AYVF4rSWQjrrQizHoQXmtVVW/view?usp=sharing)

 
    
The dataset is too large to be included in this repository. You can download it from the following link:

[Download Dataset from Google Drive]([https://drive.google.com/file/d/1g6TVJL67AYVF4rSWQjrrQizHoQXmtVVW/view?usp=sharing])

## Project Workflow
1. **Data Preprocessing**: 
   - Handle missing values.
   - One-hot encoding of categorical features.
2. **Model Selection**: 
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Support Vector Machine (SVM)
3. **Evaluation**:
   - Accuracy, Precision, Recall, and F1 Score are used to assess model performance.
4. **Insights & Recommendations**:
   - Identify key factors predicting fraud and develop prevention strategies.

## Data Preprocessing
The dataset contained missing values in critical columns like `isFraud`, `oldbalanceDest`, and `newbalanceDest`. These were handled by removing the affected rows. Additionally, the categorical column `type` was one-hot encoded to prepare it for model input.

## Using Vaex for Large Dataset
Given the size of the dataset (over 6 million rows), this project utilized **Vaex**, a high-performance DataFrame library for Python that enables out-of-core DataFrame operations. Vaex efficiently handles large datasets by performing computations without loading all the data into memory, making it ideal for this fraud detection project.

### Pros of Using Vaex:
- **Memory Efficiency**: Vaex performs calculations on data that doesn’t fit into memory by loading only the necessary chunks, avoiding memory overload.
- **Speed**: Vaex provides faster execution for typical data manipulation operations such as filtering, grouping, and aggregation.
- **Lazy Execution**: It delays computation until it's required, optimizing performance for large datasets.
- **Parallelization**: Vaex supports multi-threading and can leverage modern multi-core processors to accelerate data processing.

By using Vaex, I was able to work efficiently with such a large dataset without compromising on speed or memory performance.

## Model Selection
Multiple machine learning models were used to detect fraud:
- **Logistic Regression**: Simple baseline model.
- **Random Forest**: Powerful ensemble method known for handling large datasets.
- **XGBoost**: Gradient boosting algorithm, often used for tabular data with high performance.
- **SVM**: Support Vector Machine, though it showed poor performance in this case.

## Evaluation Metrics
The models were evaluated using the following metrics:
- **Accuracy**: Percentage of correct predictions.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall**: Proportion of actual fraud cases that were identified.
- **F1 Score**: Harmonic mean of precision and recall, which balances both metrics.

| Model                 | Accuracy  | Precision | Recall | F1 Score |
|-----------------------|-----------|-----------|--------|----------|
| Logistic Regression    | 0.998     | 0.94      | 0.15   | 0.26     |
| Random Forest          | 0.999     | 1.00      | 0.25   | 0.40     |
| XGBoost                | 0.999     | 0.94      | 0.33   | 0.49     |
| SVM                    | 0.998     | 1.00      | 0.02   | 0.04     |

## Results
- **XGBoost** had the highest F1 score, making it the most suitable model for this task.
- **Random Forest** performed well but slightly lower in recall compared to XGBoost.
- **SVM** showed poor recall, making it unsuitable for fraud detection in this case.

## Insights & Future Work
- **Key Factors Predicting Fraud**: 
  - Transaction types like `TRANSFER` and `CASH_OUT` showed strong correlations with fraud.
  - Larger transaction amounts were more likely to be fraudulent.
- **Recommendations**:
  - Implement multi-level authentication for high-value transactions.
  - Monitor transactions with frequent `TRANSFER` and `CASH_OUT` types.
  - Enhance real-time fraud detection infrastructure with XGBoost or Random Forest models.
  
