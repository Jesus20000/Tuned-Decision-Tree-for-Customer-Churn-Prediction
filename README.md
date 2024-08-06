# Tuned Decision Tree for Customer Churn Prediction

In this project, I explored and cleaned a dataset to predict customer churn using a tuned decision tree model. The dataset includes various features such as credit score, geography, age, balance, and more.

## Overview
The goal of this project is to build a decision tree model to predict whether a customer will churn based on their profile and account information. We applied hyperparameter tuning to optimize the model's performance and evaluated its effectiveness using various metrics.

## Libraries Used
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib`: For data visualization.
- `scikit-learn`: For machine learning and model evaluation.

## Data
The dataset used is `Churn_Modelling.csv` and includes the following columns:
- `RowNumber`: Index of the row.
- `CustomerId`: Unique identifier for the customer.
- `Surname`: Surname of the customer.
- `CreditScore`: Credit score of the customer.
- `Geography`: Customer's location.
- `Gender`: Gender of the customer.
- `Age`: Age of the customer.
- `Tenure`: Number of years the customer has been with the bank.
- `Balance`: Account balance of the customer.
- `NumOfProducts`: Number of products the customer has with the bank.
- `HasCrCard`: Whether the customer has a credit card.
- `IsActiveMember`: Whether the customer is an active member.
- `EstimatedSalary`: Estimated salary of the customer.
- `Exited`: Target variable indicating whether the customer has churned (1) or not (0).

## Analysis Steps
1. **Data Exploration**: Load and inspect the dataset.
2. **Preprocessing**: Clean the data by dropping irrelevant columns and handling missing values.
3. **Feature Selection**: Define and select relevant features for the model.
4. **Model Training**: Train a decision tree model with hyperparameter tuning using GridSearchCV.
5. **Model Evaluation**: Evaluate the model using various metrics such as accuracy, precision, recall, and F1 score.

## Insights
- **Best Hyperparameters**: {'classifier__max_depth': 10, 'classifier__min_samples_leaf': 20}
- **Model Performance**:
  - Accuracy: 0.858
  - Precision: 0.723
  - Recall: 0.487
  - F1 Score: 0.582

## Conclusion
The tuned decision tree model shows promise in predicting customer churn, with the F1 score indicating a balance between precision and recall. Further improvements can be made by exploring more advanced models or additional feature engineering.

## Requirements
To run this project, you need the following Python libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

## Usage

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/tuned-decision-tree-for-customer-churn-prediction.git
    ```
2. **Navigate into the Project Directory**:
    ```bash
    cd tuned-decision-tree-for-customer-churn-prediction
    ```
3. **Install the Required Libraries**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```
4. **Run the Jupyter Notebook**:
    Launch Jupyter Notebook and open the `Decision tree model with Python churn.ipynb` file to start the analysis.
    ```bash
    jupyter notebook churn.ipynb
    ```
