# ChurnShield AI

An end-to-end **Customer Churn Prediction System** using Machine Learning and Flask.

## Overview

ChurnShield AI predicts the probability of customer churn and classifies customers into risk levels through an interactive web interface.

## Features

- Customer churn prediction
- Churn probability
- Low / Medium / High / Very High risk levels
- Interactive dashboard
- Prediction history
- Input validation
- Optimized ML model
- Flask-based web application

## Machine Learning

**Final Model:** HistGradientBoostingClassifier

**Performance:**
- Accuracy: 85.50%
- Precision: 64.89%
- Recall: 62.65%
- F1 Score: 63.75%
- ROC-AUC: 86.76%

**Decision Threshold:** 0.35

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Flask
- HTML
- CSS
- JavaScript

## Project Structure

```text
Customer_Churn_Prediction/
├── dataset/
├── model/
├── notebooks/
├── static/
├── templates/
├── app.py
├── train.py
├── requirements.txt
└── README.md
