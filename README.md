# Machine-Learning


**Heart Failure Prediction**
This project focuses on predicting heart failure based on various clinical features using machine learning algorithms. Below is a detailed overview of the project's process and findings:

**Summary:**

**_1. Data Exploration (EDA)_**
Explored the dataset through various visualizations such as histograms, pairplots, boxplots, and countplots.
Analyzed the distribution of features like age, sex, chest pain type, resting blood pressure, cholesterol levels, etc.
Identified outliers and dealt with them appropriately.

**_2. Preprocessing:_**
Checked for missing values (there were none).
Converted categorical variables into numerical format using label encoding.
Split the data into features (X) and target (y) variables.
Split the dataset into training and testing sets.

**_3. Modeling:_**
Trained three different classifiers:
Logistic Regression
Decision Tree Classifier
Naive Bayes
Evaluated the models using accuracy score and classification report.


**1.1 : IMPORTING LIBRARIES**
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


**1.2 : LOADING THE DATASET**
df = pd.read_csv('heart.csv')
df.head()


**1.3 : BASIC INFORMATION OF THE DATASET**
df.info()
df.shape
df.columns
df.isna().sum()
df.describe()
df.nunique()


**1.4 : CHECK NULL VALUES**
No NULL values found.


**1.5 : STATISTICAL INFORMATION OF THE DATASET**
Descriptive statistics including mean, min, max, etc. were calculated for numerical features.


**1.6 : NUMBER OF UNIQUE VALUES IN COLUMNS**
Count of unique values in each column was calculated.


**1.7 : EDA(Exploratory Data Analysis)**
Exploratory data analysis was conducted using seaborn for various visualizations like histplot, pairplot, boxplot, and countplot.


**1.8 : PREPROCESSING (LABEL-ENCODING)**
Categorical data was converted into numerical format using label encoding.


**1.9 : SEPARATING INTO FEATURES AND TARGET COLUMNS**
Data was separated into features (X) and target (y) columns.


**1.10 : SPLITTING THE DATA INTO TRAINING AND TESTING**
The dataset was split into training and testing sets using train_test_split.


**1.11 : PREDICTING BY USING LOGISTIC REGRESSION**
Logistic regression model was trained and evaluated achieving an accuracy score of approximately 86%.


**1.12 : PREDICTING BY USING DECISION TREE CLASSIFIER**
Decision tree classifier was trained and evaluated achieving an accuracy score of approximately 82%.


**1.12 : NAIVE BAYES**
Naive Bayes classifier was trained and evaluated achieving an accuracy score of approximately 87%.


**Conclusion :**
_Logistic Regression Model achieved an accuracy score of approximately 86%.
Decision Tree Classifier achieved an accuracy score of approximately 82%.
Naive Bayes achieved an accuracy score of approximately 87%.
Naive Bayes performed slightly better than logistic regression and decision tree in terms of accuracy.
Further optimization of models could potentially improve performance.
It's crucial to consider domain knowledge and consult with healthcare professionals to enhance the predictive power of the models and ensure their practical applicability._
