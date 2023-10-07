# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load the training and test datasets
df_train = pd.read_csv("train.csv", encoding='UTF-8')
df_test = pd.read_csv("test.csv", encoding='UTF-8')

# Combine the two datasets into a single one
df_train['which_data'] = 'data_train'
df_test['which_data'] = 'data_test'
df_all = pd.concat([df_train, df_test], axis=0)

# Handle missing values
df_all['ApplicantIncome'].fillna(df_all['ApplicantIncome'].median(), inplace=True)
df_all['CoapplicantIncome'].fillna(df_all['CoapplicantIncome'].median(), inplace=True)
df_all['LoanAmount'].fillna(df_all['LoanAmount'].median(), inplace=True)
df_all['Loan_Amount_Term'].fillna(df_all['Loan_Amount_Term'].median(), inplace=True)
df_all['Credit_History'].fillna(df_all['Credit_History'].median(), inplace=True)

# Handle zero values
df_all = df_all.mask(df_all == 0).fillna(df_all.mean())

# Drop rows with missing values
df_all = df_all.dropna()

# Create dummy variables for categorical columns
df_all = pd.get_dummies(df_all, columns=['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
                                         'Self_Employed', 'Property_Area', 'Loan_Status', 'which_data'],
                        drop_first=True)

# Data exploration and visualization (you can add more visualizations as needed)
sns.pairplot(df_all)

# Data preprocessing
df_all = df_all.select_dtypes(include=['float64', 'int64'])
y = df_all["Loan_Status"]
X = df_all.drop("Loan_Status", axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize machine learning models
models = {
    'Random Forest Regressor': RandomForestRegressor(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'XGBoost Classifier': xgb.XGBClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'SVC': SVC(kernel='linear')
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    if 'Regressor' in name:
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f'{name} RMSE: {rmse:.4f}')
    else:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{name} Accuracy: {accuracy:.4f}')

# Visualization of confusion matrix for the classifier models
classifiers = [model_name for model_name in models if 'Regressor' not in model_name]
for classifier_name in classifiers:
    model = models[classifier_name]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.show()
