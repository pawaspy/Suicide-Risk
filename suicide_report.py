import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, HalvingGridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load the NSSP dataset
df = pd.read_csv('nssp_dataset.csv')

# Clean and process the data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Encode the gender variable
df['gender'] = df['gender'].map({'female': 0, 'male': 1})

# Create new features
df['age_in_years_since_18'] = df['age'] - 18

# Define the features and target variable
features = ['gender', 'age_in_years_since_18', 'previous_attempts']
target = 'suicidal_risk'

X = df[features]
y = df[target]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the models to use
models = [

    ('Logistic Regression', LogisticRegression(), {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2', 'elasticnet', 'none']}),
    ('Decision Tree', DecisionTreeClassifier(), {'max_depth': [3, 5, 7]}),
    ('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]})
]

# Train and evaluate each model
for name, model, params in models:
    print('Training', name)
    
    # Hyperparameter tuning using grid search and cross-validation
    grid_search = HalvingGridSearchCV(model, params, cv=5, scoring='accuracy')
    
# Define the hyperparameters to tune for each model
params = [
    {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']},
    {'max_depth': [3, 5, 7]},
    {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
]

# Train and evaluate each model
for name, model in models:
    print('Training', name)
    
    # Hyperparameter tuning using grid search and cross-validation
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')

    grid_search.fit(X_train, y_train)
    
    print('Best hyperparameters:', grid_search.best_params_)
    
    # Evaluate the model on the test set
    y_pred = grid_search.predict(X_test)
    
    # Calculate the performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print the performance metrics
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)
    
    # Generate classification report and confusion matrix
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    
    print('Confusion Matrix:')
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print('\n')
