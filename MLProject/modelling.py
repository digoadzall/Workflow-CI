import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Parsing argument
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='clean_telco.csv')
parser.add_argument('--alpha', type=float, default=0.5)  
args = parser.parse_args()

# Load dataset
data = pd.read_csv(args.data_path)

# Optional: hindari warning schema enforcement dari mlflow
for col in data.select_dtypes(include='int').columns:
    data[col] = data[col].astype('float64')

X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.sklearn.autolog()

with mlflow.start_run():
    # Logistic Regression dengan regularisasi L2 (default)
    # alpha → regularization strength → C = 1 / alpha
    model = LogisticRegression(C=1.0 / args.alpha, solver='liblinear')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")
