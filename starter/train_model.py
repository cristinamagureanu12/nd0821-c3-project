# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from ml.model import train_model,compute_model_metrics
from ml.data import process_data
from ml.model import inference
from metrics import model_metrics
import numpy as np
import pandas as pd
from joblib import dump

# Add code to load in the data.
def clean_dataset(df : pd.DataFrame):
    # Drop unused columns
    df.drop(['fnlgt', 'education-num', 'capital-gain', 'capital-loss'], axis='columns', inplace=True)
    df.replace({'?' : None}, inplace=True)
    df.dropna(inplace=True)
    return df

data = pd.read_csv("data/initial/census.csv", skipinitialspace=True)
data = clean_dataset(data)
# Save the new data to a new file
data.to_csv("data/cleaned/census.csv", index=False)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

dump(lb, "model/lb.joblib")
dump(encoder, "model/encoder.joblib")

# Train and save a model.
model = train_model(X_train, y_train)

X, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,    
    encoder=encoder, label="salary", training=False, lb=lb)

pred = inference(model, X)
precision, recall, fbeta = compute_model_metrics(y_test, pred)

print('precision:', precision)
print('recall', recall)
print('fbeta_scoree', fbeta)

dump(model, "model/model.joblib")

model_metrics(model, encoder, lb, test)
