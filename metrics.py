import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics

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

def model_metrics(model, encoder, lb, df):

    _, test = train_test_split(df, test_size=0.20)
    slices = []

    for cat in cat_features:
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=cat_features,
                label="salary", encoder=encoder, lb=lb, training=False)

            y_preds = trained_model.predict(X_test)
            prc, rcl, fb = compute_model_metrics(y_test, y_preds)

            line = "[%s->%s] Precision: %s " \
                   "Recall: %s FBeta: %s" % (cat, cls, prc, rcl, fb)

            slices.append(line)

    # Write data
    with open('slice_output.txt', 'w') as out:
        for slice_value in slices:
            out.write(slice_value + '\n')


trained_model = load("model/model.joblib")
encoder = load("model/encoder.joblib")
lb = load("model/lb.joblib")
df = pd.read_csv("data/cleaned/census.csv")

model_metrics(trained_model, encoder, lb, df)
