import pytest
import pandas as pd
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from starter.ml.model import train_model, inference
from starter.ml.data import process_data, clean_dataset
from joblib import load

@pytest.fixture
def data():
    df = pd.read_csv("data/initial/census.csv", skipinitialspace=True)
    df = clean_dataset(df)
    return df

def test_removed_columns(data):
    assert "fnlgt" not in data.columns
    assert "education-num" not in data.columns
    assert "capital-gain" not in data.columns
    assert "capital-loss" not in data.columns

def test_question_mark(data):
    assert '?' not in data.values

def test_null(data):
    assert data.shape == data.dropna().shape

def test_inference1():
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    array = np.array([[
                     40,
                     "Private",
                     "Bachelors",
                     "Married-civ-spouse",
                     "Exec-managerial",
                     "Husband",
                     "White",
                     "Male",
                     80,
                     "United-States"
                     ]])
    df_temp = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

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


    X, _, _, _ = process_data(
                df_temp,
                categorical_features=cat_features,
                encoder=encoder, lb=lb, training=False)

    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == ">50K"
