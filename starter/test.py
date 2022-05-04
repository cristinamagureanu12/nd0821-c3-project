import pandas as pd
import pytest
from train_model import clean_dataset

@pytest.fixture
def data():
    df = pd.read_csv("./data/initial/census.csv", skipinitialspace=True)
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

