# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from starter.ml.model import inference
from starter.ml.data import process_data
from typing import Literal
from joblib import load
import numpy as np
import os    
    
if "DYNO" in os.environ and os.path.isdir(".dvc"):    
    os.system("dvc config core.no_scm true")    
    if os.system("dvc pull") != 0:    
        exit("dvc pull failed")    
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()

class User(BaseModel):
    age: int
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        'Local-gov', 'Self-emp-inc', 'Without-pay']
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    maritalStatus: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced',
        'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
        'Widowed']
    occupation: Literal[
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
        'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
        'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
        'Craft-repair', 'Protective-serv', 'Armed-Forces',
        'Priv-house-serv']
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child',
        'Unmarried', 'Other-relative']
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
        'Other']
    sex: Literal['Male', 'Female']
    hoursPerWeek: int
    nativeCountry: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']

    class Config:
        schema_extra = {
            "example": {
                "age": "23",
                "workclass": "State-gov",
                "education": "Bachelors",
                "maritalStatus": "Never-married",
                "occupation": "Adm-clerical",
                "relationshio": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "hoursPerWeek": 0,
                "nativeCountry": "United-States"
            }
        }


@app.get("/")
async def get_items():
    return {"message": "Greetings!"}

items = {}
@app.on_event("startup")
async def startup_event():
    items["model"] = load("model/model.joblib")
    items["encoder"] = load("model/encoder.joblib")
    items["lb"] = load("model/lb.joblib")


@app.post("/")
async def infer(user_data: User):
    if "model" not in items:
        model = load("model/model.joblib")
    else:
        model = items["model"]

    if "encoder" not in items:
        encoder = load("model/encoder.joblib")
    else:
        encoder = items["encoder"]

    if "lb" not in items:
        lb = load("model/lb.joblib")
    else:
        lb = items["lb"]

    array = np.array([[
                     user_data.age,
                     user_data.workclass,
                     user_data.education,
                     user_data.maritalStatus,
                     user_data.occupation,
                     user_data.relationship,
                     user_data.race,
                     user_data.sex,
                     user_data.hoursPerWeek,
                     user_data.nativeCountry
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
    return {"prediction": y}

