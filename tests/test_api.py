"""
Api servermodule test
"""
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    api_client = TestClient(app)
    return api_client


def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Greetings!"}


def test_get_malformed(client):
    r = client.get("/bad_url")
    assert r.status_code != 200


def test_below_50k(client):
    r = client.post("/", json={
        "age": 29,
        "workclass": "Local-gov",
        "education": "Some-college",
        "maritalStatus": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_over_50k(client):
    r = client.post("/", json={
        "age": 49,
        "workclass": "Private",
        "education": "Bachelors",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}
