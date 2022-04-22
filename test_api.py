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
