import requests

data = {
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
    }

r = requests.post('https://crisndapp.herokuapp.com', json=data)

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
