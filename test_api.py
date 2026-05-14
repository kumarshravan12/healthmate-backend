import urllib.request
import json

url = "http://127.0.0.1:5000/predict"

def test_api(query):
    data = json.dumps({"symptoms": query}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as f:
        return json.loads(f.read().decode('utf-8'))

print("--- Symptom Test ---")
print(test_api("itching skin rash nodal skin eruptions"))

print("\n--- MedQuAD Knowledge Test ---")
print(test_api("What are the symptoms of Lymphocytic Choriomeningitis (LCM)?"))
