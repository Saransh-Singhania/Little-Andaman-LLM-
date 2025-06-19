import requests

url = "http://localhost:8000/generate"
payload = {
    "prompt": "What is Little Andaman?",
    "top_k": 3
}
response = requests.post(url, json=payload)
print(response.json())