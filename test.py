import requests

# Replace with your server URL
BASE_URL = "http://localhost:8000"

# Example query to ask
query = "Hi, how are you?"

response = requests.post(f"{BASE_URL}/ask", json={"query": query})

if response.status_code == 200:
    data = response.json()
    print("Answer:", data.get("answer"))
else:
    print("Error:", response.status_code, response.text)
