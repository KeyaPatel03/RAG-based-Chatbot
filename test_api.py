import requests
import json
import time

url = "http://localhost:8000/chat"
query = "What courses are available?"

print(f"Testing API at {url} with query: '{query}'")

max_retries = 10
for i in range(max_retries):
    try:
        response = requests.post(url, json={"query": query})
        
        if response.status_code == 200:
            print("Response Status: 200 OK")
            data = response.json()
            print("Answer:", data.get("answer"))
            print("Sources:", data.get("sources"))
            break
        else:
            print(f"Failed with status code: {response.status_code}")
            print(response.text)
            break
            
    except requests.exceptions.ConnectionError:
        print(f"Connection failed, retrying in 2 seconds... ({i+1}/{max_retries})")
        time.sleep(2)
else:
    print("Could not connect to the API after multiple attempts.")
    
