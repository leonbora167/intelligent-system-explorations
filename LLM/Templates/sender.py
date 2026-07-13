import requests 

payload = {"user_prompt" : "Tell me in few words what are you ?" }


url = "http://localhost:8000/inference"
response = requests.post(url, 
                         json=payload)

print(response.json())