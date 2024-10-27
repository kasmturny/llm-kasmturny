import requests
import json

url = "https://api.chatanywhere.com.cn/v1/embeddings"

payload = json.dumps({
   "model": "text-embedding-ada-002",
   "input": "The food was delicious and the waiter..."
})
headers = {
   'Authorization': 'Bearer sk-nzdAcEUHjtnx9Ls9Yp0LlmTIYG7LfPZEof76uL5HmoNrmsPZ',
   'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
   'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)