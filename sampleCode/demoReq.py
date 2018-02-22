import os
import json
import requests
import pandas as pd

header = {'Content-Type': 'application/json', 
                  'Accept': 'application/json'}


df = pd.read_csv('test.csv', encoding ="utf-8-sig")
df = df.head()

data = df.to_json(orient='records')

print(data)

resp = requests.post("http://10.51.239.166:8000/predict", data = json.dumps(data), headers = header)

print(resp.status_code)

print(resp.json())




