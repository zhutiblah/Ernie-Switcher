import requests

url = "http://localhost:8000/predict"
payload = {"sequence": "AACCAAACACACAAACGCACAAUUUUUUUUCUUUCCACUUGUAUUUUCUUAACAGAGGAGAAAGAAAAUGCAAGUGGAAAACCUGGCGGCAGCGCAAAAGAUGCGUAAAGGAGAA"}  # 替换成你真实的 115bp 序列
response = requests.post(url, json=payload)

print(response.json())
