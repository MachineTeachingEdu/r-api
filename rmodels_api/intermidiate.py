import requests

headers = {
    'accept': '*/*',
    'Content-Type': 'application/json',
}

json_data = {
    'data_list': [[19, 51, 16, 3, 4, 7, 4, 1, 1, 1]]
}
response = requests.post('http://127.0.0.1:8000/api/predict/', headers=headers, json=json_data)

print(response.json())


json_data = {
    'user': 1,
    'risk': response.json()['prediction'],
}

print(json_data)

x = requests.post('http://localhost:8020/pt-br/api/v1/dropoutRisk/', json=json_data, headers=headers)

print(x.json())