import requests
import json

# Definir URL e credenciais
url = 'https://api.mooring.technomar.com.br/rest/calc/environmentalCondition/forecast/wave/water/weather/range/station/20001?startDate=2024-01-27T23:30:00.000Z&endDate=2024-10-31T23:30:00.000Z'

usr = 'service.ukc@technomar.com.br'
pss = 'ukc@2023'

# Obter Token
token_url = 'https://api.accounts.technomar.com.br/rest/authenticator/token/password-grant'
token_data = {
    'grant_type': 'password',
    'client_id': '62ad5c760d3d743765242ac9',
    'client_secret': 'TZ0BZCNQ5rR2U966Wocit0TgI0FkReuVV1yRygsNLKg=',
    'username': usr,
    'password': pss,
    'scope': '',
    'redirect_uri': ''
}

# Enviar solicitação de token
response = requests.post(token_url, json=token_data)
token = response.json().get('access_token')

# Verificar se o token foi obtido
if not token:
    raise Exception("Erro ao obter o token de autenticação")

# Definir cabeçalhos da requisição
headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}

# Enviar requisição à API
response = requests.get(url, headers=headers)

# Verificar se a solicitação foi bem-sucedida
if response.status_code == 200:
    forecast = response.json()
    
    # Salvar o JSON em um arquivo
    with open('forecast_2.json', 'w') as file:
        json.dump(forecast, file, indent=2)
    print("Arquivo 'forecast.json' salvo com sucesso.")
else:
    raise Exception(f"Erro na requisição: {response.status_code}")
