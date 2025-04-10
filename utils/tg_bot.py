import requests
from kaggle_secrets import UserSecretsClient

def get_tokens():
    secrets = UserSecretsClient()
    TELEGRAM_TOKEN = secrets.get_secret('TELEGRAM_TOKEN')
    GEORGY_ID = secrets.get_secret('GEORGY_ID')
    DANIL_ID = secrets.get_secret('DANIL_ID')
    return TELEGRAM_TOKEN, CHAT_ID

TELEGRAM_TOKEN, CHAT_ID = get_tokens()

# --- 2. Отправляем уведомление и файл в Telegram ---
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": text}
    requests.post(url, params=params)

def send_telegram_file(file_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    with open(file_path, 'rb') as f:
        files = {"document": f}
        data = {"chat_id": CHAT_ID}
        requests.post(url, files=files, data=data)
